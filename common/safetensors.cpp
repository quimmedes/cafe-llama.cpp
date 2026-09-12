#include "common.h"

#include "log.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"

#include "../src/llama-arch.h"
#include "../src/llama-model-saver.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <vector>

// Load a Hugging Face safetensors checkpoint without converting it to GGUF:
// the checkpoint metadata is turned into an in-memory gguf_context and the weights are read
// from the shards while the model buffers are filled. Only the text model is loaded, the
// vision tower of multimodal checkpoints is skipped.

namespace fs = std::filesystem;
using json = nlohmann::json;

// FP8 weights use per 128x128 block scales
static constexpr int64_t ST_BLOCK = 128;

static const float * st_e4m3_table() {
    static const std::array<float, 256> table = [] {
        std::array<float, 256> res {};
        for (int i = 0; i < 256; ++i) {
            const int s = (i >> 7) & 1;
            const int e = (i >> 3) & 15;
            const int m = i & 7;
            const float v = e == 0 ? std::ldexp(m, -9) : std::ldexp(1.0f + m / 8.0f, e - 7);
            res[i] = s ? -v : v;
        }
        return res;
    }();
    return table.data();
}

static float st_bf16_to_f32(uint16_t v) {
    const uint32_t bits = (uint32_t) v << 16;
    float res;
    memcpy(&res, &bits, sizeof(res));
    return res;
}

static uint16_t st_f32_to_bf16(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    bits += 0x7fff + ((bits >> 16) & 1); // round to nearest
    return (uint16_t) (bits >> 16);
}

enum st_dtype {
    ST_DT_F32,
    ST_DT_F16,
    ST_DT_BF16,
    ST_DT_F8,
};

static int st_dtype_size(enum st_dtype t) {
    return t == ST_DT_F8 ? 1 : (t == ST_DT_F32 ? 4 : 2);
}

static enum st_dtype st_dtype_from_name(const std::string & name) {
    if (name == "F32") {
        return ST_DT_F32;
    }
    if (name == "F16") {
        return ST_DT_F16;
    }
    if (name == "BF16") {
        return ST_DT_BF16;
    }
    if (name == "F8_E4M3") {
        return ST_DT_F8;
    }
    throw std::runtime_error("unsupported safetensors dtype: " + name);
}

static enum ggml_type st_ggml_type(enum st_dtype t) {
    switch (t) {
        case ST_DT_F32:  return GGML_TYPE_F32;
        case ST_DT_F16:  return GGML_TYPE_F16;
        case ST_DT_BF16: return GGML_TYPE_BF16;
        default:         throw std::runtime_error("unsupported source dtype");
    }
}

struct st_shard {
    int         idx = 0; // index in the file list handed to the loader
    std::string name;
    fs::path    path;
    FILE *      file = nullptr;
    uint64_t    data_offs = 0;

    ~st_shard() {
        if (file != nullptr) {
            fclose(file);
        }
    }

    void read(uint64_t offs, void * dst, size_t size) const {
        if (fseek(file, (long) (data_offs + offs), SEEK_SET) != 0) {
            throw std::runtime_error(string_format("seek failed in '%s' at %zu", name.c_str(), (size_t) offs));
        }
        if (size > 0 && fread(dst, 1, size, file) != size) {
            throw std::runtime_error(string_format("short read in '%s' at %zu (%zu bytes)", name.c_str(), (size_t) offs, size));
        }
    }
};

// one tensor inside a shard, described by the safetensors header
struct st_ref {
    const st_shard * shard = nullptr;
    uint64_t         offs  = 0;
    int64_t          ne[6] = {1, 1, 1, 1, 1, 1}; // as stored: [rows, cols, ...]
    int              ndim  = 1;
    enum st_dtype    dtype = ST_DT_F32;

    int64_t nrows() const { return ndim >= 2 ? ne[0] : 1; }
    int64_t ncols() const { return ndim >= 2 ? ne[1] : ne[0]; }

    uint64_t nbytes() const {
        int64_t n = 1;
        for (int i = 0; i < ndim; ++i) {
            n *= ne[i];
        }
        return (uint64_t) n * st_dtype_size(dtype);
    }
};

static st_ref st_make_ref(const st_shard * shard, uint64_t offs, const std::vector<int64_t> & shape, enum st_dtype dtype) {
    st_ref res;
    res.shard = shard;
    res.offs  = offs;
    res.dtype = dtype;

    std::vector<int64_t> dims = shape;
    while (dims.size() > 1 && dims[0] == 1) {
        dims.erase(dims.begin());
    }
    if (dims.size() == 3 && dims[1] == 1) {
        dims.erase(dims.begin() + 1);
    }

    res.ndim = (int) dims.size();
    for (size_t i = 0; i < dims.size() && i < 6; ++i) {
        res.ne[i] = dims[i];
    }
    return res;
}

// permute V heads from grouped (by K head) to tiled order, as expected by ggml
struct st_reorder {
    int     dim   = -1; // -1: none, 0: rows, 1: cols
    int64_t offs  = 0;
    int64_t count = 0;
    int64_t head  = 1;
};

enum st_role {
    ST_ROLE_WEIGHT, // fp8 weights are stored as the dequantization type
    ST_ROLE_F32,    // activations/norms are kept in fp32
};

// a checkpoint can be loaded as the text model or as a standalone MTP head
enum st_mode {
    ST_MODE_MODEL,
    ST_MODE_MTP,
};

struct st_plan {
    std::string   name; // ggml tensor name
    enum ggml_type type = GGML_TYPE_BF16;
    int64_t       ne[4] = {1, 1, 1, 1};
    int           ndim  = 1;
    int           add_one = 0;  // +1, used by Qwen RMSNorm weights
    int           neg_exp = 0;  // -exp, used by A_log
    int64_t       src_row_stride = 0; // elements between two source rows, 0 means ncols
    int64_t       emit_nrows = 0;     // rows of the emitted matrix, 0 means derived from ne
    int64_t       emit_ncols = 0;
    st_reorder    reorder;
    st_ref        src;
    st_ref        scale;        // fp8 block scales
    int           has_scale = 0;
    std::vector<st_ref> exps;       // per-expert weights, when non-empty
    std::vector<st_ref> exps_scale; // per-expert scales
    int64_t       n_expert = 0;
    int           verbatim = 0;     // data is stored as-is in the shard and read directly by the loader
};

struct st_loader {
    std::vector<std::unique_ptr<st_shard>> shards;
    std::unordered_map<std::string, const st_plan *> plans;
    std::vector<std::unique_ptr<st_plan>> storage;

    ggml_type fp8_type = GGML_TYPE_Q8_0;

    // linear attention head configuration, used for the V head permutation
    int64_t num_k_heads = 0;
    int64_t num_v_heads = 0;
    int64_t head_k_dim  = 0;
    int64_t head_v_dim  = 0;

    std::vector<uint8_t> scratch;
};

static std::string st_read_text_file(const fs::path & path) {
    std::string res;
    FILE * f = fopen(path.string().c_str(), "rb");
    if (f == nullptr) {
        throw std::runtime_error("failed to open " + path.string());
    }
    char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f) ) > 0) {
        res.append(buf, n);
    }
    fclose(f);
    return res;
}

static json st_read_json(const fs::path & path) {
    return json::parse(st_read_text_file(path));
}

// ---------------------------------------------------------------------------
// tensor mapping
// ---------------------------------------------------------------------------

enum st_reorder_kind {
    ST_RE_NONE,
    ST_RE_QKV_ROWS, // rows of the V part of qkv/conv1d
    ST_RE_Z_ROWS,   // rows of the z gate
    ST_RE_OUT_COLS, // columns of the output projection
    ST_RE_HEADS,    // rows of the per head parameters (head dim 1)
};

struct st_map {
    std::string gguf;
    enum st_role role = ST_ROLE_WEIGHT;
    int add_one = 0;
    int neg_exp = 0;
    enum st_reorder_kind reorder = ST_RE_NONE;
};

// returns false when the tensor is not part of the text model
static bool st_map_tensor(const std::string & name, st_map & res, int64_t & expert, std::string & expert_base) {

    std::string n = name;

    // global tensors
    if (n == "model.language_model.embed_tokens.weight") {
        res.gguf = "token_embd.weight";
        return true;
    }
    if (n == "model.language_model.norm.weight") {
        res.gguf = "output_norm.weight";
        res.role = ST_ROLE_F32;
        res.add_one = 1;
        return true;
    }
    if (n == "lm_head.weight") {
        res.gguf = "output.weight";
        return true;
    }
    if (n == "mtp.fc.weight") {
        res.gguf = "blk.40.nextn.eh_proj.weight";
        return true;
    }
    if (n == "mtp.pre_fc_norm_embedding.weight") {
        res.gguf = "blk.40.nextn.enorm.weight";
        res.role = ST_ROLE_F32;
        res.add_one = 1;
        return true;
    }
    if (n == "mtp.pre_fc_norm_hidden.weight") {
        res.gguf = "blk.40.nextn.hnorm.weight";
        res.role = ST_ROLE_F32;
        res.add_one = 1;
        return true;
    }
    if (n == "mtp.norm.weight") {
        res.gguf = "blk.40.nextn.shared_head_norm.weight";
        res.role = ST_ROLE_F32;
        res.add_one = 1;
        return true;
    }

    // the vision tower is not part of the text model
    if (n.rfind("model.visual.", 0) == 0 || n.rfind("visual.", 0) == 0) {
        return false;
    }

    // MTP block is exported as the last layer
    if (n.rfind("mtp.layers.0.", 0) == 0) {
        n = "model.language_model.layers.40." + n.substr(strlen("mtp.layers.0."));
    }

    static const char * prefix = "model.language_model.layers.";
    if (n.rfind(prefix, 0) != 0) {
        return false;
    }

    const size_t pos = strlen(prefix);
    const size_t dot = n.find('.', pos);
    if (dot == std::string::npos) {
        return false;
    }

    const int il = atoi(n.substr(pos, dot - pos).c_str());
    const std::string rest = n.substr(dot + 1);

    const std::string base = "blk." + std::to_string(il) + ".";

    // routed experts: stacked into a single 3d tensor
    if (rest.rfind("mlp.experts.", 0) == 0) {
        const std::string tail = rest.substr(strlen("mlp.experts."));
        const size_t p = tail.find('.');
        if (p == std::string::npos) {
            return false;
        }
        expert = atoll(tail.substr(0, p).c_str());
        const std::string proj = tail.substr(p + 1);
        if (proj == "gate_proj.weight") {
            expert_base = base + "ffn_gate_exps.weight";
        } else if (proj == "up_proj.weight") {
            expert_base = base + "ffn_up_exps.weight";
        } else if (proj == "down_proj.weight") {
            expert_base = base + "ffn_down_exps.weight";
        } else {
            return false;
        }
        return true;
    }

    struct entry {
        const char * suffix;
        const char * gguf;
        enum st_role role;
        int add_one;
        int neg_exp;
        enum st_reorder_kind reorder;
    };

    static const entry entries[] = {
        { "input_layernorm.weight",             "attn_norm.weight",           ST_ROLE_F32,    1, 0, ST_RE_NONE },
        { "post_attention_layernorm.weight",    "post_attention_norm.weight", ST_ROLE_F32,    1, 0, ST_RE_NONE },
        { "self_attn.q_proj.weight",            "attn_q.weight",              ST_ROLE_WEIGHT, 0, 0, ST_RE_NONE },
        { "self_attn.k_proj.weight",            "attn_k.weight",              ST_ROLE_WEIGHT, 0, 0, ST_RE_NONE },
        { "self_attn.v_proj.weight",            "attn_v.weight",              ST_ROLE_WEIGHT, 0, 0, ST_RE_NONE },
        { "self_attn.o_proj.weight",            "attn_output.weight",         ST_ROLE_WEIGHT, 0, 0, ST_RE_NONE },
        { "self_attn.q_norm.weight",            "attn_q_norm.weight",         ST_ROLE_F32,    1, 0, ST_RE_NONE },
        { "self_attn.k_norm.weight",            "attn_k_norm.weight",         ST_ROLE_F32,    1, 0, ST_RE_NONE },
        { "linear_attn.in_proj_qkv.weight",     "attn_qkv.weight",            ST_ROLE_WEIGHT, 0, 0, ST_RE_QKV_ROWS },
        { "linear_attn.in_proj_z.weight",       "attn_gate.weight",           ST_ROLE_WEIGHT, 0, 0, ST_RE_Z_ROWS },
        { "linear_attn.conv1d.weight",          "ssm_conv1d.weight",          ST_ROLE_F32,    0, 0, ST_RE_QKV_ROWS },
        { "linear_attn.dt_bias",                "ssm_dt.bias",                ST_ROLE_F32,    0, 0, ST_RE_HEADS },
        { "linear_attn.A_log",                  "ssm_a",                      ST_ROLE_F32,    0, 1, ST_RE_HEADS },
        { "linear_attn.in_proj_b.weight",       "ssm_beta.weight",            ST_ROLE_WEIGHT, 0, 0, ST_RE_HEADS },
        { "linear_attn.in_proj_a.weight",       "ssm_alpha.weight",           ST_ROLE_WEIGHT, 0, 0, ST_RE_HEADS },
        { "linear_attn.norm.weight",            "ssm_norm.weight",            ST_ROLE_F32,    0, 0, ST_RE_NONE },
        { "linear_attn.out_proj.weight",        "ssm_out.weight",             ST_ROLE_WEIGHT, 0, 0, ST_RE_OUT_COLS },
        { "mlp.gate.weight",                    "ffn_gate_inp.weight",        ST_ROLE_F32,    0, 0, ST_RE_NONE },
        { "mlp.shared_expert_gate.weight",      "ffn_gate_inp_shexp.weight",  ST_ROLE_F32,    0, 0, ST_RE_NONE },
        { "mlp.shared_expert.gate_proj.weight", "ffn_gate_shexp.weight",      ST_ROLE_WEIGHT, 0, 0, ST_RE_NONE },
        { "mlp.shared_expert.up_proj.weight",   "ffn_up_shexp.weight",        ST_ROLE_WEIGHT, 0, 0, ST_RE_NONE },
        { "mlp.shared_expert.down_proj.weight", "ffn_down_shexp.weight",      ST_ROLE_WEIGHT, 0, 0, ST_RE_NONE },
    };

    for (const auto & e : entries) {
        if (rest == e.suffix) {
            res.gguf = base + e.gguf;
            res.role = e.role;
            res.add_one = e.add_one;
            res.neg_exp = e.neg_exp;
            res.reorder = e.reorder;
            return true;
        }
    }

    return false;
}

// ---------------------------------------------------------------------------
// metadata
// ---------------------------------------------------------------------------

static void st_add_meta_arch(st_loader & L, gguf_context * meta, const json & cfg, const std::string & dir_name) {
    const json & tc = cfg.at("text_config");

    auto get_u32 = [&](const char * key) {
        return tc.at(key).get<uint32_t>();
    };

    const uint32_t n_layer  = get_u32("num_hidden_layers");
    const uint32_t n_nextn  = tc.value("mtp_num_hidden_layers", 0u);
    const uint32_t n_embd   = get_u32("hidden_size");
    const uint32_t n_head   = get_u32("num_attention_heads");
    const uint32_t head_dim = get_u32("head_dim");
    const uint32_t n_expert = get_u32("num_experts");
    const uint32_t n_ff_exp = get_u32("moe_intermediate_size");
    const uint32_t n_ff_sh  = get_u32("shared_expert_intermediate_size");

    const uint32_t head_k_dim = get_u32("linear_key_head_dim");
    const uint32_t head_v_dim = get_u32("linear_value_head_dim");
    const uint32_t n_k_heads  = get_u32("linear_num_key_heads");
    const uint32_t n_v_heads  = get_u32("linear_num_value_heads");

    L.num_k_heads = n_k_heads;
    L.num_v_heads = n_v_heads;
    L.head_k_dim  = head_k_dim;
    L.head_v_dim  = head_v_dim;

    // mrope sections: [11, 11, 10, 0] unless the checkpoint specifies its own
    std::vector<int32_t> rope_sections = { 11, 11, 10, 0 };
    if (tc.contains("rope_parameters") && tc.at("rope_parameters").contains("mrope_section")) {
        rope_sections = tc.at("rope_parameters").at("mrope_section").get<std::vector<int32_t>>();
        rope_sections.resize(4, 0);
    }

    float rope_theta = 1000000.0f;
    if (tc.contains("rope_parameters") && tc.at("rope_parameters").contains("rope_theta")) {
        rope_theta = tc.at("rope_parameters").at("rope_theta").get<float>();
    }

    const float partial_rot = tc.value("partial_rotary_factor", 0.25f);

    std::vector<uint32_t> recurrent;
    recurrent.reserve(n_layer + n_nextn);
    for (const auto & t : tc.at("layer_types")) {
        recurrent.push_back(t.get<std::string>() == "linear_attention" ? 1 : 0);
    }
    recurrent.resize(n_layer + n_nextn, 0); // MTP layers are attention-only

    llama_model_saver ms(LLM_ARCH_QWEN35MOE, meta);
    const LLM_KV kv(LLM_ARCH_QWEN35MOE);

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE, "qwen35moe");
    ms.add_kv(LLM_KV_GENERAL_NAME,         dir_name.c_str());

    ms.add_kv(LLM_KV_BLOCK_COUNT,              n_layer + n_nextn);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,           get_u32("max_position_embeddings"));
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,         n_embd);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,     n_head);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,  get_u32("num_key_value_heads"));
    ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,     head_dim);
    ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,   head_dim);
    ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,     (uint32_t) (head_dim * partial_rot));
    ms.add_kv(LLM_KV_ROPE_FREQ_BASE,           rope_theta);
    gguf_set_arr_data(meta, kv(LLM_KV_ROPE_DIMENSION_SECTIONS).c_str(), GGUF_TYPE_INT32, rope_sections.data(), rope_sections.size());
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, tc.at("rms_norm_eps").get<float>());

    ms.add_kv(LLM_KV_EXPERT_COUNT,               n_expert);
    ms.add_kv(LLM_KV_EXPERT_USED_COUNT,          get_u32("num_experts_per_tok"));
    ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, n_ff_exp);
    ms.add_kv(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, n_ff_sh);
    ms.add_kv(LLM_KV_NEXTN_PREDICT_LAYERS,       n_nextn);

    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,    get_u32("linear_conv_kernel_dim"));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,     head_k_dim);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,    n_k_heads);
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK, n_v_heads);
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,     head_v_dim * n_v_heads);
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL, get_u32("full_attention_interval"));

    {
        std::vector<uint8_t> recr(n_layer + n_nextn);
        for (size_t i = 0; i < recurrent.size(); ++i) {
            recr[i] = (uint8_t) recurrent[i];
        }
        gguf_set_arr_data(meta, kv(LLM_KV_ATTENTION_RECURRENT_LAYERS).c_str(), GGUF_TYPE_BOOL, recr.data(), recr.size());
    }
}

static void st_add_meta_vocab(gguf_context * meta, const fs::path & dir, const json & cfg) {
    const json & tc = cfg.at("text_config");

    const uint32_t n_vocab = tc.at("vocab_size").get<uint32_t>();

    const json vocab = st_read_json(dir / "vocab.json");
    const json tokc  = st_read_json(dir / "tokenizer.json");

    // added tokens keep their own type, base vocabulary tokens are NORMAL
    struct st_added { std::string content; bool special; };
    std::unordered_map<uint32_t, st_added> added;
    if (tokc.contains("added_tokens")) {
        for (const auto & t : tokc.at("added_tokens")) {
            const uint32_t id = t.at("id").get<uint32_t>();
            added[id] = { t.at("content").get<std::string>(), t.value("special", false) };
        }
    }

    std::vector<std::string> tokens(n_vocab);
    std::vector<int32_t>     types(n_vocab);
    std::vector<uint8_t>     present(n_vocab, 0);
    for (const auto & [tok, id] : vocab.items()) {
        if (id.get<uint32_t>() < n_vocab) {
            tokens[id.get<uint32_t>()] = tok;
            present[id.get<uint32_t>()] = 1;
        }
    }

    auto looks_special = [](const std::string & s) {
        if (s == "<pad>" || s == "<mask>" || s == "<2mass>" || s == "[@BOS@]") {
            return true;
        }
        if (s.size() >= 4 && s.compare(0, 2, "<|") == 0 && s.compare(s.size() - 2, 2, "|>") == 0) {
            return true;
        }
        if (s.size() >= 8 && s.compare(0, 7, "<unused") == 0 && s.back() == '>') {
            return true;
        }
        return false;
    };

    for (uint32_t i = 0; i < n_vocab; ++i) {
        const auto it = added.find(i);
        if (!present[i] && it == added.end()) {
            tokens[i] = "[PAD" + std::to_string(i) + "]";
            types[i]  = 5; // UNUSED
        } else if (it != added.end()) {
            tokens[i] = it->second.content;
            types[i]  = (it->second.special || looks_special(tokens[i])) ? 3 : 4; // CONTROL : USER_DEFINED
        } else {
            types[i] = 1; // NORMAL
        }
    }

    std::vector<std::string> merges;
    {
        FILE * f = fopen((dir / "merges.txt").string().c_str(), "rb");
        if (f == nullptr) {
            throw std::runtime_error("failed to open merges.txt");
        }
        char buf[4096];
        while (fgets(buf, sizeof(buf), f) != nullptr) {
            std::string line = buf;
            while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
                line.pop_back();
            }
            if (!line.empty()) {
                merges.push_back(line);
            }
        }
        fclose(f);
    }

    llama_model_saver ms(LLM_ARCH_QWEN35MOE, meta);
    const LLM_KV kv(LLM_ARCH_QWEN35MOE);

    ms.add_kv(LLM_KV_TOKENIZER_MODEL,  "gpt2");
    ms.add_kv(LLM_KV_TOKENIZER_PRE,    "qwen35");
    ms.add_kv(LLM_KV_TOKENIZER_LIST,   tokens);
    ms.add_kv(LLM_KV_TOKENIZER_MERGES, merges);
    gguf_set_arr_data(meta, kv(LLM_KV_TOKENIZER_TOKEN_TYPE).c_str(), GGUF_TYPE_INT32, types.data(), types.size());

    // special tokens are named in tokenizer_config.json and fall back to the model config
    std::unordered_map<std::string, uint32_t> token_ids;
    for (uint32_t i = 0; i < n_vocab; ++i) {
        if (!present[i] || !tokens[i].empty()) {
            token_ids[tokens[i]] = i;
        }
    }

    const json tokc_cfg = st_read_json(dir / "tokenizer_config.json");

    auto special_id = [&](const char * key, uint32_t & out) {
        out = 0;
        if (!tokc_cfg.contains(key) || tokc_cfg.at(key).is_null()) {
            return false;
        }
        const json & v = tokc_cfg.at(key);
        const std::string name = v.is_string() ? v.get<std::string>() : v.value("content", std::string());
        const auto it = token_ids.find(name);
        if (it == token_ids.end()) {
            return false;
        }
        out = it->second;
        return true;
    };

    auto config_id = [&](const char * key, uint32_t def) {
        if (!tc.contains(key) || tc.at(key).is_null()) {
            return def;
        }
        const json & v = tc.at(key);
        return v.is_array() ? (v.empty() ? def : v.at(0).get<uint32_t>()) : v.get<uint32_t>();
    };

    uint32_t eos = 0;
    uint32_t pad = 0;
    uint32_t bos = 0;
    if (!special_id("eos_token", eos)) {
        eos = config_id("eos_token_id", 0);
    }
    if (!special_id("pad_token", pad)) {
        pad = config_id("pad_token_id", eos);
    }
    if (!special_id("bos_token", bos)) {
        bos = config_id("bos_token_id", eos);
    }

    ms.add_kv(LLM_KV_TOKENIZER_BOS_ID, bos);
    ms.add_kv(LLM_KV_TOKENIZER_EOS_ID, eos);
    ms.add_kv(LLM_KV_TOKENIZER_PAD_ID, pad);
    ms.add_kv(LLM_KV_TOKENIZER_ADD_BOS, tokc_cfg.value("add_bos_token", false));

    const fs::path tmpl = dir / "chat_template.jinja";
    if (fs::is_regular_file(tmpl)) {
        ms.add_kv(LLM_KV_TOKENIZER_CHAT_TEMPLATE, st_read_text_file(tmpl).c_str());
    }
}

// ---------------------------------------------------------------------------
// tensor data
// ---------------------------------------------------------------------------

static unsigned st_threads() {
    static const unsigned n = std::max(1u, std::thread::hardware_concurrency());
    return n;
}

static size_t st_nbytes(enum ggml_type type, const int64_t ne[4], int ndim) {
    size_t size = ggml_row_size(type, ne[0]);
    for (int i = 1; i < ndim; ++i) {
        size *= ne[i];
    }
    return size;
}

static float st_read_f32(const uint8_t * buf, enum st_dtype dtype, int64_t idx) {
    switch (dtype) {
        case ST_DT_F32: {
            float v;
            memcpy(&v, buf + idx * 4, sizeof(v));
            return v;
        }
        case ST_DT_BF16: {
            uint16_t v;
            memcpy(&v, buf + idx * 2, sizeof(v));
            return st_bf16_to_f32(v);
        }
        case ST_DT_F16: {
            uint16_t v;
            memcpy(&v, buf + idx * 2, sizeof(v));
            return ggml_fp16_to_fp32(v);
        }
        default:
            return st_e4m3_table()[buf[idx]];
    }
}

static st_reorder st_make_reorder(const st_loader & L, enum st_reorder_kind kind) {
    if (kind == ST_RE_NONE) {
        return {};
    }

    const int64_t v_rows = L.num_v_heads * L.head_v_dim;
    const int64_t vpk    = L.num_v_heads / L.num_k_heads;

    if (kind == ST_RE_QKV_ROWS) {
        return { 0, 2 * L.head_k_dim * L.num_k_heads, v_rows, L.head_v_dim };
    }
    if (kind == ST_RE_Z_ROWS) {
        return { 0, 0, v_rows, L.head_v_dim };
    }
    if (kind == ST_RE_OUT_COLS) {
        return { 1, 0, v_rows, L.head_v_dim };
    }
    (void) vpk;
    return { 0, 0, L.num_v_heads, 1 };
}

// linear attention stores V heads grouped by K head, ggml expects them tiled
static int64_t st_reorder_src(const st_loader & L, const st_reorder & r, int64_t i) {
    if (r.dim < 0 || i < r.offs || i >= r.offs + r.count) {
        return i;
    }
    const int64_t j  = i - r.offs;
    const int64_t d  = r.head > 1 ? j % r.head : 0;
    const int64_t gk = r.head > 1 ? j / r.head : j;
    const int64_t g  = gk / L.num_k_heads;
    const int64_t k  = gk % L.num_k_heads;
    if (g >= L.num_v_heads / L.num_k_heads) {
        return i;
    }
    return r.offs + (k * (L.num_v_heads / L.num_k_heads) + g) * r.head + d;
}

// emit one matrix of nrows x ncols elements, ncols is the ggml row length
static void st_emit_matrix(
        const st_loader & L, const st_plan & p, const st_ref & src, const st_ref & scale, bool has_scale,
        int64_t nrows, int64_t ncols, uint8_t * dst) {
    const enum ggml_type type = p.type;

    // the source row width is the width of the tensor in the checkpoint, which is narrower than
    // the emitted row when two sources are concatenated
    const int64_t src_ncols = src.ndim >= 2 ? src.ne[1] : src.ne[0];
    const size_t src_row_bytes = (size_t) src_ncols * st_dtype_size(src.dtype);
    const size_t dst_row_bytes = type == GGML_TYPE_F32 ? (size_t) ncols * 4 :
                                 type == GGML_TYPE_Q8_0 ? ggml_row_size(type, ncols) : (size_t) ncols * 2;

    // unmodified data is copied in chunks straight into the model buffer
    if (src.dtype != ST_DT_F8 && st_ggml_type(src.dtype) == type && p.reorder.dim < 0 && !p.add_one && !p.neg_exp &&
        p.emit_ncols == 0) {
        const int64_t rows_per_chunk = std::max<int64_t>(1, (16 * 1024 * 1024) / (int64_t) src_row_bytes);
        for (int64_t r = 0; r < nrows; r += rows_per_chunk) {
            const int64_t n = std::min(rows_per_chunk, nrows - r);
            src.shard->read(src.offs + (uint64_t) r * src_row_bytes, dst + (size_t) r * dst_row_bytes, (size_t) n * src_row_bytes);
        }
        return;
    }

    // the source has its own row count, which is smaller than the emitted one when the
    // parallel branch is concatenated or when the rest is zero filled
    const int64_t src_rows = src.ndim >= 2 ? src.ne[0] : 1;

    // the source rows may be strided, e.g. the temporal planes of the patch embedding
    const size_t src_row_bytes_ext = p.src_row_stride != 0 ? (size_t) p.src_row_stride * st_dtype_size(src.dtype) : src_row_bytes;
    const size_t src_row_bytes_used = src_row_bytes_ext != src_row_bytes
        ? (size_t) (src_rows - 1) * src_row_bytes_ext + src_row_bytes
        : (size_t) src_rows * src_row_bytes;

    std::vector<uint8_t> src_buf(src_row_bytes_used);
    src.shard->read(src.offs, src_buf.data(), src_buf.size());

    std::vector<float> scale_buf;
    int64_t scale_stride = 0;
    if (has_scale) {
        scale_stride = scale.ncols(); // number of 128 column blocks
        const size_t n = (size_t) scale.nrows() * scale.ncols();
        std::vector<uint8_t> raw(n * st_dtype_size(scale.dtype));
        scale.shard->read(scale.offs, raw.data(), raw.size());
        scale_buf.resize(n);
        for (size_t i = 0; i < n; ++i) {
            scale_buf[i] = st_read_f32(raw.data(), scale.dtype, i);
        }
    }

    auto emit_rows = [&](int64_t r0, int64_t r1) {
        std::vector<float> row(ncols);
        for (int64_t r = r0; r < r1; ++r) {
            const int64_t sr = p.reorder.dim == 0 ? st_reorder_src(L, p.reorder, r) : r;
            const uint8_t * srow = src_buf.data() + (size_t) sr * src_row_bytes_ext;
            const float * sscale = has_scale ? scale_buf.data() + (sr / ST_BLOCK) * scale_stride : nullptr;

            for (int64_t c = 0; c < ncols; ++c) {
                const int64_t sc = p.reorder.dim == 1 ? st_reorder_src(L, p.reorder, c) : c;
                float v = src.dtype == ST_DT_F8 ? st_e4m3_table()[srow[sc]] * sscale[sc / ST_BLOCK]
                                                : st_read_f32(srow, src.dtype, sc);
                if (p.add_one) {
                    v += 1.0f;
                }
                if (p.neg_exp) {
                    v = -expf(v);
                }
                row[c] = v;
            }

            uint8_t * drow = dst + (size_t) r * dst_row_bytes;
            if (type == GGML_TYPE_Q8_0) {
                ggml_quantize_chunk(GGML_TYPE_Q8_0, row.data(), drow, 0, 1, ncols, nullptr);
            } else if (type == GGML_TYPE_F32) {
                memcpy(drow, row.data(), (size_t) ncols * 4);
            } else if (type == GGML_TYPE_BF16) {
                for (int64_t c = 0; c < ncols; ++c) {
                    ((uint16_t *) drow)[c] = st_f32_to_bf16(row[c]);
                }
            } else {
                ggml_fp32_to_fp16_row(row.data(), (ggml_fp16_t *) drow, ncols);
            }
        }
    };

    // the per element work is heavy for quantized weights, split the rows over the threads
    const int64_t n_threads = src.dtype == ST_DT_F8 ? (int64_t) std::min<unsigned>(st_threads(), (unsigned) nrows) : 1;
    if (n_threads <= 1) {
        emit_rows(0, nrows);
        return;
    }

    const int64_t rows_per_thread = (nrows + n_threads - 1) / n_threads;
    std::vector<std::thread> workers;
    workers.reserve(n_threads - 1);
    for (int64_t t = 0; t < n_threads - 1; ++t) {
        const int64_t r0 = t * rows_per_thread;
        const int64_t r1 = std::min(nrows, r0 + rows_per_thread);
        workers.emplace_back(emit_rows, r0, r1);
    }
    emit_rows((n_threads - 1) * rows_per_thread, nrows);
    for (auto & w : workers) {
        w.join();
    }
}

static void st_emit(const st_loader & L, const st_plan & p, uint8_t * out) {
    if (!p.exps.empty()) {
        const int64_t ne0 = p.ne[0];
        const int64_t ne1 = p.ne[1];
        const int64_t ne2 = p.ne[2];
        const size_t expert_bytes = st_nbytes(p.type, p.ne, p.ndim) / (size_t) ne2;

        for (int64_t e = 0; e < ne2; ++e) {
            if (p.exps[e].shard == nullptr) {
                throw std::runtime_error(string_format("missing expert %lld of tensor '%s'", (long long) e, p.name.c_str()));
            }
            const bool has_scale = !p.exps_scale.empty() && p.exps_scale[e].shard != nullptr;
            st_emit_matrix(L, p, p.exps[e], has_scale ? p.exps_scale[e] : st_ref{}, has_scale, ne1, ne0, out + e * expert_bytes);
        }
        return;
    }

    const int64_t nrows = p.emit_nrows > 0 ? p.emit_nrows : (p.ndim >= 2 ? p.ne[1] : 1);
    const int64_t ncols = p.emit_ncols > 0 ? p.emit_ncols : p.ne[0];
    st_emit_matrix(L, p, p.src, p.scale, p.has_scale != 0, nrows, ncols, out);
}

// state shared with the loader through the model source callbacks
struct st_source {
    const st_loader * loader = nullptr;
    std::vector<std::string> files;
    std::vector<const char *> file_ptrs;
};

static bool st_source_get_offset(const char * tensor_name, int32_t * file_idx, uint64_t * offset, void * userdata) {
    const st_source * src = (const st_source *) userdata;

    const auto it = src->loader->plans.find(tensor_name);
    if (it == src->loader->plans.end()) {
        return false;
    }

    const st_plan & p = *it->second;
    if (!p.verbatim) {
        return false;
    }

    *file_idx = p.src.shard->idx;
    *offset   = p.src.shard->data_offs + p.src.offs; // skip the safetensors header
    return true;
}

static void st_source_get_data(const char * tensor_name, void * data, size_t size, void * userdata) {
    const st_loader & L = *((const st_source *) userdata)->loader;

    if (data == nullptr) {
        return; // memory fitting does not allocate the tensors
    }

    const auto it = L.plans.find(tensor_name);
    if (it == L.plans.end()) {
        throw std::runtime_error(std::string("no source for tensor '") + tensor_name + "'");
    }

    const st_plan & p = *it->second;
    const size_t nbytes = st_nbytes(p.type, p.ne, p.ndim);
    if (nbytes != size) {
        throw std::runtime_error(string_format("size mismatch for tensor '%s': expected %zu, got %zu", tensor_name, nbytes, size));
    }

    st_emit(L, p, (uint8_t *) data);
}

// ---------------------------------------------------------------------------
// entry points
// ---------------------------------------------------------------------------

struct llama_model * common_safetensors_load_model(const std::string & path, const std::string & fp8_type, const struct llama_model_params & params, enum st_mode mode);

static fs::path st_checkpoint_dir(const std::string & path);

static void st_add_meta_agnes(gguf_context * meta, const fs::path & dir, const json & cfg, const std::string & model_name);
static void st_build_plans_agnes(st_loader & L, gguf_context * meta, const fs::path & dir, const json & cfg, enum st_mode mode);

struct llama_model * common_model_load_from_file(const std::string & path, const std::string & safetensors_outtype, const struct llama_model_params & params, bool mtp_only) {
    if (common_safetensors_is_checkpoint(path)) {
        return common_safetensors_load_model(path, safetensors_outtype, params, mtp_only ? ST_MODE_MTP : ST_MODE_MODEL);
    }
    return llama_model_load_from_file(path.c_str(), params);
}

bool common_safetensors_has_mtp_head(const std::string & path) {
    std::error_code ec;
    if (!common_safetensors_is_checkpoint(path)) {
        return false;
    }
    const fs::path dir = st_checkpoint_dir(path);

    try {
        const fs::path index_file = dir / "model.safetensors.index.json";
        if (fs::is_regular_file(index_file)) {
            const json index_json = st_read_json(index_file);
            for (const auto & [name, file] : index_json.at("weight_map").items()) {
                (void) file;
                if (name.rfind("mtp.", 0) == 0 || name.rfind("model.mtp.", 0) == 0) {
                    return true;
                }
            }
            return false;
        }

        for (const auto & entry : fs::directory_iterator(dir, ec)) {
            if (entry.path().extension() != ".safetensors") {
                continue;
            }
            std::ifstream file(entry.path(), std::ios::binary);
            uint64_t header_len = 0;
            file.read((char *) &header_len, sizeof(header_len));
            std::string header(header_len, '\0');
            file.read(header.data(), header_len);
            for (const auto & [name, info] : json::parse(header).items()) {
                (void) info;
                if (name.rfind("mtp.", 0) == 0 || name.rfind("model.mtp.", 0) == 0) {
                    return true;
                }
            }
        }
    } catch (const std::exception & e) {
        LOG_WRN("safetensors: cannot look for an MTP head in '%s': %s\n", path.c_str(), e.what());
    }

    return false;
}

// resolves the directory with the checkpoint files, also for a Hugging Face hub cache entry
static fs::path st_checkpoint_dir(const std::string & path) {
    std::error_code ec;
    const fs::path p(path);

    if (fs::is_regular_file(p, ec)) {
        return p.parent_path();
    }
    if (!fs::is_directory(p, ec)) {
        return p;
    }

    // models--<org>--<name>/snapshots/<revision>/
    const fs::path snapshots = p / "snapshots";
    if (fs::is_directory(snapshots, ec)) {
        for (fs::directory_iterator it(snapshots, ec), end; !ec && it != end; it.increment(ec)) {
            if (it->is_directory(ec) && fs::is_regular_file(it->path() / "config.json", ec)) {
                return it->path();
            }
        }
    }

    return p;
}

bool common_safetensors_is_checkpoint(const std::string & path) {
    std::error_code ec;
    const fs::path p(path);

    if (fs::is_regular_file(p, ec)) {
        return p.extension() == ".safetensors";
    }

    const fs::path dir = st_checkpoint_dir(path);
    if (!fs::is_directory(dir, ec) || !fs::is_regular_file(dir / "config.json", ec)) {
        return false;
    }


    if (fs::is_regular_file(dir / "model.safetensors.index.json", ec)) {
        return true;
    }

    for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
        if (it->path().extension() == ".safetensors") {
            return true;
        }
    }

    return false;
}

static void st_build_plans(st_loader & L, gguf_context * meta, const fs::path & dir, const json & cfg, enum st_mode mode) {
    const uint32_t n_expert = cfg.at("text_config").at("num_experts").get<uint32_t>();
    const int      n_layer  = cfg.at("text_config").at("num_hidden_layers").get<int>();

    std::vector<std::string> part_names;
    const fs::path index_file = dir / "model.safetensors.index.json";
    if (fs::is_regular_file(index_file)) {
        const json index_json = st_read_json(index_file);
        for (const auto & [name, file] : index_json.at("weight_map").items()) {
            (void) name;
            part_names.push_back(file.get<std::string>());
        }
    } else {
        std::error_code ec;
        for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
            if (it->path().extension() == ".safetensors") {
                part_names.push_back(it->path().filename().string());
            }
        }
    }

    if (part_names.empty()) {
        throw std::runtime_error("no safetensors shards found in " + dir.string());
    }

    std::sort(part_names.begin(), part_names.end());
    part_names.erase(std::unique(part_names.begin(), part_names.end()), part_names.end());

    std::unordered_map<std::string, st_ref> tensors;
    for (const auto & part : part_names) {
        auto shard = std::make_unique<st_shard>();
        shard->idx  = (int) L.shards.size();
        shard->name = part;
        shard->path = dir / part;
        shard->file = fopen(shard->path.string().c_str(), "rb");
        if (shard->file == nullptr) {
            throw std::runtime_error("failed to open " + shard->path.string());
        }

        uint64_t header_len = 0;
        if (fread(&header_len, 1, sizeof(header_len), shard->file) != sizeof(header_len)) {
            throw std::runtime_error("failed to read the header of " + shard->path.string());
        }
        if (header_len == 0 || header_len > (1u << 30)) {
            throw std::runtime_error("invalid safetensors header size in " + shard->path.string());
        }

        std::string header(header_len, '\0');
        if (fread(header.data(), 1, header_len, shard->file) != header_len) {
            throw std::runtime_error("short read of the header of " + shard->path.string());
        }
        shard->data_offs = 8 + header_len;

        const json header_json = json::parse(header);
        for (const auto & [name, info] : header_json.items()) {
            if (name == "__metadata__") {
                continue;
            }
            const auto offsets = info.at("data_offsets");
            tensors[name] = st_make_ref(shard.get(), offsets.at(0).get<uint64_t>(), info.at("shape").get<std::vector<int64_t>>(),
                    st_dtype_from_name(info.at("dtype").get<std::string>()));
        }

        L.shards.push_back(std::move(shard));
    }

    std::unordered_map<std::string, std::vector<st_ref>> exps;
    std::unordered_map<std::string, std::vector<st_ref>> exps_scale;

    auto add_plan = [&](std::unique_ptr<st_plan> plan) {
        if (L.plans.count(plan->name) != 0) {
            throw std::runtime_error("duplicated tensor '" + plan->name + "'");
        }
        L.plans[plan->name] = plan.get();
        L.storage.push_back(std::move(plan));
    };

    size_t n_skipped = 0;

    for (const auto & [name, ref] : tensors) {
        if (name.size() > 10 && name.compare(name.size() - 10, 10, "_scale_inv") == 0) {
            continue;
        }

        // an MTP head export contains only the MTP block, the trunk layers stay in the target model
        static const char * layers_prefix = "model.language_model.layers.";
        if (mode == ST_MODE_MTP && name.rfind(layers_prefix, 0) == 0 && atoi(name.c_str() + strlen(layers_prefix)) < n_layer) {
            n_skipped++;
            continue;
        }

        st_map m;
        int64_t expert = -1;
        std::string expert_base;
        if (!st_map_tensor(name, m, expert, expert_base)) {
            n_skipped++;
            continue;
        }

        if (expert >= 0) {
            auto & weights = exps[expert_base];
            auto & scales  = exps_scale[expert_base];
            if (weights.empty()) {
                weights.resize(n_expert);
                scales.resize(n_expert);
            }
            if (expert >= (int64_t) weights.size()) {
                throw std::runtime_error(string_format("expert index %lld out of range for '%s'", (long long) expert, expert_base.c_str()));
            }
            weights[expert] = ref;
            const auto it_scale = tensors.find(name + "_scale_inv");
            if (it_scale != tensors.end()) {
                scales[expert] = it_scale->second;
            }
            continue;
        }

        auto plan = std::make_unique<st_plan>();
        plan->name    = m.gguf;
        plan->src     = ref;
        plan->add_one = m.add_one;
        plan->neg_exp = m.neg_exp;
        plan->reorder = st_make_reorder(L, m.reorder);
        plan->ndim    = ref.ndim;

        if (ref.ndim == 1) {
            plan->ne[0] = ref.ne[0];
        } else {
            plan->ne[0] = ref.ne[1];
            plan->ne[1] = ref.ne[0];
        }

        if (ref.dtype == ST_DT_F8) {
            const auto it_scale = tensors.find(name + "_scale_inv");
            if (it_scale == tensors.end()) {
                throw std::runtime_error("missing block scales for tensor " + name);
            }
            plan->has_scale = 1;
            plan->scale     = it_scale->second;
        }

        plan->type = m.role == ST_ROLE_F32 ? GGML_TYPE_F32 :
                     ref.dtype == ST_DT_F8  ? L.fp8_type : st_ggml_type(ref.dtype);

        // a 1d tensor has a single row, so the permutation applies to the element index
        if (plan->ndim == 1 && plan->reorder.dim == 0) {
            plan->reorder.dim = 1;
        }

        // tensors that are stored as-is are read by the loader itself, with mmap like a GGUF file
        plan->verbatim = ref.dtype != ST_DT_F8 && st_ggml_type(ref.dtype) == plan->type &&
                         plan->reorder.dim < 0 && !plan->add_one && !plan->neg_exp;

        add_plan(std::move(plan));
    }

    for (auto & [base, weights] : exps) {
        auto plan = std::make_unique<st_plan>();
        plan->name     = base;
        plan->ndim     = 3;
        plan->n_expert = n_expert;
        plan->exps     = std::move(weights);
        plan->exps_scale = std::move(exps_scale[base]);
        plan->ne[0]    = plan->exps[0].ne[1];
        plan->ne[1]    = plan->exps[0].ne[0];
        plan->ne[2]    = n_expert;
        plan->type     = plan->exps[0].dtype == ST_DT_F8 ? L.fp8_type : st_ggml_type(plan->exps[0].dtype);

        for (int64_t e = 0; e < (int64_t) plan->exps.size(); ++e) {
            if (plan->exps[e].shard == nullptr) {
                throw std::runtime_error(string_format("missing expert %lld of tensor '%s'", (long long) e, base.c_str()));
            }
        }

        add_plan(std::move(plan));
    }

    for (const auto & plan : L.storage) {
        struct ggml_tensor tensor;
        memset(&tensor, 0, sizeof(tensor));
        tensor.type = plan->type;
        for (int i = 0; i < 4; ++i) {
            tensor.ne[i] = plan->ne[i];
        }
        ggml_set_name(&tensor, plan->name.c_str());
        gguf_add_tensor(meta, &tensor);
    }

    if (n_skipped > 0) {
        LOG_INF("safetensors: skipped %zu tensors (vision tower or unused)\n", n_skipped);
    }
    LOG_INF("safetensors: %zu tensors, %zu shards\n", L.storage.size(), L.shards.size());
}

// the metadata, the shard file handles and the source data must outlive the model
static std::vector<std::unique_ptr<st_loader>> st_keep;
static std::vector<std::unique_ptr<st_source>> st_keep_source;
static std::vector<gguf_context *> st_keep_meta;

struct llama_model * common_safetensors_load_model(const std::string & path, const std::string & fp8_type, const struct llama_model_params & params, enum st_mode mode) {
    const fs::path dir = st_checkpoint_dir(path);

    try {
        const json cfg = json::parse(st_read_text_file(dir / "config.json"));

        const std::string model_type = cfg.value("model_type", std::string());
        if (model_type != "qwen3_5_moe" && model_type != "agnes") {
            throw std::runtime_error("unsupported safetensors architecture: " + model_type);
        }

        auto L = std::make_unique<st_loader>();
        if (fp8_type == "q8_0") {
            L->fp8_type = GGML_TYPE_Q8_0;
        } else if (fp8_type == "f16") {
            L->fp8_type = GGML_TYPE_F16;
        } else if (fp8_type == "bf16") {
            L->fp8_type = GGML_TYPE_BF16;
        } else {
            throw std::runtime_error("unsupported --safetensors-outtype: " + fp8_type);
        }

        gguf_context * meta = gguf_init_empty();

        if (model_type == "agnes") {
            st_add_meta_agnes(meta, dir, cfg, dir.filename().string());
            st_add_meta_vocab(meta, dir, cfg);
            st_build_plans_agnes(*L, meta, dir, cfg, mode);
        } else {
            st_add_meta_arch(*L, meta, cfg, dir.filename().string());
            st_add_meta_vocab(meta, dir, cfg);
            st_build_plans(*L, meta, dir, cfg, mode);
        }

        llama_model_params mparams = params;

        auto src = std::make_unique<st_source>();
        src->loader = L.get();
        src->files.reserve(L->shards.size());
        for (const auto & shard : L->shards) {
            src->files.push_back(shard->path.string());
        }
        for (const auto & file : src->files) {
            src->file_ptrs.push_back(file.c_str());
        }

        const llama_model_source source = {
            /*.files      =*/ src->file_ptrs.data(),
            /*.n_files    =*/ src->file_ptrs.size(),
            /*.get_offset =*/ st_source_get_offset,
            /*.get_data   =*/ st_source_get_data,
            /*.userdata   =*/ src.get(),
        };

        llama_model * model = llama_model_load_from_source(meta, &source, mparams);
        if (model == nullptr) {
            gguf_free(meta);
            return nullptr;
        }

        st_keep.push_back(std::move(L));
        st_keep_source.push_back(std::move(src));
        st_keep_meta.push_back(meta);
        return model;
    } catch (const std::exception & e) {
        LOG_ERR("safetensors: %s\n", e.what());
        return nullptr;
    }
}

// ---------------------------------------------------------------------------
// mmproj (CLIP / vision tower)
// ---------------------------------------------------------------------------

static bool st_ends_with(const std::string & s, const char * suffix) {
    const size_t n = strlen(suffix);
    return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

static bool st_clip_is_f16(const std::string & name) {
    // everything that is not a linear weight is kept in fp32, like the converter does
    return st_ends_with(name, "attn_qkv.weight") || st_ends_with(name, "attn_out.weight") ||
           st_ends_with(name, "ffn_up.weight")   || st_ends_with(name, "ffn_down.weight") ||
           name == "mm.0.weight" || name == "mm.2.weight";
}

static void st_add_meta_clip(gguf_context * meta, const fs::path & dir, const json & cfg, const std::string & model_name) {
    const json & v = cfg.at("vision_config");
    const json & t = cfg.at("text_config");

    const uint32_t hidden    = v.at("hidden_size").get<uint32_t>();
    const uint32_t depth     = v.at("depth").get<uint32_t>();
    const uint32_t patch     = v.at("patch_size").get<uint32_t>();
    const uint32_t n_pos     = v.value("num_position_embeddings", 0u);
    const uint32_t image_sz  = n_pos > 0 ? (uint32_t) (std::lround(std::sqrt((double) n_pos)) * patch) : 0;

    gguf_set_val_str(meta, "general.architecture", "clip");
    gguf_set_val_str(meta, "general.type",         "mmproj");
    gguf_set_val_str(meta, "general.name",         model_name.c_str());

    gguf_set_val_bool(meta, "clip.has_vision_encoder", true);
    gguf_set_val_u32 (meta, "clip.vision.projection_dim", v.at("out_hidden_size").get<uint32_t>());
    gguf_set_val_u32 (meta, "clip.vision.image_size",     image_sz);
    gguf_set_val_u32 (meta, "clip.vision.patch_size",     patch);
    gguf_set_val_u32 (meta, "clip.vision.embedding_length", hidden);
    gguf_set_val_u32 (meta, "clip.vision.feed_forward_length", v.at("intermediate_size").get<uint32_t>());
    gguf_set_val_u32 (meta, "clip.vision.block_count",        depth);
    gguf_set_val_u32 (meta, "clip.vision.attention.head_count", v.at("num_heads").get<uint32_t>());
    gguf_set_val_str (meta, "clip.projector_type", "qwen3vl_merger");
    gguf_set_val_bool(meta, "clip.use_gelu", true);
    gguf_set_val_u32 (meta, "clip.vision.spatial_merge_size", v.value("spatial_merge_size", 2u));
    gguf_set_val_f32 (meta, "clip.vision.attention.layer_norm_epsilon", t.at("rms_norm_eps").get<float>());

    // image normalization, taken from the image processor config when present
    std::vector<float> mean = { 0.5f, 0.5f, 0.5f };
    std::vector<float> stdev = { 0.5f, 0.5f, 0.5f };
    {
        const fs::path pre = dir / "preprocessor_config.json";
        if (fs::is_regular_file(pre)) {
            const json cfg = st_read_json(pre);
            if (cfg.contains("image_mean")) {
                mean = cfg.at("image_mean").get<std::vector<float>>();
            }
            if (cfg.contains("image_std")) {
                stdev = cfg.at("image_std").get<std::vector<float>>();
            }
        }
    }
    gguf_set_arr_data(meta, "clip.vision.image_mean", GGUF_TYPE_FLOAT32, mean.data(),  mean.size());
    gguf_set_arr_data(meta, "clip.vision.image_std",  GGUF_TYPE_FLOAT32, stdev.data(), stdev.size());

    // deepstack: the checkpoint uses none of the layers
    std::vector<uint8_t> deepstack(depth, 0);
    for (const auto & idx : v.value("deepstack_visual_indexes", std::vector<int>())) {
        if (idx >= 0 && idx < (int) deepstack.size()) {
            deepstack[idx] = 1;
        }
    }
    gguf_set_arr_data(meta, "clip.vision.is_deepstack_layers", GGUF_TYPE_BOOL, deepstack.data(), deepstack.size());
}

// maps a tensor of the vision tower to its CLIP tensor, returns false when it is not part of the vision model
static bool st_map_tensor_clip(const std::string & name, std::string & out, int & patch_split) {
    static const std::string prefix = "model.visual.";
    if (name.rfind(prefix, 0) != 0) {
        return false;
    }
    const std::string rest = name.substr(prefix.size());
    patch_split = -1;

    // vision blocks
    if (rest.rfind("blocks.", 0) == 0) {
        const size_t p = rest.find('.', strlen("blocks."));
        if (p == std::string::npos) {
            return false;
        }
        const int il = atoi(rest.substr(strlen("blocks."), p - strlen("blocks.")).c_str());
        const std::string tail = rest.substr(p + 1);

        struct entry { const char * from; const char * to; };
        static const entry entries[] = {
            { "norm1.weight",       "ln1.weight" },
            { "norm1.bias",         "ln1.bias" },
            { "norm2.weight",       "ln2.weight" },
            { "norm2.bias",         "ln2.bias" },
            { "attn.qkv.weight",    "attn_qkv.weight" },
            { "attn.qkv.bias",      "attn_qkv.bias" },
            { "attn.proj.weight",   "attn_out.weight" },
            { "attn.proj.bias",     "attn_out.bias" },
            { "mlp.linear_fc1.weight", "ffn_up.weight" },
            { "mlp.linear_fc1.bias",   "ffn_up.bias" },
            { "mlp.linear_fc2.weight", "ffn_down.weight" },
            { "mlp.linear_fc2.bias",   "ffn_down.bias" },
        };
        for (const auto & e : entries) {
            if (tail == e.from) {
                out = string_format("v.blk.%d.%s", il, e.to);
                return true;
            }
        }
        return false;
    }

    // merger
    if (rest == "merger.linear_fc1.weight") { out = "mm.0.weight"; return true; }
    if (rest == "merger.linear_fc1.bias")   { out = "mm.0.bias";   return true; }
    if (rest == "merger.linear_fc2.weight") { out = "mm.2.weight"; return true; }
    if (rest == "merger.linear_fc2.bias")   { out = "mm.2.bias";   return true; }
    if (rest == "merger.norm.weight")       { out = "v.post_ln.weight"; return true; }
    if (rest == "merger.norm.bias")         { out = "v.post_ln.bias";   return true; }

    // patch embedding: the converter splits the temporal dimension in two tensors
    if (rest == "patch_embed.proj.weight") { out = "v.patch_embd.weight"; patch_split = 0; return true; }
    if (rest == "patch_embed.proj.bias")   { out = "v.patch_embd.bias";   return true; }

    if (rest == "pos_embed.weight") { out = "v.position_embd.weight"; return true; }

    return false;
}

static void st_build_plans_clip(st_loader & L, gguf_context * meta, const fs::path & dir, const json & cfg) {
    (void) cfg;

    std::vector<std::string> part_names;
    const fs::path index_file = dir / "model.safetensors.index.json";
    if (fs::is_regular_file(index_file)) {
        const json index_json = st_read_json(index_file);
        for (const auto & [name, file] : index_json.at("weight_map").items()) {
            (void) name;
            part_names.push_back(file.get<std::string>());
        }
    } else {
        std::error_code ec;
        for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
            if (it->path().extension() == ".safetensors") {
                part_names.push_back(it->path().filename().string());
            }
        }
    }
    std::sort(part_names.begin(), part_names.end());
    part_names.erase(std::unique(part_names.begin(), part_names.end()), part_names.end());

    std::unordered_map<std::string, st_ref> tensors;
    for (const auto & part : part_names) {
        auto shard = std::make_unique<st_shard>();
        shard->idx  = (int) L.shards.size();
        shard->name = part;
        shard->path = dir / part;
        shard->file = fopen(shard->path.string().c_str(), "rb");
        if (shard->file == nullptr) {
            throw std::runtime_error("failed to open " + shard->path.string());
        }

        uint64_t header_len = 0;
        if (fread(&header_len, 1, sizeof(header_len), shard->file) != sizeof(header_len) || header_len == 0 || header_len > (1u << 30)) {
            throw std::runtime_error("invalid safetensors header in " + shard->path.string());
        }
        std::string header(header_len, '\0');
        if (fread(header.data(), 1, header_len, shard->file) != header_len) {
            throw std::runtime_error("short read of the header of " + shard->path.string());
        }
        shard->data_offs = 8 + header_len;

        const json header_json = json::parse(header);
        for (const auto & [name, info] : header_json.items()) {
            if (name == "__metadata__") {
                continue;
            }
            const auto offsets = info.at("data_offsets");
            tensors[name] = st_make_ref(shard.get(), offsets.at(0).get<uint64_t>(), info.at("shape").get<std::vector<int64_t>>(),
                    st_dtype_from_name(info.at("dtype").get<std::string>()));
        }

        L.shards.push_back(std::move(shard));
    }

    size_t n_skipped = 0;

    for (const auto & [name, ref] : tensors) {
        std::string gguf_name;
        int patch_split = -1;
        if (!st_map_tensor_clip(name, gguf_name, patch_split)) {
            n_skipped++;
            continue;
        }

        if (patch_split >= 0) {
            // the temporal patch dimension is split into one tensor per plane, each keeping the
            // [w, h, channels, outputs] shape that the 2d convolution expects
            if (ref.ndim != 5) {
                throw std::runtime_error("unexpected patch embedding shape for " + name);
            }
            const int64_t plane  = ref.ne[2];
            const int64_t ncols  = ref.ne[3] * ref.ne[4];

            for (int64_t t = 0; t < plane; ++t) {
                auto plan = std::make_unique<st_plan>();
                plan->name = t == 0 ? gguf_name : gguf_name + "." + std::to_string(t);
                plan->src  = ref;
                plan->src.offs += (uint64_t) t * ncols * st_dtype_size(ref.dtype);
                plan->src_row_stride = ncols * plane;
                plan->ndim       = 4;
                plan->ne[0]      = ref.ne[4];
                plan->ne[1]      = ref.ne[3];
                plan->ne[2]      = ref.ne[1];
                plan->ne[3]      = ref.ne[0];
                plan->emit_nrows = ref.ne[0] * ref.ne[1];
                plan->emit_ncols = ncols;
                plan->type       = GGML_TYPE_F16;
                L.plans[plan->name] = plan.get();
                L.storage.push_back(std::move(plan));
            }
            continue;
        }

        auto plan = std::make_unique<st_plan>();
        plan->name = gguf_name;
        plan->src  = ref;
        plan->type = st_clip_is_f16(gguf_name) ? GGML_TYPE_F16 : GGML_TYPE_F32;

        if (ref.ndim == 1) {
            plan->ndim  = 1;
            plan->ne[0] = ref.ne[0];
        } else {
            plan->ndim  = 2;
            plan->ne[0] = ref.ne[ref.ndim - 1];
            plan->ne[1] = ref.ne[ref.ndim - 2];
            if (ref.ndim > 2) {
                throw std::runtime_error("unexpected shape for vision tensor " + name);
            }
        }

        if (L.plans.count(plan->name) != 0) {
            throw std::runtime_error("duplicated tensor '" + plan->name + "'");
        }
        L.plans[plan->name] = plan.get();
        L.storage.push_back(std::move(plan));
    }

    for (const auto & plan : L.storage) {
        struct ggml_tensor tensor;
        memset(&tensor, 0, sizeof(tensor));
        tensor.type = plan->type;
        for (int i = 0; i < 4; ++i) {
            tensor.ne[i] = plan->ne[i];
        }
        ggml_set_name(&tensor, plan->name.c_str());
        gguf_add_tensor(meta, &tensor);
    }

    LOG_INF("safetensors: vision tower with %zu tensors, %zu shards (%zu skipped)\n", L.storage.size(), L.shards.size(), n_skipped);
}

// state of a mmproj checkpoint, handed to mtmd
struct st_mmproj_impl {
    std::unique_ptr<st_loader>  loader;
    std::vector<std::string>   files;
    std::vector<const char *>  file_ptrs;
};

static bool st_mmproj_get_offset(const char * tensor_name, int32_t * file_idx, uint64_t * offset, void * userdata) {
    (void) tensor_name; (void) file_idx; (void) offset; (void) userdata;
    return false; // all vision tensors need a conversion
}

static void st_mmproj_get_data(const char * tensor_name, void * data, size_t size, void * userdata) {
    const auto * src = (const common_mmproj_source *) userdata;
    const st_loader & L = *((const st_mmproj_impl *) src->impl)->loader;

    const auto it = L.plans.find(tensor_name);
    if (it == L.plans.end()) {
        throw std::runtime_error(std::string("no source for tensor '") + tensor_name + "'");
    }

    const st_plan & p = *it->second;
    const size_t nbytes = st_nbytes(p.type, p.ne, p.ndim);
    if (nbytes != size) {
        throw std::runtime_error(string_format("size mismatch for tensor '%s': expected %zu, got %zu", tensor_name, nbytes, size));
    }

    st_emit(L, p, (uint8_t *) data);
}

common_mmproj_source * common_mmproj_source_create(const std::string & path) {
    if (!common_safetensors_is_checkpoint(path)) {
        return nullptr;
    }
    const fs::path dir = st_checkpoint_dir(path);

    try {
        const json cfg = st_read_json(dir / "config.json");
        if (!cfg.contains("vision_config")) {
            return nullptr;
        }

        auto * res = new common_mmproj_source();
        auto * impl = new st_mmproj_impl();

        res->metadata = gguf_init_empty();
        res->impl     = impl;

        st_add_meta_clip(res->metadata, dir, cfg, dir.filename().string());
        impl->loader = std::make_unique<st_loader>();
        st_build_plans_clip(*impl->loader, res->metadata, dir, cfg);

        impl->files.reserve(impl->loader->shards.size());
        for (const auto & shard : impl->loader->shards) {
            impl->files.push_back(shard->path.string());
        }
        for (const auto & file : impl->files) {
            impl->file_ptrs.push_back(file.c_str());
        }

        res->source = {
            /*.files      =*/ impl->file_ptrs.data(),
            /*.n_files    =*/ impl->file_ptrs.size(),
            /*.get_offset =*/ st_mmproj_get_offset,
            /*.get_data   =*/ st_mmproj_get_data,
            /*.userdata   =*/ res,
        };

        return res;
    } catch (const std::exception & e) {
        LOG_ERR("safetensors: cannot read the vision tower of '%s': %s\n", path.c_str(), e.what());
        return nullptr;
    }
}

void common_mmproj_source_free(common_mmproj_source * src) {
    if (src == nullptr) {
        return;
    }
    if (src->metadata != nullptr) {
        gguf_free(src->metadata);
    }
    delete (st_mmproj_impl *) src->impl;
    delete src;
}

// ---------------------------------------------------------------------------
// Agnes 3.0: hybrid delta rule / global attention, dense FFN with a parallel branch
// ---------------------------------------------------------------------------

static void st_add_meta_agnes(gguf_context * meta, const fs::path & dir, const json & cfg, const std::string & model_name) {
    const json & t = cfg.at("text_config");

    const uint32_t n_layer  = t.at("num_hidden_layers").get<uint32_t>();
    const uint32_t n_nextn  = t.value("mtp_num_hidden_layers", 0u);
    const uint32_t n_embd   = t.at("hidden_size").get<uint32_t>();
    const uint32_t n_head   = t.at("num_attention_heads").get<uint32_t>();
    const uint32_t head_dim = t.at("head_dim").get<uint32_t>();
    const uint32_t n_ff     = t.at("intermediate_size").get<uint32_t>();
    const uint32_t n_ff_par = t.value("parallel_ffn_intermediate_size", 0u);

    const uint32_t head_k_dim = t.at("linear_key_head_dim").get<uint32_t>();
    const uint32_t head_v_dim = t.at("linear_value_head_dim").get<uint32_t>();
    const uint32_t n_k_heads  = t.at("linear_num_key_heads").get<uint32_t>();
    const uint32_t n_v_heads  = t.at("linear_num_value_heads").get<uint32_t>();

    std::vector<int32_t> rope_sections = { 11, 11, 10, 0 };
    if (t.contains("rope_parameters") && t.at("rope_parameters").contains("mrope_section")) {
        rope_sections = t.at("rope_parameters").at("mrope_section").get<std::vector<int32_t>>();
        rope_sections.resize(4, 0);
    }

    float rope_theta = 1000000.0f;
    if (t.contains("rope_parameters") && t.at("rope_parameters").contains("rope_theta")) {
        rope_theta = t.at("rope_parameters").at("rope_theta").get<float>();
    }

    std::vector<uint32_t> recurrent;
    recurrent.reserve(n_layer + n_nextn);
    for (const auto & type : t.at("layer_types")) {
        recurrent.push_back(type.get<std::string>() == "agnes_delta_attention" ? 1 : 0);
    }
    recurrent.resize(n_layer + n_nextn, 0); // the MTP layer is attention only


    llama_model_saver ms(LLM_ARCH_QWEN35, meta);
    const LLM_KV kv(LLM_ARCH_QWEN35);

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE, "qwen35");
    ms.add_kv(LLM_KV_GENERAL_NAME,         model_name.c_str());

    ms.add_kv(LLM_KV_BLOCK_COUNT,                n_layer + n_nextn);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,             t.at("max_position_embeddings").get<uint32_t>());
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,           n_embd);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,       n_head);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,    t.at("num_key_value_heads").get<uint32_t>());
    ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,       head_dim);
    ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,     head_dim);
    ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       (uint32_t) (head_dim * t.value("partial_rotary_factor", 0.25f)));
    ms.add_kv(LLM_KV_ROPE_FREQ_BASE,             rope_theta);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, t.at("rms_norm_eps").get<float>());
    ms.add_kv(LLM_KV_NEXTN_PREDICT_LAYERS,       n_nextn);
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL,    t.value("global_attention_interval", 4u));
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,          n_ff);
    ms.add_kv(LLM_KV_FEED_FORWARD_PARALLEL_LENGTH, n_ff_par);

    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,    t.at("linear_conv_kernel_dim").get<uint32_t>());
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,     head_k_dim);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,    n_k_heads);
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK, n_v_heads);
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,     head_v_dim * n_v_heads);

    std::vector<uint8_t> recr(recurrent.size());
    for (size_t i = 0; i < recurrent.size(); ++i) {
        recr[i] = (uint8_t) recurrent[i];
    }

    gguf_set_arr_data(meta, kv(LLM_KV_ROPE_DIMENSION_SECTIONS).c_str(), GGUF_TYPE_INT32, rope_sections.data(), rope_sections.size());
    gguf_set_arr_data(meta, kv(LLM_KV_ATTENTION_RECURRENT_LAYERS).c_str(), GGUF_TYPE_BOOL, recr.data(), recr.size());
}

// maps a tensor of the text model, delta_attn performs the role of linear_attn in Qwen3.5
static bool st_map_tensor_agnes(const std::string & name, int64_t n_layer, st_map & res) {
    if (name == "model.language_model.embed_tokens.weight") {
        res.gguf = "token_embd.weight";
        return true;
    }
    if (name == "model.language_model.norm.weight") {
        res.gguf    = "output_norm.weight";
        res.role    = ST_ROLE_F32;
        res.add_one = 1;
        return true;
    }
    if (name == "lm_head.weight") {
        res.gguf = "output.weight";
        return true;
    }
    if (name.rfind("model.visual.", 0) == 0) {
        return false; // vision tower, loaded separately
    }

    std::string rest;
    int il = -1;

    static const std::string prefix = "model.language_model.layers.";
    if (name.rfind(prefix, 0) == 0) {
        const size_t dot = name.find('.', prefix.size());
        if (dot == std::string::npos) {
            return false;
        }
        il   = atoi(name.substr(prefix.size(), dot - prefix.size()).c_str());
        rest = name.substr(dot + 1);
    } else if (name == "mtp.fc.weight") {
        res.gguf = "blk." + std::to_string(n_layer) + ".nextn.eh_proj.weight";
        return true;
    } else if (name == "mtp.pre_fc_norm_embedding.weight") {
        res.gguf    = "blk." + std::to_string(n_layer) + ".nextn.enorm.weight";
        res.role    = ST_ROLE_F32;
        res.add_one = 1;
        return true;
    } else if (name == "mtp.pre_fc_norm_hidden.weight") {
        res.gguf    = "blk." + std::to_string(n_layer) + ".nextn.hnorm.weight";
        res.role    = ST_ROLE_F32;
        res.add_one = 1;
        return true;
    } else if (name == "mtp.norm.weight") {
        res.gguf    = "blk." + std::to_string(n_layer) + ".nextn.shared_head_norm.weight";
        res.role    = ST_ROLE_F32;
        res.add_one = 1;
        return true;
    } else if (name.rfind("mtp.layers.0.", 0) == 0) {
        il   = n_layer;
        rest = name.substr(strlen("mtp.layers.0."));
    } else {
        return false;
    }

    const std::string base = "blk." + std::to_string(il) + ".";

    struct entry {
        const char * suffix;
        const char * gguf;
        enum st_role role;
        int add_one;
        int neg_exp;
        enum st_reorder_kind reorder;
    };

    static const entry entries[] = {
        { "input_layernorm.weight",             "attn_norm.weight",           ST_ROLE_F32,    1, 0, ST_RE_NONE },
        { "post_attention_layernorm.weight",    "post_attention_norm.weight", ST_ROLE_F32,    1, 0, ST_RE_NONE },
        { "global_attn.q_proj.weight",          "attn_q.weight",              ST_ROLE_WEIGHT, 0, 0, ST_RE_NONE },
        { "global_attn.k_proj.weight",          "attn_k.weight",              ST_ROLE_WEIGHT, 0, 0, ST_RE_NONE },
        { "global_attn.v_proj.weight",          "attn_v.weight",              ST_ROLE_WEIGHT, 0, 0, ST_RE_NONE },
        { "global_attn.o_proj.weight",          "attn_output.weight",         ST_ROLE_WEIGHT, 0, 0, ST_RE_NONE },
        { "global_attn.q_norm.weight",          "attn_q_norm.weight",         ST_ROLE_F32,    1, 0, ST_RE_NONE },
        { "global_attn.k_norm.weight",          "attn_k_norm.weight",         ST_ROLE_F32,    1, 0, ST_RE_NONE },
        { "delta_attn.in_proj_qkv.weight",      "attn_qkv.weight",            ST_ROLE_WEIGHT, 0, 0, ST_RE_QKV_ROWS },
        { "delta_attn.in_proj_z.weight",        "attn_gate.weight",           ST_ROLE_WEIGHT, 0, 0, ST_RE_Z_ROWS },
        { "delta_attn.conv1d.weight",           "ssm_conv1d.weight",          ST_ROLE_F32,    0, 0, ST_RE_QKV_ROWS },
        { "delta_attn.dt_bias",                 "ssm_dt.bias",                ST_ROLE_F32,    0, 0, ST_RE_HEADS },
        { "delta_attn.A_log",                   "ssm_a",                      ST_ROLE_F32,    0, 1, ST_RE_HEADS },
        { "delta_attn.in_proj_b.weight",        "ssm_beta.weight",            ST_ROLE_WEIGHT, 0, 0, ST_RE_HEADS },
        { "delta_attn.in_proj_a.weight",        "ssm_alpha.weight",           ST_ROLE_WEIGHT, 0, 0, ST_RE_HEADS },
        { "delta_attn.norm.weight",             "ssm_norm.weight",            ST_ROLE_F32,    0, 0, ST_RE_NONE },
        { "delta_attn.out_proj.weight",         "ssm_out.weight",             ST_ROLE_WEIGHT, 0, 0, ST_RE_OUT_COLS },
        { "mlp.gate_proj.weight",               "ffn_gate.weight",            ST_ROLE_WEIGHT, 0, 0, ST_RE_NONE },
        { "mlp.up_proj.weight",                 "ffn_up.weight",              ST_ROLE_WEIGHT, 0, 0, ST_RE_NONE },
        { "mlp.down_proj.weight",               "ffn_down.weight",            ST_ROLE_WEIGHT, 0, 0, ST_RE_NONE },
        { "mlp.parallel_ffn.gate_proj.weight",  "ffn_gate_par.weight",        ST_ROLE_WEIGHT, 0, 0, ST_RE_NONE },
        { "mlp.parallel_ffn.up_proj.weight",    "ffn_up_par.weight",          ST_ROLE_WEIGHT, 0, 0, ST_RE_NONE },
        { "mlp.parallel_ffn.down_proj.weight",  "ffn_down_par.weight",        ST_ROLE_WEIGHT, 0, 0, ST_RE_NONE },
    };

    for (const auto & e : entries) {
        if (rest == e.suffix) {
            res.gguf    = base + e.gguf;
            res.role    = e.role;
            res.add_one = e.add_one;
            res.neg_exp = e.neg_exp;
            res.reorder = e.reorder;
            return true;
        }
    }

    return false;
}

static void st_build_plans_agnes(st_loader & L, gguf_context * meta, const fs::path & dir, const json & cfg, enum st_mode mode) {
    const json & t = cfg.at("text_config");

    const uint32_t n_layer  = t.at("num_hidden_layers").get<uint32_t>();
    const uint32_t n_ff     = t.at("intermediate_size").get<uint32_t>();
    const uint32_t n_ff_par = t.value("parallel_ffn_intermediate_size", 0u);

    L.num_k_heads = t.at("linear_num_key_heads").get<uint32_t>();
    L.num_v_heads = t.at("linear_num_value_heads").get<uint32_t>();
    L.head_k_dim  = t.at("linear_key_head_dim").get<uint32_t>();
    L.head_v_dim  = t.at("linear_value_head_dim").get<uint32_t>();

    std::vector<std::string> part_names;
    const fs::path index_file = dir / "model.safetensors.index.json";
    if (fs::is_regular_file(index_file)) {
        const json index_json = st_read_json(index_file);
        for (const auto & [name, file] : index_json.at("weight_map").items()) {
            (void) name;
            part_names.push_back(file.get<std::string>());
        }
    } else {
        std::error_code ec;
        for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
            if (it->path().extension() == ".safetensors") {
                part_names.push_back(it->path().filename().string());
            }
        }
    }
    std::sort(part_names.begin(), part_names.end());
    part_names.erase(std::unique(part_names.begin(), part_names.end()), part_names.end());

    std::unordered_map<std::string, st_ref> tensors;
    for (const auto & part : part_names) {
        auto shard = std::make_unique<st_shard>();
        shard->idx  = (int) L.shards.size();
        shard->name = part;
        shard->path = dir / part;
        shard->file = fopen(shard->path.string().c_str(), "rb");
        if (shard->file == nullptr) {
            throw std::runtime_error("failed to open " + shard->path.string());
        }

        uint64_t header_len = 0;
        if (fread(&header_len, 1, sizeof(header_len), shard->file) != sizeof(header_len) || header_len == 0 || header_len > (1u << 30)) {
            throw std::runtime_error("invalid safetensors header in " + shard->path.string());
        }
        std::string header(header_len, '\0');
        if (fread(header.data(), 1, header_len, shard->file) != header_len) {
            throw std::runtime_error("short read of the header of " + shard->path.string());
        }
        shard->data_offs = 8 + header_len;

        const json header_json = json::parse(header);
        for (const auto & [name, info] : header_json.items()) {
            if (name == "__metadata__") {
                continue;
            }
            const auto offsets = info.at("data_offsets");
            tensors[name] = st_make_ref(shard.get(), offsets.at(0).get<uint64_t>(), info.at("shape").get<std::vector<int64_t>>(),
                    st_dtype_from_name(info.at("dtype").get<std::string>()));
        }

        L.shards.push_back(std::move(shard));
    }

    auto add_plan = [&](std::unique_ptr<st_plan> plan) {
        if (L.plans.count(plan->name) != 0) {
            throw std::runtime_error("duplicated tensor '" + plan->name + "'");
        }
        L.plans[plan->name] = plan.get();
        L.storage.push_back(std::move(plan));
    };

    size_t n_skipped = 0;

    for (const auto & [name, ref] : tensors) {
        static const char * layers_prefix = "model.language_model.layers.";
        if (mode == ST_MODE_MTP && name.rfind(layers_prefix, 0) == 0 && atoi(name.c_str() + strlen(layers_prefix)) < (int) n_layer) {
            n_skipped++;
            continue;
        }

        st_map m;
        if (!st_map_tensor_agnes(name, n_layer, m)) {
            n_skipped++;
            continue;
        }

        auto plan = std::make_unique<st_plan>();
        plan->name    = m.gguf;
        plan->src     = ref;
        plan->add_one = m.add_one;
        plan->neg_exp = m.neg_exp;
        plan->reorder = st_make_reorder(L, m.reorder);
        plan->ndim    = ref.ndim;

        if (ref.ndim == 1) {
            plan->ne[0] = ref.ne[0];
        } else {
            plan->ne[0] = ref.ne[1];
            plan->ne[1] = ref.ne[0];
        }

        plan->type = m.role == ST_ROLE_F32 ? GGML_TYPE_F32 : st_ggml_type(ref.dtype);

        if (plan->ndim == 1 && plan->reorder.dim == 0) {
            plan->reorder.dim = 1;
        }

        plan->verbatim = ref.dtype != ST_DT_F8 && st_ggml_type(ref.dtype) == plan->type &&
                         plan->reorder.dim < 0 && !plan->add_one && !plan->neg_exp;

        add_plan(std::move(plan));
    }

    for (const auto & plan : L.storage) {
        struct ggml_tensor tensor;
        memset(&tensor, 0, sizeof(tensor));
        tensor.type = plan->type;
        for (int i = 0; i < 4; ++i) {
            tensor.ne[i] = plan->ne[i];
        }
        ggml_set_name(&tensor, plan->name.c_str());
        gguf_add_tensor(meta, &tensor);
    }

    LOG_INF("safetensors: agnes text model with %zu tensors, %zu shards (%zu skipped)\n", L.storage.size(), L.shards.size(), n_skipped);
}
