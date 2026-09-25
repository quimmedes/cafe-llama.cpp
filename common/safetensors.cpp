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
#include <mutex>
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

// fseek takes a 32 bit offset on windows, the shards are larger than that
#ifdef _WIN32
#    define st_fseek _fseeki64
#else
#    define st_fseek fseeko
#endif

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
    ST_DT_I32, // packed int4 weights, 8 values per element
    ST_DT_I64, // shape of an unpacked packed weight
    ST_DT_U8,  // packed NVFP4 weights, 2 values per element
    ST_DT_I16, // EXL3 trellis words
    ST_DT_U16,
    ST_DT_U32,
    ST_DT_U64,
    ST_DT_F64,
    ST_DT_BOOL,
};

static int st_dtype_size(enum st_dtype t) {
    switch (t) {
        case ST_DT_F8:
        case ST_DT_U8:
        case ST_DT_BOOL: return 1;
        case ST_DT_F32:
        case ST_DT_I32:
        case ST_DT_U32:  return 4;
        case ST_DT_I64:
        case ST_DT_U64:
        case ST_DT_F64:  return 8;
        default:         return 2;
    }
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
    if (name == "I32") {
        return ST_DT_I32;
    }
    if (name == "I64") {
        return ST_DT_I64;
    }
    if (name == "U8") {
        return ST_DT_U8;
    }
    if (name == "I16") {
        return ST_DT_I16;
    }
    if (name == "U16") {
        return ST_DT_U16;
    }
    if (name == "U32") {
        return ST_DT_U32;
    }
    if (name == "U64") {
        return ST_DT_U64;
    }
    if (name == "F64") {
        return ST_DT_F64;
    }
    if (name == "BOOL") {
        return ST_DT_BOOL;
    }
    throw std::runtime_error("unsupported safetensors dtype: " + name);
}

static enum ggml_type st_ggml_type(enum st_dtype t) {
    switch (t) {
        case ST_DT_F32:  return GGML_TYPE_F32;
        case ST_DT_F16:  return GGML_TYPE_F16;
        case ST_DT_BF16: return GGML_TYPE_BF16;
        // integer tensors are read through st_read_f32, so they land in a float type
        case ST_DT_I16:
        case ST_DT_U16:
        case ST_DT_U32:
        case ST_DT_U64:
        case ST_DT_F64:
        case ST_DT_BOOL: return GGML_TYPE_F32;
        default:         throw std::runtime_error("unsupported source dtype");
    }
}

static bool st_ends_with(const std::string & s, const char * suffix) {
    const size_t n = strlen(suffix);
    return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

// a checkpoint often stores null for the values that only apply to another variant
static bool st_json_set(const json & obj, const char * key) {
    return obj.contains(key) && !obj.at(key).is_null();
}

template <typename T>
static T st_json_value(const json & obj, const char * key, const T & def) {
    return st_json_set(obj, key) ? obj.at(key).get<T>() : def;
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
        if (st_fseek(file, (int64_t) (data_offs + offs), SEEK_SET) != 0) {
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

// the *_shape tensor of a packed weight holds the true [rows, cols] of the unpacked weight
static void st_read_shape(const st_ref & shape, int64_t & rows, int64_t & cols) {
    if (shape.dtype != ST_DT_I64 || shape.ne[0] != 2) {
        throw std::runtime_error("unexpected weight_shape tensor");
    }
    int64_t val[2];
    shape.shard->read(shape.offs, val, sizeof(val));
    rows = val[0];
    cols = val[1];
}

// permute V heads from grouped (by K head) to tiled order, as expected by ggml
struct st_reorder {
    int     dim   = -1; // -1: none, 0: rows, 1: cols
    int64_t offs  = 0;
    int64_t count = 0;
    int64_t head  = 1;
    int     rope  = 0;  // rows hold the two halves of each rotary pair apart, ggml wants them interleaved
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
    st_ref        scale;        // fp8 block scales, or the group scales of a packed weight
    int           has_scale = 0;
    int           pack_bits = 0;    // compressed-tensors packed weights: bits per value
    int64_t       pack_group = 0;   // elements per scale group
    int           nvfp4 = 0;        // modelopt NVFP4 weights: nibble packed, one E4M3 scale per 16 columns
    int           scale_per_tensor = 0; // the source scale of an fp8 weight is a single value, not a block
    int           scale_per_row = 0;    // ... or one value per output channel
    int           f8_native = 0;    // keep the fp8 weights of the checkpoint, one fp32 scale per 128 values
    int           scale_per_block = 0;  // the fp8 scale of the source has one value per 128x128 block
    int           reciprocal = 0;   // the source is the reciprocal of the scale ggml applies
    int           rope_freqs = 0;   // a rope frequency factor tensor, generated instead of read
    int64_t       rope_freqs_rot = 0;   // number of rotated pairs of that tensor
    std::vector<st_ref> exps;       // per-expert weights, when non-empty
    std::vector<st_ref> exps_scale; // per-expert scales
    std::vector<st_ref> exps_scale2; // per-expert global scales (NVFP4)
    std::vector<st_ref> exps_iscale; // per-expert input scales (NVFP4)
    int           exps_nvfp4 = 0;   // the experts are NVFP4, packed like a single weight
    int64_t       n_expert = 0;
    std::vector<st_ref> stack_src;  // one scalar source per entry of a stacked sidecar
    int           stack_reciprocal = 0;
    int           split_rows = 0;   // a flat source that the architecture reads as rows
    int           verbatim = 0;     // data is stored as-is in the shard and read directly by the loader
    int           exl3 = 0;         // exl3 quantized weight: trellis indices plus per row and column factors
    int64_t       exl3_bits = 0;    // width of one trellis index
    st_ref        exl3_suh;         // per input channel factor
    st_ref        exl3_svh;         // per output channel factor
    int           mlx_pq2 = 0;      // MLX affine 2-bit weight (U32 weights + F16 scales -> PQ2_0)
};

struct st_loader {
    std::vector<std::unique_ptr<st_shard>> shards;
    std::unordered_map<std::string, const st_plan *> plans;
    std::vector<std::unique_ptr<st_plan>> storage;

    ggml_type fp8_type  = GGML_TYPE_Q8_0;
    ggml_type int4_type = GGML_TYPE_Q4_0;
    ggml_type exl3_type = GGML_TYPE_COUNT; // exl3: unset means follow the bit width of each tensor
    int       fp8_native = 0; // keep the fp8 weights as they are stored in the checkpoint

    // compressed-tensors pack-quantized weights
    int     pack_bits  = 0;
    int64_t pack_group = 0;

    // linear attention head configuration, used for the V head permutation
    int64_t num_k_heads = 0;
    int64_t num_v_heads = 0;
    int64_t head_k_dim  = 0;
    int64_t head_v_dim  = 0;

    int64_t ssm_groups = 0; // mamba2 groups, the ssm norm of the nemotron-h family is split by them

    // attention head configuration, used by the q/k permutation of the generic models
    int64_t n_head    = 0;
    int64_t n_head_kv = 0;
    int64_t head_dim  = 0;

    bool prism_hadamard = false;
    bool hadamard_gdn_v_grouped = false;

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
    ST_RE_ROPE_Q,   // rows of the query projection of a model that keeps the rotary pairs apart
    ST_RE_ROPE_K,   // rows of the key projection of such a model
};

struct st_map {
    std::string gguf;
    enum st_role role = ST_ROLE_WEIGHT;
    int add_one = 0;
    int neg_exp = 0;
    int split_rows = 0; // a flat source that the architecture reads as that many rows
    enum st_reorder_kind reorder = ST_RE_NONE;
    int fused = 0;      // experts fused in a single 3d tensor: 1 gate and up, 2 down
};

// returns false when the tensor is not part of the text model
static bool st_map_tensor(const std::string & name, int64_t n_layer, st_map & res, int64_t & expert, std::string & expert_base) {

    std::string n = name;

    if (n.rfind("language_model.", 0) == 0) {
        n = n.substr(strlen("language_model."));
    }

    // global tensors
    if (n == "model.language_model.embed_tokens.weight" || n == "model.embed_tokens.weight") {
        res.gguf = "token_embd.weight";
        return true;
    }
    if (n == "model.language_model.norm.weight" || n == "model.norm.weight") {
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
        res.gguf = "blk." + std::to_string(n_layer) + ".nextn.eh_proj.weight";
        return true;
    }
    if (n == "mtp.pre_fc_norm_embedding.weight") {
        res.gguf = "blk." + std::to_string(n_layer) + ".nextn.enorm.weight";
        res.role = ST_ROLE_F32;
        res.add_one = 1;
        return true;
    }
    if (n == "mtp.pre_fc_norm_hidden.weight") {
        res.gguf = "blk." + std::to_string(n_layer) + ".nextn.hnorm.weight";
        res.role = ST_ROLE_F32;
        res.add_one = 1;
        return true;
    }
    if (n == "mtp.norm.weight") {
        res.gguf = "blk." + std::to_string(n_layer) + ".nextn.shared_head_norm.weight";
        res.role = ST_ROLE_F32;
        res.add_one = 1;
        return true;
    }

    // the vision tower is not part of the text model
    if (n.rfind("model.visual.", 0) == 0 || n.rfind("visual.", 0) == 0 || n.rfind("vision_tower.", 0) == 0) {
        return false;
    }

    // MTP block is exported as the last layer
    if (n.rfind("mtp.layers.0.", 0) == 0) {
        n = "model.language_model.layers." + std::to_string(n_layer) + "." + n.substr(strlen("mtp.layers.0."));
    }

    static const char * prefix_lm = "model.language_model.layers.";
    static const char * prefix_l  = "model.layers.";
    const char * prefix = nullptr;
    if (n.rfind(prefix_lm, 0) == 0) {
        prefix = prefix_lm;
    } else if (n.rfind(prefix_l, 0) == 0) {
        prefix = prefix_l;
    } else {
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
            // MTP export stores the experts of the layer fused in a 3d tensor
            if (tail == "gate_up_proj") {
                res.fused = 1;
                expert_base = base + "ffn_up_exps.weight";
                return true;
            }
            if (tail == "down_proj") {
                res.fused = 2;
                expert_base = base + "ffn_down_exps.weight";
                return true;
            }
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
        { "mlp.gate_proj.weight",               "ffn_gate.weight",            ST_ROLE_WEIGHT, 0, 0, ST_RE_NONE },
        { "mlp.up_proj.weight",                 "ffn_up.weight",              ST_ROLE_WEIGHT, 0, 0, ST_RE_NONE },
        { "mlp.down_proj.weight",               "ffn_down.weight",            ST_ROLE_WEIGHT, 0, 0, ST_RE_NONE },
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

static void st_add_meta_arch(st_loader & L, gguf_context * meta, const json & cfg, const std::string & dir_name, bool moe) {
    const json & tc = cfg.at("text_config");

    auto get_u32 = [&](const char * key) {
        return st_json_set(tc, key) ? tc.at(key).get<uint32_t>() : 0u;
    };

    const uint32_t n_layer  = get_u32("num_hidden_layers");
    const uint32_t n_nextn  = st_json_value(tc, "mtp_num_hidden_layers", 0u);
    const uint32_t n_embd   = get_u32("hidden_size");
    const uint32_t n_head   = get_u32("num_attention_heads");
    const uint32_t head_dim = get_u32("head_dim");
    const uint32_t n_expert = moe ? get_u32("num_experts") : 0;
    const uint32_t n_ff_exp = moe ? get_u32("moe_intermediate_size") : 0;
    const uint32_t n_ff_sh  = moe ? get_u32("shared_expert_intermediate_size") : 0;

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

    const float partial_rot = st_json_value(tc, "partial_rotary_factor", 0.25f);

    std::vector<uint32_t> recurrent;
    recurrent.reserve(n_layer + n_nextn);
    for (const auto & t : tc.at("layer_types")) {
        recurrent.push_back(t.get<std::string>() == "linear_attention" ? 1 : 0);
    }
    recurrent.resize(n_layer + n_nextn, 0); // MTP layers are attention-only

    llama_model_saver ms(moe ? LLM_ARCH_QWEN35MOE : LLM_ARCH_QWEN35, meta);
    const LLM_KV kv(moe ? LLM_ARCH_QWEN35MOE : LLM_ARCH_QWEN35);

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE, moe ? "qwen35moe" : "qwen35");
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

    if (moe) {
        ms.add_kv(LLM_KV_EXPERT_COUNT,               n_expert);
        ms.add_kv(LLM_KV_EXPERT_USED_COUNT,          get_u32("num_experts_per_tok"));
        ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, n_ff_exp);
        ms.add_kv(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, n_ff_sh);
    } else {
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, get_u32("intermediate_size"));
    }
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

static void st_add_meta_hadamard(st_loader & L, gguf_context * meta, const fs::path & dir, int64_t n_layer) {
    const fs::path hpath = dir / "hadamard.json";
    if (!fs::is_regular_file(hpath)) {
        return;
    }
    const json hj = st_read_json(hpath);

    const uint32_t version = st_json_value(hj, "prism.hadamard.version", 1u);
    gguf_set_val_u32(meta, "prism.hadamard.version", version);

    const uint32_t block_size = st_json_value(hj, "prism.hadamard.block_size", 1024u);
    gguf_set_val_u32(meta, "prism.hadamard.block_size", block_size);

    const std::string transform = st_json_value(hj, "prism.hadamard.transform", std::string("normalized-sylvester-walsh-hadamard"));
    gguf_set_val_str(meta, "prism.hadamard.transform", transform.c_str());

    const std::string axis = st_json_value(hj, "prism.hadamard.axis", std::string("input-last-dimension"));
    gguf_set_val_str(meta, "prism.hadamard.axis", axis.c_str());

    const std::string sign_mode = st_json_value(hj, "prism.hadamard.sign_mode", std::string("explicit"));
    gguf_set_val_str(meta, "prism.hadamard.sign_mode", sign_mode.c_str());

    const bool gdn_v_grouped = st_json_value(hj, "prism.hadamard.gdn_v_grouped", false);
    gguf_set_val_bool(meta, "prism.hadamard.gdn_v_grouped", gdn_v_grouped);
    L.hadamard_gdn_v_grouped = gdn_v_grouped;

    std::vector<std::string> weight_names;
    if (hj.contains("prism.hadamard.weight_names")) {
        for (const auto & item : hj.at("prism.hadamard.weight_names")) {
            const std::string orig = item.get<std::string>();
            st_map m;
            int64_t expert = -1;
            std::string expert_base;
            if (st_map_tensor(orig, n_layer, m, expert, expert_base)) {
                weight_names.push_back(m.gguf);
            }
        }
    }
    std::vector<const char *> wn_ptrs;
    wn_ptrs.reserve(weight_names.size());
    for (const auto & s : weight_names) {
        wn_ptrs.push_back(s.c_str());
    }
    gguf_set_arr_str(meta, "prism.hadamard.weight_names", wn_ptrs.data(), wn_ptrs.size());

    std::vector<std::string> inv_names;
    if (hj.contains("prism.hadamard.inverse_weight_names")) {
        for (const auto & item : hj.at("prism.hadamard.inverse_weight_names")) {
            const std::string orig = item.get<std::string>();
            st_map m;
            int64_t expert = -1;
            std::string expert_base;
            if (st_map_tensor(orig, n_layer, m, expert, expert_base)) {
                inv_names.push_back(m.gguf);
            }
        }
    }
    std::vector<const char *> inv_ptrs;
    inv_ptrs.reserve(inv_names.size());
    for (const auto & s : inv_names) {
        inv_ptrs.push_back(s.c_str());
    }
    gguf_set_arr_str(meta, "prism.hadamard.inverse_weight_names", inv_ptrs.data(), inv_ptrs.size());

    if (hj.contains("prism.hadamard.sign_widths")) {
        std::vector<int32_t> widths = hj.at("prism.hadamard.sign_widths").get<std::vector<int32_t>>();
        gguf_set_arr_data(meta, "prism.hadamard.sign_widths", GGUF_TYPE_INT32, widths.data(), widths.size());
    }

    if (hj.contains("prism.hadamard.sign_values")) {
        std::vector<int32_t> vals;
        const auto & jvals = hj.at("prism.hadamard.sign_values");
        vals.reserve(jvals.size());
        for (const auto & v : jvals) {
            vals.push_back((int32_t) v.get<double>());
        }
        gguf_set_arr_data(meta, "prism.hadamard.sign_values", GGUF_TYPE_INT32, vals.data(), vals.size());
    }
}

// the tokenizer of a checkpoint, gemma 4 uses the sentencepiece style pieces of this repo
enum st_vocab_kind {
    ST_VOCAB_BPE,
    ST_VOCAB_GEMMA4,
    ST_VOCAB_NEMOTRON, // same byte pair encoding, with the pre-tokenizer of the nemotron tokenizer
};

static void st_add_meta_vocab(gguf_context * meta, const fs::path & dir, const json & cfg, enum st_vocab_kind kind) {
    const json & tc = cfg.contains("text_config") ? cfg.at("text_config") : cfg;
    const bool gemma4 = kind == ST_VOCAB_GEMMA4;

    // the vocabulary size lives in the text configuration of a multimodal checkpoint, and in the top level of a plain one
    const uint32_t n_vocab = st_json_set(tc, "vocab_size") ? tc.at("vocab_size").get<uint32_t>() :
                             st_json_set(cfg, "vocab_size") ? cfg.at("vocab_size").get<uint32_t>() : 0;

    const fs::path vocab_file  = dir / "vocab.json";
    const fs::path merges_file = dir / "merges.txt";

    // the slow tokenizer ships vocab.json/merges.txt, the fast one keeps both in tokenizer.json
    json vocab;
    std::vector<std::string> merges;
    if (fs::is_regular_file(vocab_file)) {
        vocab = st_read_json(vocab_file);
    }
    if (fs::is_regular_file(merges_file)) {
        FILE * f = fopen(merges_file.string().c_str(), "rb");
        if (f == nullptr) {
            throw std::runtime_error("failed to open " + merges_file.string());
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

    const json tokc = st_read_json(dir / "tokenizer.json");
    if (vocab.is_null() || merges.empty()) {
        // only a byte pair encoding checkpoint can be read directly
        const bool bpe = tokc.contains("model") && st_json_value(tokc.at("model"), "type", std::string()) == "BPE";
        if (bpe) {
            const json & tm = tokc.at("model");
            if (vocab.is_null()) {
                vocab = tm.at("vocab");
            }
            if (merges.empty() && tm.contains("merges")) {
                for (const auto & m : tm.at("merges")) {
                    std::string merge;
                    if (m.is_array()) {
                        for (const auto & part : m) {
                            if (!merge.empty()) {
                                merge += ' ';
                            }
                            for (const char c : part.get<std::string>()) {
                                // a literal space inside a merge is encoded as U+0120, like the converter does
                                merge += c == ' ' ? "\xc4\xa0" : std::string(1, c);
                            }
                        }
                    } else {
                        merge = m.get<std::string>();
                    }
                    merges.push_back(merge);
                }
            }
        }
    }

    if (vocab.is_null() || merges.empty()) {
        throw std::runtime_error("cannot read the tokenizer of " + dir.string() +
                ": only byte pair encoding checkpoints are supported directly (vocab.json + merges.txt, or a BPE tokenizer.json)"
                " - convert this checkpoint to GGUF first");
    }

    // added tokens keep their own type, base vocabulary tokens are NORMAL
    struct st_added { std::string content; bool special; };
    std::unordered_map<uint32_t, st_added> added;
    if (tokc.contains("added_tokens")) {
        for (const auto & t : tokc.at("added_tokens")) {
            const uint32_t id = t.at("id").get<uint32_t>();
            added[id] = { t.at("content").get<std::string>(), st_json_value(t, "special", false) };
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

    llama_model_saver ms(LLM_ARCH_QWEN35MOE, meta);
    const LLM_KV kv(LLM_ARCH_QWEN35MOE);

    const char * tokenizer_model = gemma4 ? "gemma4" : "gpt2";
    const char * tokenizer_pre   = gemma4 ? "gemma4" : kind == ST_VOCAB_NEMOTRON ? "pixtral" : "qwen35";
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,  tokenizer_model);
    ms.add_kv(LLM_KV_TOKENIZER_PRE,    tokenizer_pre);
    ms.add_kv(LLM_KV_TOKENIZER_LIST,   tokens);
    ms.add_kv(LLM_KV_TOKENIZER_MERGES, merges);
    gguf_set_arr_data(meta, kv(LLM_KV_TOKENIZER_TOKEN_TYPE).c_str(), GGUF_TYPE_INT32, types.data(), types.size());
    if (gemma4) {
        // the pieces of these checkpoints carry no scores, the loader only uses the merge ranks
        const std::vector<float> scores(n_vocab, 0.0f);
        gguf_set_arr_data(meta, kv(LLM_KV_TOKENIZER_SCORES).c_str(), GGUF_TYPE_FLOAT32, scores.data(), scores.size());
    }

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
        const std::string name = v.is_string() ? v.get<std::string>() : st_json_value(v, "content", std::string());
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
    ms.add_kv(LLM_KV_TOKENIZER_ADD_BOS, gemma4 ? true : st_json_value(tokc_cfg, "add_bos_token", false));

    const fs::path tmpl = dir / "chat_template.jinja";
    if (fs::is_regular_file(tmpl)) {
        ms.add_kv(LLM_KV_TOKENIZER_CHAT_TEMPLATE, st_read_text_file(tmpl).c_str());
    } else if (tokc_cfg.contains("chat_template")) {
        // the template is usually stored in the tokenizer config when there is no file of its own
        const json & t = tokc_cfg.at("chat_template");
        const json & first = t.is_array() ? (t.empty() ? t : t.at(0)) : t;
        if (first.is_string()) {
            ms.add_kv(LLM_KV_TOKENIZER_CHAT_TEMPLATE, first.get<std::string>().c_str());
        }
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
        case ST_DT_I16: {
            int16_t v;
            memcpy(&v, buf + idx * 2, sizeof(v));
            return (float) v;
        }
        case ST_DT_U16: {
            uint16_t v;
            memcpy(&v, buf + idx * 2, sizeof(v));
            return (float) v;
        }
        case ST_DT_U32: {
            uint32_t v;
            memcpy(&v, buf + idx * 4, sizeof(v));
            return (float) v;
        }
        case ST_DT_U64: {
            uint64_t v;
            memcpy(&v, buf + idx * 8, sizeof(v));
            return (float) v;
        }
        case ST_DT_F64: {
            double v;
            memcpy(&v, buf + idx * 8, sizeof(v));
            return (float) v;
        }
        case ST_DT_BOOL:
            return buf[idx] != 0 ? 1.0f : 0.0f;
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
    if (kind == ST_RE_ROPE_Q || kind == ST_RE_ROPE_K) {
        // llama style checkpoints keep the two halves of every rotary pair apart, ggml rotates them interleaved
        const int64_t n_head = kind == ST_RE_ROPE_Q ? L.n_head : L.n_head_kv;
        st_reorder res;
        res.dim   = 0;
        res.count = n_head * L.head_dim;
        res.head  = L.head_dim;
        res.rope  = 1;
        return res;
    }
    (void) vpk;
    return { 0, 0, L.num_v_heads, 1 };
}

// linear attention stores V heads grouped by K head, ggml expects them tiled
static int64_t st_reorder_src(const st_loader & L, const st_reorder & r, int64_t i) {
    if (r.dim < 0 || i < r.offs || i >= r.offs + r.count) {
        return i;
    }
    if (r.rope) {
        const int64_t h = i / r.head;
        const int64_t j = i % r.head;
        return h * r.head + (j % 2) * (r.head / 2) + j / 2;
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
    // the emitted row when two sources are concatenated, or when the weights are packed
    const int64_t src_ncols = src.ndim >= 2 ? src.ne[1] : src.ne[0];
    const size_t src_row_bytes = (size_t) src_ncols * st_dtype_size(src.dtype);
    const size_t dst_row_bytes = ggml_is_quantized(type) ? ggml_row_size(type, ncols) :
                                 type == GGML_TYPE_F32 ? (size_t) ncols * 4 : (size_t) ncols * 2;

    // unmodified data is copied in chunks straight into the model buffer
    if (p.pack_bits == 0 && src.dtype != ST_DT_F8 && st_ggml_type(src.dtype) == type && p.reorder.dim < 0 &&
        !p.add_one && !p.neg_exp && p.emit_ncols == 0) {
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

    // the source rows may be strided, e.g. the temporal planes of the patch embedding:
    // such a plan reads one row per emitted row, so the buffer covers the emitted matrix
    const size_t src_row_bytes_ext = p.src_row_stride != 0 ? (size_t) p.src_row_stride * st_dtype_size(src.dtype) : src_row_bytes;
    const size_t src_row_bytes_used = src_row_bytes_ext != src_row_bytes
        ? (size_t) (nrows - 1) * src_row_bytes_ext + (size_t) ncols * st_dtype_size(src.dtype)
        : (size_t) src_rows * src_row_bytes;

    std::vector<uint8_t> src_buf(src_row_bytes_used);
    src.shard->read(src.offs, src_buf.data(), src_buf.size());

    std::vector<float> scale_buf;
    int64_t scale_stride = 0;
    if (has_scale) {
        scale_stride = scale.ncols(); // number of 128 column blocks, or of groups for a packed weight
        const size_t n = (size_t) scale.nrows() * scale.ncols();
        std::vector<uint8_t> raw(n * st_dtype_size(scale.dtype));
        scale.shard->read(scale.offs, raw.data(), raw.size());
        scale_buf.resize(n);
        for (size_t i = 0; i < n; ++i) {
            scale_buf[i] = st_read_f32(raw.data(), scale.dtype, i);
        }
    }

    // a packed weight stores several values per element, the scales of a packed weight are per row
    const int64_t  pack_per_word = p.pack_bits > 0 ? 32 / p.pack_bits : 0;
    const uint32_t pack_mask     = p.pack_bits > 0 ? (1u << p.pack_bits) - 1 : 0;
    const int32_t  pack_bias     = p.pack_bits > 0 ? 1 << (p.pack_bits - 1) : 0;

    auto emit_rows = [&](int64_t r0, int64_t r1) {
        std::vector<float> row(ncols);
        for (int64_t r = r0; r < r1; ++r) {
            const int64_t sr = p.reorder.dim == 0 ? st_reorder_src(L, p.reorder, r) : r;
            const uint8_t * srow = src_buf.data() + (size_t) sr * src_row_bytes_ext;
            const float * sscale = !has_scale || p.scale_per_tensor ? nullptr :
                scale_buf.data() + (p.pack_bits > 0 || p.scale_per_row ? sr : sr / ST_BLOCK) * scale_stride;

            for (int64_t c = 0; c < ncols; ++c) {
                const int64_t sc = p.reorder.dim == 1 ? st_reorder_src(L, p.reorder, c) : c;
                float v;
                if (p.pack_bits > 0) {
                    const uint32_t word = ((const uint32_t *) srow)[sc / pack_per_word];
                    const int32_t  q    = (int32_t) ((word >> ((sc % pack_per_word) * p.pack_bits)) & pack_mask) - pack_bias;
                    v = (float) q * sscale[sc / p.pack_group];
                } else if (src.dtype == ST_DT_F8) {
                    v = st_e4m3_table()[srow[sc]] * (p.scale_per_tensor ? scale_buf[0] :
                                                     p.scale_per_row ? sscale[0] : sscale[sc / ST_BLOCK]);
                } else {
                    v = st_read_f32(srow, src.dtype, sc);
                }
                if (p.add_one) {
                    v += 1.0f;
                }
                if (p.neg_exp) {
                    v = -expf(v);
                }
                row[c] = v;
            }

            uint8_t * drow = dst + (size_t) r * dst_row_bytes;
            if (type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_0) {
                ggml_quantize_chunk(type, row.data(), drow, 0, 1, ncols, nullptr);
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
    const int64_t n_threads = (src.dtype == ST_DT_F8 || p.pack_bits > 0) ?
        (int64_t) std::min<unsigned>(st_threads(), (unsigned) nrows) : 1;
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

// NVFP4 weights are stored with 2 values per byte and one E4M3 scale per group of 16 columns,
// ggml packs them in blocks of [4 scales, 32 packed bytes] over 64 columns
static void st_emit_nvfp4(const st_loader & L, const st_plan & p, uint8_t * out) {
    const int64_t qk          = ggml_blck_size(GGML_TYPE_NVFP4);
    const int64_t n_sub       = qk / 16;    // one scale per 16 values
    const int64_t sub_bytes   = 16 / 2;     // packed bytes of one group
    const int64_t block_bytes = n_sub + qk / 2;

    const int64_t nrows = p.ne[1];
    const int64_t ncols = p.ne[0];

    if (ncols % qk != 0) {
        throw std::runtime_error(string_format("row length %lld of tensor '%s' is not a multiple of %lld",
                (long long) ncols, p.name.c_str(), (long long) qk));
    }
    const int64_t n_block = ncols / qk;

    const size_t src_row_bytes   = (size_t) p.src.ne[1];   // 2 values per byte
    const size_t scale_row_bytes = (size_t) p.scale.ne[1]; // 16 values per scale
    const size_t dst_row_bytes   = ggml_row_size(GGML_TYPE_NVFP4, ncols);

    const int64_t src_rows   = p.src.ne[0];
    const int64_t scale_rows = p.scale.ne[0];
    if (src_rows != nrows || scale_rows != nrows || scale_row_bytes * 16 != (size_t) ncols || src_row_bytes * 2 != (size_t) ncols) {
        throw std::runtime_error(string_format("unexpected NVFP4 layout for tensor '%s'", p.name.c_str()));
    }

    // a column permutation must move whole blocks, the packed layout has no room to split one
    if (p.reorder.dim == 1 && (p.reorder.offs % qk != 0 || p.reorder.head % qk != 0)) {
        throw std::runtime_error(string_format("cannot permute the columns of NVFP4 tensor '%s'", p.name.c_str()));
    }

    std::vector<uint8_t> src_buf((size_t) src_rows * src_row_bytes);
    p.src.shard->read(p.src.offs, src_buf.data(), src_buf.size());
    std::vector<uint8_t> scale_buf((size_t) scale_rows * scale_row_bytes);
    p.scale.shard->read(p.scale.offs, scale_buf.data(), scale_buf.size());

    // the first half of a group is packed in the low nibbles, the second half in the high nibbles
    auto nibble = [](uint8_t byte, int half) {
        return (uint8_t) ((byte >> (4 * half)) & 0x0F);
    };

    auto emit_rows = [&](int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; ++r) {
            const int64_t sr = p.reorder.dim == 0 ? st_reorder_src(L, p.reorder, r) : r;
            const uint8_t * srow       = src_buf.data()   + (size_t) sr * src_row_bytes;
            const uint8_t * srow_scale = scale_buf.data() + (size_t) sr * scale_row_bytes;
            uint8_t * drow = out + (size_t) r * dst_row_bytes;

            for (int64_t b = 0; b < n_block; ++b) {
                const int64_t sc = p.reorder.dim == 1 ? st_reorder_src(L, p.reorder, b * qk) / qk : b;

                uint8_t * dblock = drow + (size_t) b * (size_t) block_bytes;
                for (int64_t g = 0; g < n_sub; ++g) {
                    // ggml reads the scales as unsigned, so the sign bit of the source E4M3 value is dropped
                    dblock[g] = (uint8_t) (srow_scale[sc * n_sub + g] & 0x7F);

                    const uint8_t * sgroup = srow + (size_t) (sc * n_sub + g) * sub_bytes;
                    for (int64_t j = 0; j < sub_bytes; ++j) {
                        const uint8_t lo = nibble(sgroup[j / 2], j % 2);
                        const uint8_t hi = nibble(sgroup[sub_bytes / 2 + j / 2], j % 2);
                        dblock[n_sub + g * sub_bytes + j] = (uint8_t) (lo | (hi << 4));
                    }
                }
            }
        }
    };

    const int64_t n_threads = (int64_t) std::min<unsigned>(st_threads(), (unsigned) nrows);
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

// the fp8 weights of a checkpoint keep their codes, the emitter only adds the scale of every block of 128 values
static void st_emit_f8_native(const st_loader & L, const st_plan & p, uint8_t * out) {
    const int64_t qk    = ggml_blck_size(GGML_TYPE_F8_E4M3);
    const size_t  block_bytes = ggml_type_size(GGML_TYPE_F8_E4M3);
    const int64_t nrows = p.ne[1];
    const int64_t ncols = p.ne[0];

    if (ncols % qk != 0) {
        throw std::runtime_error(string_format("row length %lld of tensor '%s' is not a multiple of %lld",
                (long long) ncols, p.name.c_str(), (long long) qk));
    }
    const int64_t n_block = ncols / qk;
    const size_t  row_bytes = (size_t) n_block * block_bytes;

    const int64_t src_rows = p.src.ne[0];
    const int64_t src_cols = p.src.ne[1];
    if (src_rows != nrows || src_cols != ncols) {
        throw std::runtime_error(string_format("unexpected fp8 layout for tensor '%s'", p.name.c_str()));
    }

    // a column permutation has to move whole blocks, the block layout has no room to split one
    if (p.reorder.dim == 1 && (p.reorder.offs % qk != 0 || p.reorder.head % qk != 0)) {
        throw std::runtime_error(string_format("cannot permute the columns of fp8 tensor '%s'", p.name.c_str()));
    }

    std::vector<uint8_t> src_buf((size_t) src_rows * src_cols);
    p.src.shard->read(p.src.offs, src_buf.data(), src_buf.size());

    // the scale of the source is one value per block, or one per row, or one for the whole tensor
    std::vector<float> scale_buf;
    int64_t scale_stride = 1;
    if (p.has_scale) {
        const st_ref & sc = p.scale;
        scale_stride = sc.ndim >= 2 ? sc.ne[1] : 1;
        const size_t n = (size_t) sc.nrows() * sc.ncols();
        std::vector<uint8_t> raw(n * st_dtype_size(sc.dtype));
        sc.shard->read(sc.offs, raw.data(), raw.size());
        scale_buf.resize(n);
        for (size_t i = 0; i < n; ++i) {
            scale_buf[i] = st_read_f32(raw.data(), sc.dtype, i);
        }
    }

    for (int64_t r = 0; r < nrows; ++r) {
        const int64_t sr = p.reorder.dim == 0 ? st_reorder_src(L, p.reorder, r) : r;
        const uint8_t * srow = src_buf.data() + (size_t) sr * src_cols;
        uint8_t * drow = out + (size_t) r * row_bytes;

        for (int64_t b = 0; b < n_block; ++b) {
            const int64_t sc = p.reorder.dim == 1 ? st_reorder_src(L, p.reorder, b * qk) / qk : b;

            float d = 1.0f;
            if (p.has_scale) {
                if (p.scale_per_tensor) {
                    d = scale_buf[0];
                } else if (p.scale_per_row) {
                    d = scale_buf[sr];
                } else {
                    d = scale_buf[(sr / 128) * scale_stride + sc];
                }
            }

            uint8_t * dblock = drow + (size_t) b * block_bytes;
            memcpy(dblock, &d, sizeof(d));
            memcpy(dblock + sizeof(d), srow + sc * qk, qk);
        }
    }

}

// MLX affine 2-bit weights: packed uint32s + fp16 group-128 scales -> PQ2_0 blocks (34 bytes: 2-byte scale + 32-byte bits)
static void st_emit_mlx_pq2(const st_loader & L, const st_plan & p, uint8_t * dst) {
    const int64_t qk          = ggml_blck_size(GGML_TYPE_PQ2_0);
    const size_t  block_bytes = ggml_type_size(GGML_TYPE_PQ2_0);
    const int64_t nrows       = p.emit_nrows > 0 ? p.emit_nrows : (p.ndim >= 2 ? p.ne[1] : 1);
    const int64_t ncols       = p.emit_ncols > 0 ? p.emit_ncols : p.ne[0];

    if (ncols % qk != 0) {
        throw std::runtime_error(string_format("row length %lld of tensor '%s' is not a multiple of %lld",
                (long long) ncols, p.name.c_str(), (long long) qk));
    }
    const int64_t n_block = ncols / qk;

    const size_t row_weight_bytes = (size_t) n_block * 32;
    const size_t row_scale_bytes  = (size_t) n_block * 2;
    const size_t dst_row_bytes    = (size_t) n_block * block_bytes;

    const size_t total_weight_bytes = (size_t) nrows * row_weight_bytes;
    const size_t total_scale_bytes  = (size_t) nrows * row_scale_bytes;

    std::vector<uint8_t> weight_buf(total_weight_bytes);
    p.src.shard->read(p.src.offs, weight_buf.data(), weight_buf.size());

    std::vector<uint8_t> scale_buf(total_scale_bytes);
    p.scale.shard->read(p.scale.offs, scale_buf.data(), scale_buf.size());

    auto emit_rows = [&](int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; ++r) {
            const int64_t sr = p.reorder.dim == 0 ? st_reorder_src(L, p.reorder, r) : r;
            const uint8_t * srow_s = scale_buf.data() + (size_t) sr * row_scale_bytes;
            const uint8_t * srow_w = weight_buf.data() + (size_t) sr * row_weight_bytes;
            uint8_t       * drow   = dst + (size_t) r * dst_row_bytes;

            for (int64_t b = 0; b < n_block; ++b) {
                uint8_t * dblock = drow + (size_t) b * block_bytes;
                memcpy(dblock, srow_s + (size_t) b * 2, 2);
                memcpy(dblock + 2, srow_w + (size_t) b * 32, 32);
            }
        }
    };

    const int64_t n_threads = (int64_t) std::min<unsigned>(st_threads(), (unsigned) nrows);
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


// exl3 (ExLlamaV3): a quantized linear is stored as coded trellis indices plus one factor per
// input and per output channel. Decoding follows the reference implementation: 16x16 tiles of
// 256 indices of 'bits' width each, a 16 bit window sliding through the "3 instruction" codebook,
// then a Hadamard transform of 128 along both axes with the factors applied between them.
static float st_exl3_codebook(uint32_t w) {
    // mul1 codebook: multiply, sum the four bytes with 0x6400, read the result as fp16 and rescale
    const uint32_t x = w * 0x83DCD12Du;
    const uint32_t sum = (x & 0xFFu) + ((x >> 8) & 0xFFu) + ((x >> 16) & 0xFFu) + ((x >> 24) & 0xFFu) + 0x6400u;
    const float v = ggml_fp16_to_fp32((uint16_t) sum);
    const float k_inv  = ggml_fp16_to_fp32(0x1eee); //  1 / 147.7
    const float k_bias = ggml_fp16_to_fp32(0xc931); // -10.39
    return fmaf(v, k_inv, k_bias);
}

static const float * st_exl3_codebook_table() {
    static std::vector<float> lut;
    static std::once_flag flag;
    std::call_once(flag, []() {
        lut.resize(65536);
        for (uint32_t i = 0; i < 65536; ++i) {
            lut[i] = st_exl3_codebook(i);
        }
    });
    return lut.data();
}

static void st_exl3_fwht(float * x, int64_t n) {
    // natural order Sylvester Hadamard, the same matrix the reference builds
    for (int64_t h = 1; h < n; h *= 2) {
        for (int64_t i = 0; i < n; i += 2 * h) {
            for (int64_t j = 0; j < h; ++j) {
                const float a = x[i + j], b = x[i + j + h];
                x[i + j] = a + b;
                x[i + j + h] = a - b;
            }
        }
    }
    const float norm = 1.0f / sqrtf((float) n);
    for (int64_t i = 0; i < n; ++i) {
        x[i] *= norm;
    }
}

static void st_emit_exl3(const st_loader & L, const st_plan & p, uint8_t * out) {
    (void) L;
    const int64_t bits  = p.exl3_bits;
    const int64_t n_in  = p.ne[0];
    const int64_t n_out = p.ne[1];
    const bool swapped  = p.src.ne[0] * 16 != n_in; // the trellis is stored output major
    const int64_t tiles_k = swapped ? p.src.ne[1] : p.src.ne[0];
    const int64_t tiles_n = swapped ? p.src.ne[0] : p.src.ne[1];

    if (bits < 1 || bits > 8 || p.src.ne[2] != 16 * bits ||
        tiles_k * 16 != n_in || tiles_n * 16 != n_out) {
        throw std::runtime_error(string_format("unexpected exl3 trellis layout of tensor '%s'", p.name.c_str()));
    }
    if (n_in % 128 != 0 || n_out % 128 != 0) {
        throw std::runtime_error(string_format("exl3 tensor '%s' is not a multiple of 128 on both axes", p.name.c_str()));
    }

    // the tile is permuted into the order the encoder's tensor core layout expects
    int perm[256];
    for (int t = 0; t < 32; ++t) {
        const int r0 = (t % 4) * 2, c0 = t / 4;
        const int r[4] = { r0, r0 + 1, r0 + 8, r0 + 9 };
        const int c[2] = { c0, c0 + 8 };
        for (int o = 0; o < 8; ++o) {
            perm[t * 8 + o] = r[o % 4] * 16 + c[o / 4];
        }
    }

    const int64_t nw = 8 * bits; // uint32 per packed tile
    std::vector<uint32_t> trellis((size_t) tiles_k * tiles_n * nw);
    p.src.shard->read(p.src.offs, trellis.data(), trellis.size() * sizeof(uint32_t));

    std::vector<float> w((size_t) n_in * n_out);
    struct bit_extr {
        uint16_t i0;
        uint16_t i1;
        uint8_t  s0;
        uint8_t  e_div16;
        uint8_t  e_mod16;
    };
    bit_extr extr[256];
    for (int idx = 0; idx < 256; ++idx) {
        const int64_t b0 = idx * bits + bits - 16 + 256 * bits;
        const int64_t b1 = b0 + 16;
        const int64_t i0 = b0 / 32, i1 = (b1 - 1) / 32;
        const int64_t s0 = (i1 + 1) * 32 - b1;
        const int e = perm[idx];
        extr[idx] = { (uint16_t) (i0 % nw), (uint16_t) (i1 % nw), (uint8_t) s0, (uint8_t) (e / 16), (uint8_t) (e % 16) };
    }

    const float * codebook = st_exl3_codebook_table();

    auto decode_rows = [&](int64_t tk0, int64_t tk1) {
    for (int64_t tk = tk0; tk < tk1; ++tk) {
        for (int64_t tn = 0; tn < tiles_n; ++tn) {
            const int64_t ti = swapped ? tn : tk;
            const int64_t tj = swapped ? tk : tn;
            const uint32_t * tile = trellis.data() + ((size_t) ti * (swapped ? tiles_k : tiles_n) + tj) * nw;
            for (int64_t idx = 0; idx < 256; ++idx) {
                const auto & ex = extr[idx];
                const uint32_t a = tile[ex.i0], b = tile[ex.i1];
                const uint32_t w0 = (uint32_t) ((((uint64_t) a << 32) | b) >> ex.s0) & 0xFFFFu;
                w[(size_t) (tk * 16 + ex.e_div16) * n_out + (tn * 16 + ex.e_mod16)] = codebook[w0];
            }
        }
    }
    };
    {
        const int64_t n_threads = (int64_t) std::min<unsigned>(st_threads(), (unsigned) tiles_k);
        if (n_threads <= 1) {
            decode_rows(0, tiles_k);
        } else {
            std::vector<std::thread> workers;
            workers.reserve(n_threads - 1);
            const int64_t per_thread = (tiles_k + n_threads - 1) / n_threads;
            for (int64_t t = 1; t < n_threads; ++t) {
                workers.emplace_back(decode_rows, t * per_thread, std::min(tiles_k, (t + 1) * per_thread));
            }
            decode_rows(0, std::min(tiles_k, per_thread));
            for (auto & wkr : workers) {
                wkr.join();
            }
        }
    }

    std::vector<float> suh((size_t) n_in), svh((size_t) n_out);
    {
        auto readf = [](const st_ref & r, std::vector<float> & dst) {
            std::vector<uint8_t> raw((size_t) r.nrows() * r.ncols() * st_dtype_size(r.dtype));
            r.shard->read(r.offs, raw.data(), raw.size());
            for (size_t i = 0; i < dst.size(); ++i) {
                dst[i] = st_read_f32(raw.data(), r.dtype, i);
            }
        };
        readf(p.exl3_suh, suh);
        readf(p.exl3_svh, svh);
    }

    // the kernels compute y = svh * Had_n(inner * Had_k(suh * x)), so the weight is
    // W = svh * (H inner H) * suh: both transforms first, then the two factors
    auto fwht_cols = [&](int64_t o0, int64_t o1) {
        std::vector<float> scratch(128);
        for (int64_t kin = 0; kin < n_in; kin += 128) {
            for (int64_t o = o0; o < o1; ++o) {
                for (int64_t i = 0; i < 128; ++i) {
                    scratch[i] = w[(size_t) (kin + i) * n_out + o];
                }
                st_exl3_fwht(scratch.data(), 128);
                for (int64_t i = 0; i < 128; ++i) {
                    w[(size_t) (kin + i) * n_out + o] = scratch[i];
                }
            }
        }
    };
    {
        const int64_t n_threads = (int64_t) std::min<unsigned>(st_threads(), (unsigned) n_out);
        if (n_threads <= 1) {
            fwht_cols(0, n_out);
        } else {
            std::vector<std::thread> workers;
            workers.reserve(n_threads - 1);
            const int64_t per_thread = (n_out + n_threads - 1) / n_threads;
            for (int64_t t = 1; t < n_threads; ++t) {
                workers.emplace_back(fwht_cols, t * per_thread, std::min(n_out, (t + 1) * per_thread));
            }
            fwht_cols(0, std::min(n_out, per_thread));
            for (auto & wkr : workers) {
                wkr.join();
            }
        }
    }
    auto fwht_rows = [&](int64_t kin0, int64_t kin1) {
        for (int64_t kin = kin0; kin < kin1; ++kin) {
            float * row = w.data() + (size_t) kin * n_out;
            for (int64_t nout = 0; nout < n_out; nout += 128) {
                st_exl3_fwht(row + nout, 128);
            }
            const float su = suh[kin];
            for (int64_t o = 0; o < n_out; ++o) {
                row[o] *= su * svh[o];
            }
        }
    };
    {
        const int64_t n_threads = (int64_t) std::min<unsigned>(st_threads(), (unsigned) n_in);
        if (n_threads <= 1) {
            fwht_rows(0, n_in);
        } else {
            std::vector<std::thread> workers;
            workers.reserve(n_threads - 1);
            const int64_t per_thread = (n_in + n_threads - 1) / n_threads;
            for (int64_t t = 1; t < n_threads; ++t) {
                workers.emplace_back(fwht_rows, t * per_thread, std::min(n_in, (t + 1) * per_thread));
            }
            fwht_rows(0, std::min(n_in, per_thread));
            for (auto & wkr : workers) {
                wkr.join();
            }
        }
    }

    // emit the rows of the weight, one output channel each
    const enum ggml_type type = p.type;
    const size_t dst_row_bytes = ggml_is_quantized(type) ? ggml_row_size(type, n_in) :
                                 type == GGML_TYPE_F32 ? (size_t) n_in * 4 : (size_t) n_in * 2;
    auto emit_rows = [&](int64_t o0, int64_t o1) {
    std::vector<float> row((size_t) n_in);
    for (int64_t o = o0; o < o1; ++o) {
        // the architecture may read the rows and columns of the weight in another order
        const int64_t so = p.reorder.dim == 0 ? st_reorder_src(L, p.reorder, o) : o;
        for (int64_t i = 0; i < n_in; ++i) {
            const int64_t si = p.reorder.dim == 1 ? st_reorder_src(L, p.reorder, i) : i;
            row[i] = w[(size_t) si * n_out + so];
        }
        uint8_t * drow = out + (size_t) o * dst_row_bytes;
        if (ggml_is_quantized(type)) {
            ggml_quantize_chunk(type, row.data(), drow, 0, 1, n_in, nullptr);
        } else if (type == GGML_TYPE_F32) {
            memcpy(drow, row.data(), (size_t) n_in * 4);
        } else if (type == GGML_TYPE_BF16) {
            for (int64_t i = 0; i < n_in; ++i) {
                ((uint16_t *) drow)[i] = st_f32_to_bf16(row[i]);
            }
        } else {
            ggml_fp32_to_fp16_row(row.data(), (ggml_fp16_t *) drow, n_in);
        }
    }
    };
    {
        const int64_t n_threads = (int64_t) std::min<unsigned>(st_threads(), (unsigned) n_out);
        if (n_threads <= 1) {
            emit_rows(0, n_out);
        } else {
            std::vector<std::thread> workers;
            workers.reserve(n_threads - 1);
            const int64_t per_thread = (n_out + n_threads - 1) / n_threads;
            for (int64_t t = 1; t < n_threads; ++t) {
                workers.emplace_back(emit_rows, t * per_thread, std::min(n_out, (t + 1) * per_thread));
            }
            emit_rows(0, std::min(n_out, per_thread));
            for (auto & wkr : workers) {
                wkr.join();
            }
        }
    }
}

static void st_emit(const st_loader & L, const st_plan & p, uint8_t * out) {
    if (p.exl3) {
        st_emit_exl3(L, p, out);
        return;
    }

    if (p.split_rows > 0) {
        // a flat source that the architecture reads as rows, e.g. the mamba2 group norm
        const int64_t n = p.ne[0] * p.ne[1];
        std::vector<uint8_t> buf((size_t) n * st_dtype_size(p.src.dtype));
        p.src.shard->read(p.src.offs, buf.data(), buf.size());
        for (int64_t i = 0; i < n; ++i) {
            float v = st_read_f32(buf.data(), p.src.dtype, i);
            if (p.add_one) {
                v += 1.0f;
            }
            if (p.neg_exp) {
                v = -expf(v);
            }
            if (p.type == GGML_TYPE_F32) {
                ((float *) out)[i] = v;
            } else if (p.type == GGML_TYPE_BF16) {
                ((uint16_t *) out)[i] = st_f32_to_bf16(v);
            } else {
                ((ggml_fp16_t *) out)[i] = ggml_fp32_to_fp16(v);
            }
        }
        return;
    }

    if (p.stack_reciprocal || !p.stack_src.empty()) {
        // one value per expert, read from a scalar tensor of each of them
        float * dst = (float *) out;
        for (size_t i = 0; i < p.stack_src.size(); ++i) {
            uint8_t buf[sizeof(float)];
            p.stack_src[i].shard->read(p.stack_src[i].offs, buf, sizeof(buf));
            float v;
            memcpy(&v, buf, sizeof(v));
            dst[i] = p.stack_reciprocal ? 1.0f / v : v;
        }
        return;
    }

    if (p.reciprocal) {
        // compressed-tensors stores the global scale as the reciprocal of the factor ggml applies
        uint8_t buf[sizeof(float)];
        p.src.shard->read(p.src.offs, buf, sizeof(buf));
        float v;
        memcpy(&v, buf, sizeof(v));
        v = 1.0f / v;
        memcpy(out, &v, sizeof(v));
        return;
    }

    if (p.rope_freqs) {
        // gemma 4 rotates a part of the head with the proportional rope, the rest is disabled with a huge factor
        float * dst = (float *) out;
        for (int64_t i = 0; i < p.ne[0]; ++i) {
            dst[i] = i < p.rope_freqs_rot ? 1.0f : 1e30f;
        }
        return;
    }

    if (!p.exps.empty() && p.exps_nvfp4) {
        // every expert keeps its own packed weights and E4M3 scales
        const size_t expert_bytes = ggml_row_size(GGML_TYPE_NVFP4, p.ne[0]) * p.ne[1];

        st_plan sub = p;
        sub.exps        = {};
        sub.exps_scale  = {};
        sub.exps_scale2 = {};
        sub.exps_iscale = {};
        sub.exps_nvfp4  = 0;
        sub.nvfp4       = 1;
        sub.ndim        = 2;
        sub.ne[2]       = 1;
        sub.ne[3]       = 1;

        for (int64_t e = 0; e < (int64_t) p.exps.size(); ++e) {
            sub.src   = p.exps[e];
            sub.scale = p.exps_scale[e];
            st_emit_nvfp4(L, sub, out + (size_t) e * expert_bytes);
        }
        return;
    }

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

    if (p.f8_native) {
        st_emit_f8_native(L, p, out);
        return;
    }

    if (p.nvfp4) {
        st_emit_nvfp4(L, p, out);
        return;
    }

    if (p.mlx_pq2) {
        st_emit_mlx_pq2(L, p, out);
        return;
    }
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
    if (p.exl3 || !p.verbatim) {
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

static void st_add_meta_agnes(gguf_context * meta, const json & cfg, const std::string & model_name);
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
        // a checkpoint can keep its MTP head in a shard of its own, on top of the ones the index lists
        const fs::path index_file = dir / "model.safetensors.index.json";
        if (fs::is_regular_file(index_file)) {
            const json index_json = st_read_json(index_file);
            for (const auto & [name, file] : index_json.at("weight_map").items()) {
                (void) file;
                if (name.rfind("mtp.", 0) == 0 || name.rfind("model.mtp.", 0) == 0) {
                    return true;
                }
            }
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
            const auto shard_header = json::parse(header);
            for (const auto & [name, info] : shard_header.items()) {
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

// all shards of a checkpoint, including the files the index does not list, like a separate MTP head
static std::vector<std::string> st_list_shards(const fs::path & dir) {
    std::vector<std::string> res;

    const fs::path index_file = dir / "model.safetensors.index.json";
    if (fs::is_regular_file(index_file)) {
        const json index_json = st_read_json(index_file);
        for (const auto & [name, file] : index_json.at("weight_map").items()) {
            (void) name;
            res.push_back(file.get<std::string>());
        }
    }

    std::error_code ec;
    for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
        if (it->path().extension() == ".safetensors") {
            res.push_back(it->path().filename().string());
        }
    }

    std::sort(res.begin(), res.end());
    res.erase(std::unique(res.begin(), res.end()), res.end());
    return res;
}

// compressed-tensors stores the int4 weights nibble packed, with one scale per group of columns
static void st_parse_quant_config(st_loader & L, const json & cfg) {
    if (!cfg.contains("quantization_config")) {
        return;
    }
    const json & q = cfg.at("quantization_config");
    const std::string method = st_json_value(q, "quant_method", std::string());
    if (method != "compressed-tensors" ||
        st_json_value(q, "format", std::string()) != "pack-quantized") {
        return;
    }

    const json & groups = q.at("config_groups");
    if (groups.size() != 1) {
        throw std::runtime_error(string_format("unsupported packed checkpoint: expected 1 config group, got %zu", groups.size()));
    }
    const json & w = groups.begin().value().at("weights");
    if (st_json_value(w, "strategy", std::string()) != "group" || st_json_value(w, "type", std::string("int")) != "int" ||
        !st_json_value(w, "symmetric", false)) {
        throw std::runtime_error("unsupported packed checkpoint: only symmetric int weights quantized per group are supported");
    }

    L.pack_bits  = w.at("num_bits").get<int>();
    L.pack_group = w.at("group_size").get<int64_t>();
    if (L.pack_bits != 4 || L.pack_group <= 0) {
        throw std::runtime_error(string_format("unsupported packed checkpoint: %d bit weights with group size %lld",
                L.pack_bits, (long long) L.pack_group));
    }
}

// ---------------------------------------------------------------------------
// generic text models: architectures that have no hand written mapping
// ---------------------------------------------------------------------------

// ways in which a checkpoint family differs from the plain decoder skeleton
enum st_generic_flag {
    ST_GF_NORM_ADD_ONE = 1u << 0, // the checkpoint stores the RMSNorm weights as (1 + w)
    ST_GF_ROPE_PERMUTE = 1u << 1, // q/k keep the two halves of every rotary pair apart
    ST_GF_GEMMA_NORMS  = 1u << 2, // post_attention_layernorm belongs to the attention output, the FFN has two norms
};

struct st_generic_arch {
    const char * hf;   // "model_type" of the checkpoint, lower case and without underscores
    const char * ggml; // architecture to load it as
    unsigned     flags;
};

static const st_generic_arch st_generic_archs[] = {
    { "llama",      "llama",      ST_GF_ROPE_PERMUTE },
    { "mistral",    "llama",      ST_GF_ROPE_PERMUTE },
    { "mixtral",    "llama",      ST_GF_ROPE_PERMUTE },
    { "qwen2",      "qwen2",      0 },
    { "qwen2moe",   "qwen2moe",   0 },
    { "qwen3",      "qwen3",      0 },
    { "qwen3moe",   "qwen3moe",   0 },
    { "gemma",      "gemma",      ST_GF_NORM_ADD_ONE },
    { "gemma2",     "gemma2",     ST_GF_NORM_ADD_ONE | ST_GF_GEMMA_NORMS },
    { "gemma3text", "gemma3",     ST_GF_NORM_ADD_ONE | ST_GF_GEMMA_NORMS },
    { "olmo2",      "olmo2",      0 },
    { "phi3",       "phi3",       0 },
    { "starcoder2", "starcoder2", 0 },
};

static std::string st_normalize_name(std::string name) {
    std::string res;
    for (const char c : name) {
        if (c != '_') {
            res += (char) tolower(c);
        }
    }
    return res;
}

// finds the generic handling of a checkpoint, nullptr when there is none
static const st_generic_arch * st_generic_lookup(const json & cfg) {
    std::vector<std::string> names;
    if (cfg.contains("model_type")) {
        names.push_back(cfg.at("model_type").get<std::string>());
    }
    if (cfg.contains("architectures")) {
        for (const auto & a : cfg.at("architectures")) {
            std::string n = a.get<std::string>();
            for (const char * suffix : { "ForConditionalGeneration", "ForCausalLM", "Model" }) {
                const size_t len = strlen(suffix);
                if (n.size() > len && n.compare(n.size() - len, len, suffix) == 0) {
                    n = n.substr(0, n.size() - len);
                    break;
                }
            }
            names.push_back(n);
        }
    }

    for (const auto & name : names) {
        const std::string key = st_normalize_name(name);
        for (const auto & a : st_generic_archs) {
            if (key == a.hf) {
                return &a;
            }
        }
    }
    return nullptr;
}

// maps a tensor of a plain decoder: model.layers.<i>.self_attn.q_proj.weight and friends
static bool st_map_tensor_generic(const std::string & name, int64_t n_layer, const st_generic_arch & arch,
        st_map & res, int64_t & expert, std::string & expert_base) {
    (void) n_layer;

    const int norm_add = (arch.flags & ST_GF_NORM_ADD_ONE) ? 1 : 0;
    const int rope     = (arch.flags & ST_GF_ROPE_PERMUTE) ? 1 : 0;

    if (name == "model.embed_tokens.weight" || name == "embed_tokens.weight") {
        res.gguf = "token_embd.weight";
        return true;
    }
    if (name == "model.norm.weight") {
        res.gguf    = "output_norm.weight";
        res.role    = ST_ROLE_F32;
        res.add_one = norm_add;
        return true;
    }
    if (name == "lm_head.weight") {
        res.gguf = "output.weight";
        return true;
    }

    static const char * prefix = "model.layers.";
    if (name.rfind(prefix, 0) != 0) {
        return false;
    }
    const size_t pos = strlen(prefix);
    const size_t dot = name.find('.', pos);
    if (dot == std::string::npos) {
        return false;
    }
    const int il = atoi(name.substr(pos, dot - pos).c_str());
    const std::string rest = name.substr(dot + 1);
    const std::string base = "blk." + std::to_string(il) + ".";

    // the normalization between the attention output and the residual, named after what it feeds
    if (rest == "post_attention_layernorm.weight") {
        res.gguf    = base + ((arch.flags & ST_GF_GEMMA_NORMS) ? "post_attention_norm.weight" : "ffn_norm.weight");
        res.role    = ST_ROLE_F32;
        res.add_one = norm_add;
        return true;
    }

    // routed experts are stacked into a single 3d tensor
    if (rest.rfind("mlp.experts.", 0) == 0 || rest.rfind("block_sparse_moe.experts.", 0) == 0) {
        const std::string tail = rest.substr(rest.find("experts.") + strlen("experts."));
        const size_t p = tail.find('.');
        if (p == std::string::npos) {
            return false;
        }
        expert = atoll(tail.substr(0, p).c_str());
        const std::string proj = tail.substr(p + 1);
        if (proj == "gate_proj.weight" || proj == "w1.weight") {
            expert_base = base + "ffn_gate_exps.weight";
        } else if (proj == "up_proj.weight" || proj == "w3.weight") {
            expert_base = base + "ffn_up_exps.weight";
        } else if (proj == "down_proj.weight" || proj == "w2.weight") {
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
        int norm; // 1: the +1 of a (1 + w) normalization applies when the family uses it
        int rope; // 0: plain rows, 1: query rows, 2: key rows
    };

    static const entry entries[] = {
        { "input_layernorm.weight",            "attn_norm.weight",      ST_ROLE_F32,    1, 0 },
        { "self_attn.q_proj.weight",           "attn_q.weight",         ST_ROLE_WEIGHT, 0, 1 },
        { "self_attn.q_proj.bias",             "attn_q.bias",           ST_ROLE_F32,    0, 1 },
        { "self_attn.k_proj.weight",           "attn_k.weight",         ST_ROLE_WEIGHT, 0, 2 },
        { "self_attn.k_proj.bias",             "attn_k.bias",           ST_ROLE_F32,    0, 2 },
        { "self_attn.v_proj.weight",           "attn_v.weight",         ST_ROLE_WEIGHT, 0, 0 },
        { "self_attn.v_proj.bias",             "attn_v.bias",           ST_ROLE_F32,    0, 0 },
        { "self_attn.o_proj.weight",           "attn_output.weight",    ST_ROLE_WEIGHT, 0, 0 },
        { "self_attn.o_proj.bias",             "attn_output.bias",      ST_ROLE_F32,    0, 0 },
        { "self_attn.q_norm.weight",           "attn_q_norm.weight",    ST_ROLE_F32,    1, 0 },
        { "self_attn.k_norm.weight",           "attn_k_norm.weight",    ST_ROLE_F32,    1, 0 },
        { "mlp.gate_proj.weight",              "ffn_gate.weight",       ST_ROLE_WEIGHT, 0, 0 },
        { "mlp.up_proj.weight",                "ffn_up.weight",         ST_ROLE_WEIGHT, 0, 0 },
        { "mlp.down_proj.weight",              "ffn_down.weight",       ST_ROLE_WEIGHT, 0, 0 },
        { "mlp.gate.weight",                   "ffn_gate_inp.weight",   ST_ROLE_F32,    0, 0 },
        { "pre_feedforward_layernorm.weight",  "ffn_norm.weight",       ST_ROLE_F32,    1, 0 },
        { "post_feedforward_layernorm.weight", "post_ffw_norm.weight",  ST_ROLE_F32,    1, 0 },
    };

    for (const auto & e : entries) {
        if (rest == e.suffix) {
            res.gguf    = base + e.gguf;
            res.role    = e.role;
            res.add_one = e.norm ? norm_add : 0;
            if (e.rope && rope) {
                res.reorder = e.rope == 1 ? ST_RE_ROPE_Q : ST_RE_ROPE_K;
            }
            return true;
        }
    }

    return false;
}

// the metadata of a plain decoder, derived from the checkpoint configuration
static void st_add_meta_generic(st_loader & L, gguf_context * meta, const json & cfg, const std::string & dir_name, const st_generic_arch & arch) {
    const json & t = cfg.contains("text_config") ? cfg.at("text_config") : cfg;

    auto get_u32 = [&](std::initializer_list<const char *> keys, uint32_t def) {
        for (const char * key : keys) {
            if (t.contains(key) && !t.at(key).is_null()) {
                return t.at(key).get<uint32_t>();
            }
        }
        return def;
    };
    auto get_f32 = [&](std::initializer_list<const char *> keys, float def) {
        for (const char * key : keys) {
            if (t.contains(key) && !t.at(key).is_null()) {
                return t.at(key).get<float>();
            }
        }
        return def;
    };

    const std::string a = arch.ggml;
    auto add_u32 = [&](const char * suffix, uint32_t value) { gguf_set_val_u32(meta, (a + "." + suffix).c_str(), value); };
    auto add_f32 = [&](const char * suffix, float    value) { gguf_set_val_f32(meta, (a + "." + suffix).c_str(), value); };

    const uint32_t n_layer = get_u32({ "num_hidden_layers", "n_layers", "num_layers" }, 0);
    const uint32_t n_embd  = get_u32({ "hidden_size", "n_embd", "d_model" }, 0);
    const uint32_t n_head  = get_u32({ "num_attention_heads", "n_heads" }, 0);
    if (n_layer == 0 || n_embd == 0 || n_head == 0) {
        throw std::runtime_error("cannot read the text configuration of this checkpoint");
    }

    const uint32_t n_ff     = get_u32({ "intermediate_size", "ffn_dim", "intermediate_dim" }, 0);
    const uint32_t n_head_kv = get_u32({ "num_key_value_heads", "n_kv_heads" }, n_head);
    const uint32_t head_dim  = get_u32({ "head_dim" }, n_embd / n_head);

    L.n_head    = n_head;
    L.n_head_kv = n_head_kv;
    L.head_dim  = head_dim;

    gguf_set_val_str(meta, "general.architecture", a.c_str());
    gguf_set_val_str(meta, "general.name",         dir_name.c_str());
    gguf_set_val_str(meta, "general.type",         "model");

    add_u32("block_count",         n_layer);
    add_u32("context_length",      get_u32({ "max_position_embeddings", "n_ctx", "seq_length" }, 8192));
    add_u32("embedding_length",    n_embd);
    add_u32("feed_forward_length", n_ff);
    add_u32("vocab_size",          get_u32({ "vocab_size" }, 0));
    add_u32("attention.head_count",    n_head);
    add_u32("attention.head_count_kv", n_head_kv);
    add_u32("attention.key_length",    head_dim);
    add_u32("attention.value_length",  head_dim);
    add_f32("attention.layer_norm_rms_epsilon", get_f32({ "rms_norm_eps", "layer_norm_eps", "norm_eps" }, 1e-5f));

    add_f32("rope.freq_base", get_f32({ "rope_theta", "rotary_emb_base" }, 10000.0f));
    const float partial_rot = get_f32({ "partial_rotary_factor", "rope_pct" }, 1.0f);
    if (partial_rot < 1.0f && head_dim > 0) {
        add_u32("rope.dimension_count", (uint32_t) (head_dim * partial_rot));
    }

    // only the rope scaling types that llama.cpp knows are written
    if (t.contains("rope_scaling") && t.at("rope_scaling").is_object()) {
        const json & rs = t.at("rope_scaling");
        const std::string type = st_normalize_name(st_json_value(rs, "type", st_json_value(rs, "rope_type", std::string())));
        if (type == "linear" || type == "yarn" || type == "longrope") {
            gguf_set_val_str(meta, (a + ".rope.scaling.type").c_str(), type.c_str());
            add_f32("rope.scaling.factor", st_json_value(rs, "factor", 1.0f));
            if (rs.contains("original_max_position_embeddings")) {
                add_u32("rope.scaling.original_context_length", rs.at("original_max_position_embeddings").get<uint32_t>());
            }
        }
    }

    if (t.contains("sliding_window") && !t.at("sliding_window").is_null()) {
        add_u32("attention.sliding_window", t.at("sliding_window").get<uint32_t>());
    }
    if (t.contains("attn_logit_softcapping") && !t.at("attn_logit_softcapping").is_null()) {
        add_f32("attn_logit_softcapping", t.at("attn_logit_softcapping").get<float>());
    }
    if (t.contains("final_logit_softcapping") && !t.at("final_logit_softcapping").is_null()) {
        add_f32("final_logit_softcapping", t.at("final_logit_softcapping").get<float>());
    }

    const uint32_t n_expert = get_u32({ "num_experts", "num_local_experts" }, 0);
    if (n_expert > 0) {
        add_u32("expert_count",      n_expert);
        add_u32("expert_used_count", get_u32({ "num_experts_per_tok" }, 1));
        add_u32("expert_feed_forward_length", get_u32({ "moe_intermediate_size", "expert_intermediate_size" }, n_ff));
        const uint32_t n_ff_shexp = get_u32({ "shared_expert_intermediate_size" }, 0);
        if (n_ff_shexp > 0) {
            add_u32("expert_shared_feed_forward_length", n_ff_shexp);
        }
    }
}

// ---------------------------------------------------------------------------
// nemotron-h: mamba2, attention and moe blocks mixed by the layer pattern
// ---------------------------------------------------------------------------

// the text configuration of this family usually lives in llm_config, the caller merges it in
static void st_add_meta_nemotron(st_loader & L, gguf_context * meta, const json & t, const std::string & dir_name) {
    auto get_u32 = [&](std::initializer_list<const char *> keys, uint32_t def) {
        for (const char * key : keys) {
            if (st_json_set(t, key)) {
                return t.at(key).get<uint32_t>();
            }
        }
        return def;
    };
    auto get_f32 = [&](std::initializer_list<const char *> keys, float def) {
        for (const char * key : keys) {
            if (st_json_set(t, key)) {
                return t.at(key).get<float>();
            }
        }
        return def;
    };
    auto get_bool = [&](std::initializer_list<const char *> keys, bool def) {
        for (const char * key : keys) {
            if (st_json_set(t, key)) {
                return t.at(key).get<bool>();
            }
        }
        return def;
    };

    // M: mamba2, E: experts, *: attention
    const std::string pattern = st_json_value(t, "hybrid_override_pattern", std::string());
    const uint32_t n_layer = pattern.empty() ? get_u32({ "num_hidden_layers" }, 0) : (uint32_t) pattern.size();
    const uint32_t n_embd  = get_u32({ "hidden_size", "d_model" }, 0);
    const uint32_t n_head  = get_u32({ "num_attention_heads" }, 0);
    const uint32_t head_dim = get_u32({ "head_dim" }, 0);
    if (n_layer == 0 || n_embd == 0 || n_head == 0 || pattern.empty()) {
        throw std::runtime_error("cannot read the text configuration of this checkpoint");
    }

    const uint32_t n_kv     = get_u32({ "num_key_value_heads" }, n_head);
    const uint32_t n_ff     = get_u32({ "moe_intermediate_size" }, 0);
    const uint32_t n_expert = get_u32({ "n_routed_experts" }, 0);
    const uint32_t n_heads_mamba = get_u32({ "mamba_num_heads" }, 0);
    const uint32_t inner    = n_heads_mamba * get_u32({ "mamba_head_dim" }, 0);
    const float    eps      = get_f32({ "layer_norm_epsilon", "norm_eps" }, 1e-5f);

    L.ssm_groups = get_u32({ "n_groups" }, 1);

    // the feed forward length and the kv head count of a layer follow its kind, a zero marks the other kinds
    std::vector<int32_t> ffn_len(n_layer, 0);
    std::vector<int32_t> kv_len(n_layer, 0);
    for (uint32_t i = 0; i < n_layer; ++i) {
        if (pattern[i] == 'E') {
            ffn_len[i] = (int32_t) n_ff;
        } else if (pattern[i] == '*') {
            kv_len[i] = (int32_t) n_kv;
        }
    }

    const std::string a = "nemotron_h_moe";
    auto add_u32 = [&](const char * suffix, uint32_t value) { gguf_set_val_u32(meta, (a + "." + suffix).c_str(), value); };
    auto add_f32 = [&](const char * suffix, float    value) { gguf_set_val_f32(meta, (a + "." + suffix).c_str(), value); };

    gguf_set_val_str(meta, "general.architecture", a.c_str());
    gguf_set_val_str(meta, "general.name",         dir_name.c_str());
    gguf_set_val_str(meta, "general.type",         "model");

    add_u32("block_count",      n_layer);
    add_u32("context_length",   1u << 20); // the hybrid blocks rotate over a window the runtime chooses
    add_u32("embedding_length", n_embd);
    add_u32("vocab_size",       get_u32({ "vocab_size" }, 0));
    gguf_set_arr_data(meta, (a + ".feed_forward_length").c_str(), GGUF_TYPE_INT32, ffn_len.data(), ffn_len.size());
    add_u32("attention.head_count", n_head);
    gguf_set_arr_data(meta, (a + ".attention.head_count_kv").c_str(), GGUF_TYPE_INT32, kv_len.data(), kv_len.size());
    add_u32("attention.key_length",   head_dim);
    add_u32("attention.value_length", head_dim);
    add_f32("attention.layer_norm_rms_epsilon", eps);
    add_f32("attention.layer_norm_epsilon",     eps);
    add_f32("rope.freq_base", get_f32({ "rope_theta" }, 10000.0f));
    // the attention layers rotate a part of the head, the converter falls back to the uniform head width
    add_u32("rope.dimension_count", n_embd / n_head);
    gguf_set_val_bool(meta, (a + ".rope.scaling.finetuned").c_str(), false);

    add_u32("expert_count",                      n_expert);
    add_u32("expert_used_count",                 get_u32({ "num_experts_per_tok" }, 0));
    add_u32("expert_feed_forward_length",        n_ff);
    add_u32("expert_shared_feed_forward_length", get_u32({ "moe_shared_expert_intermediate_size" }, 0));
    add_u32("expert_shared_count",               get_u32({ "n_shared_experts" }, 0));
    add_u32("expert_group_count",                get_u32({ "n_group" }, 1));
    add_u32("expert_group_used_count",           get_u32({ "n_group_used" }, 1));
    gguf_set_val_bool(meta, (a + ".expert_weights_norm").c_str(), get_bool({ "norm_topk_prob" }, false));
    add_f32("expert_weights_scale", get_f32({ "routed_scaling_factor" }, 1.0f));

    add_u32("ssm.conv_kernel",     get_u32({ "conv_kernel" }, 0));
    add_u32("ssm.state_size",      get_u32({ "ssm_state_size" }, 0));
    add_u32("ssm.group_count",     (uint32_t) L.ssm_groups);
    add_u32("ssm.inner_size",      inner);
    add_u32("ssm.time_step_rank",  n_heads_mamba);
}

// maps a tensor of the backbone of a nemotron-h checkpoint
static bool st_map_tensor_nemotron(const std::string & name, int64_t ssm_groups,
        st_map & res, int64_t & expert, std::string & expert_base) {
    if (name == "language_model.backbone.embeddings.weight") {
        res.gguf = "token_embd.weight";
        return true;
    }
    if (name == "language_model.backbone.norm_f.weight") {
        res.gguf = "output_norm.weight";
        res.role = ST_ROLE_F32;
        return true;
    }
    if (name == "language_model.lm_head.weight") {
        res.gguf = "output.weight";
        return true;
    }

    static const char * prefix = "language_model.backbone.layers.";
    if (name.rfind(prefix, 0) != 0) {
        return false;
    }
    const size_t pos = strlen(prefix);
    const size_t dot = name.find('.', pos);
    if (dot == std::string::npos) {
        return false;
    }
    const int il = atoi(name.substr(pos, dot - pos).c_str());
    const std::string rest = name.substr(dot + 1);
    const std::string base = "blk." + std::to_string(il) + ".";

    if (rest == "norm.weight") {
        res.gguf = base + "attn_norm.weight";
        res.role = ST_ROLE_F32;
        return true;
    }
    if (rest == "mixer.gate.weight") {
        res.gguf = base + "ffn_gate_inp.weight";
        res.role = ST_ROLE_F32;
        return true;
    }
    if (rest == "mixer.gate.e_score_correction_bias") {
        res.gguf = base + "exp_probs_b.bias";
        res.role = ST_ROLE_F32;
        return true;
    }

    // the routed experts are stacked into one tensor, up and down keep their own scales
    if (rest.rfind("mixer.experts.", 0) == 0) {
        const std::string tail = rest.substr(strlen("mixer.experts."));
        const size_t p = tail.find('.');
        if (p == std::string::npos) {
            return false;
        }
        expert = atoll(tail.substr(0, p).c_str());
        const std::string proj = tail.substr(p + 1);
        if (proj == "up_proj.weight") {
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
        int neg_exp;
        int split_rows;
    };

    static const entry entries[] = {
        { "mixer.in_proj.weight",                  "ssm_in.weight",         ST_ROLE_WEIGHT, 0, 0 },
        { "mixer.conv1d.weight",                   "ssm_conv1d.weight",     ST_ROLE_F32,    0, 0 },
        { "mixer.conv1d.bias",                     "ssm_conv1d.bias",       ST_ROLE_F32,    0, 0 },
        { "mixer.dt_bias",                         "ssm_dt.bias",           ST_ROLE_F32,    0, 0 },
        { "mixer.A_log",                           "ssm_a",                 ST_ROLE_F32,    1, -1 },
        { "mixer.D",                               "ssm_d",                 ST_ROLE_F32,    0, -1 },
        { "mixer.norm.weight",                     "ssm_norm.weight",       ST_ROLE_F32,    0, 1 }, // split by the ssm groups
        { "mixer.out_proj.weight",                 "ssm_out.weight",        ST_ROLE_WEIGHT, 0, 0 },
        { "mixer.q_proj.weight",                   "attn_q.weight",         ST_ROLE_WEIGHT, 0, 0 },
        { "mixer.k_proj.weight",                   "attn_k.weight",         ST_ROLE_WEIGHT, 0, 0 },
        { "mixer.v_proj.weight",                   "attn_v.weight",         ST_ROLE_WEIGHT, 0, 0 },
        { "mixer.o_proj.weight",                   "attn_output.weight",    ST_ROLE_WEIGHT, 0, 0 },
        { "mixer.shared_experts.up_proj.weight",   "ffn_up_shexp.weight",   ST_ROLE_WEIGHT, 0, 0 },
        { "mixer.shared_experts.down_proj.weight", "ffn_down_shexp.weight", ST_ROLE_WEIGHT, 0, 0 },
    };

    for (const auto & e : entries) {
        if (rest == e.suffix) {
            res.gguf      = base + e.gguf;
            res.role      = e.role;
            res.neg_exp   = e.neg_exp;
            res.split_rows = e.split_rows == 0 ? 0 : (e.split_rows < 0 ? -1 : (int) ssm_groups);
            return true;
        }
    }

    return false;
}

// how the tensors of a checkpoint are mapped
enum st_map_kind {
    ST_MAP_QWEN,    // hand written mapping of the qwen3.5 / qwen3.6 families
    ST_MAP_GEMMA4,  // hand written mapping of the gemma 4 family
    ST_MAP_GENERIC, // plain decoder mapping
    ST_MAP_NEMOTRON, // hand written mapping of the nemotron-h family
};

// ---------------------------------------------------------------------------
// gemma 4: gemma 3 with a sliding window pattern, shared kv layers, per layer output scales
// and a proportional rope on the full attention layers
// ---------------------------------------------------------------------------

static bool st_map_tensor_gemma4(const std::string & name, int64_t n_layer, st_map & res, int64_t & expert, std::string & expert_base) {
    (void) n_layer;
    (void) expert;
    (void) expert_base;

    if (name == "model.language_model.embed_tokens.weight") {
        res.gguf = "token_embd.weight";
        return true;
    }
    if (name == "model.language_model.norm.weight") {
        // unlike the other gemma models, gemma 4 normalizes with the weight itself
        res.gguf = "output_norm.weight";
        res.role = ST_ROLE_F32;
        return true;
    }

    // the frequency factors of the full attention layers are generated, not stored
    if (name == "rope_freqs.weight") {
        res.gguf = "rope_freqs.weight";
        res.role = ST_ROLE_F32;
        return true;
    }

    // the vision and audio embedders are not part of the text model
    if (name.rfind("model.vision_embedder.", 0) == 0 || name.rfind("model.embed_vision.", 0) == 0 ||
        name.rfind("model.embed_audio.", 0) == 0 || name.rfind("model.visual.", 0) == 0) {
        return false;
    }

    static const char * prefix = "model.language_model.layers.";
    if (name.rfind(prefix, 0) != 0) {
        return false;
    }
    const size_t pos = strlen(prefix);
    const size_t dot = name.find('.', pos);
    if (dot == std::string::npos) {
        return false;
    }
    const int il = atoi(name.substr(pos, dot - pos).c_str());
    const std::string rest = name.substr(dot + 1);
    const std::string base = "blk." + std::to_string(il) + ".";

    struct entry {
        const char * suffix;
        const char * gguf;
        enum st_role role;
    };

    static const entry entries[] = {
        { "input_layernorm.weight",            "attn_norm.weight",        ST_ROLE_F32 },
        { "post_attention_layernorm.weight",   "post_attention_norm.weight", ST_ROLE_F32 },
        { "pre_feedforward_layernorm.weight",  "ffn_norm.weight",         ST_ROLE_F32 },
        { "post_feedforward_layernorm.weight", "post_ffw_norm.weight",     ST_ROLE_F32 },
        { "layer_scalar",                      "layer_output_scale.weight", ST_ROLE_F32 },
        { "self_attn.q_proj.weight",           "attn_q.weight",           ST_ROLE_WEIGHT },
        { "self_attn.k_proj.weight",           "attn_k.weight",           ST_ROLE_WEIGHT },
        { "self_attn.v_proj.weight",           "attn_v.weight",           ST_ROLE_WEIGHT },
        { "self_attn.o_proj.weight",           "attn_output.weight",      ST_ROLE_WEIGHT },
        { "self_attn.q_norm.weight",           "attn_q_norm.weight",      ST_ROLE_F32 },
        { "self_attn.k_norm.weight",           "attn_k_norm.weight",      ST_ROLE_F32 },
        { "mlp.gate_proj.weight",              "ffn_gate.weight",         ST_ROLE_WEIGHT },
        { "mlp.up_proj.weight",                "ffn_up.weight",           ST_ROLE_WEIGHT },
        { "mlp.down_proj.weight",              "ffn_down.weight",         ST_ROLE_WEIGHT },
    };

    for (const auto & e : entries) {
        if (rest == e.suffix) {
            res.gguf = base + e.gguf;
            res.role = e.role;
            return true;
        }
    }

    return false;
}

static void st_add_meta_gemma4(gguf_context * meta, const json & cfg, const std::string & dir_name) {
    const json & t = cfg.contains("text_config") ? cfg.at("text_config") : cfg;

    auto get_u32 = [&](std::initializer_list<const char *> keys, uint32_t def) {
        for (const char * key : keys) {
            if (t.contains(key) && !t.at(key).is_null()) {
                return t.at(key).get<uint32_t>();
            }
        }
        return def;
    };
    auto get_f32 = [&](std::initializer_list<const char *> keys, float def) {
        for (const char * key : keys) {
            if (t.contains(key) && !t.at(key).is_null()) {
                return t.at(key).get<float>();
            }
        }
        return def;
    };

    const std::string a = "gemma4";
    auto add_u32 = [&](const char * suffix, uint32_t value) { gguf_set_val_u32(meta, (a + "." + suffix).c_str(), value); };
    auto add_f32 = [&](const char * suffix, float    value) { gguf_set_val_f32(meta, (a + "." + suffix).c_str(), value); };

    const uint32_t n_layer = get_u32({ "num_hidden_layers" }, 0);
    const uint32_t n_head  = get_u32({ "num_attention_heads" }, 0);
    const uint32_t head_dim      = get_u32({ "head_dim" }, 0);
    const uint32_t head_dim_full = get_u32({ "global_head_dim" }, head_dim);
    if (n_layer == 0 || n_head == 0 || head_dim == 0) {
        throw std::runtime_error("cannot read the text configuration of this checkpoint");
    }

    // the sliding window layers keep their own head count, the full attention layers use the global one
    const uint32_t n_kv_swa  = get_u32({ "num_key_value_heads" }, n_head);
    const uint32_t n_kv_full = get_u32({ "num_global_key_value_heads" }, n_kv_swa);

    std::vector<uint8_t> swa;
    std::vector<int32_t> n_kv;
    swa.reserve(n_layer);
    n_kv.reserve(n_layer);
    for (const auto & type : t.at("layer_types")) {
        const bool is_swa = type.get<std::string>() == "sliding_attention";
        swa.push_back(is_swa ? 1 : 0);
        n_kv.push_back((int32_t) (is_swa ? n_kv_swa : n_kv_full));
    }
    if (swa.size() != n_layer) {
        throw std::runtime_error("layer_types does not match num_hidden_layers");
    }

    // the rope of the full attention layers is proportional, the rest of the head is disabled by the frequency factors
    const json rope = st_json_value(t, "rope_parameters", json::object());
    const json rope_full = rope.contains("full_attention") ? rope.at("full_attention") : json::object();
    const json rope_swa  = rope.contains("sliding_attention") ? rope.at("sliding_attention") : json::object();

    gguf_set_val_str(meta, "general.architecture", a.c_str());
    gguf_set_val_str(meta, "general.name",         dir_name.c_str());
    gguf_set_val_str(meta, "general.type",         "model");

    add_u32("block_count",         n_layer);
    add_u32("context_length",      get_u32({ "max_position_embeddings" }, 8192));
    add_u32("embedding_length",    get_u32({ "hidden_size" }, 0));
    add_u32("feed_forward_length", get_u32({ "intermediate_size" }, 0));
    add_u32("vocab_size",          get_u32({ "vocab_size" }, 0));
    add_u32("attention.head_count", n_head);
    gguf_set_arr_data(meta, (a + ".attention.head_count_kv").c_str(), GGUF_TYPE_INT32, n_kv.data(), n_kv.size());
    add_u32("attention.key_length",     head_dim_full);
    add_u32("attention.value_length",   head_dim_full);
    add_u32("attention.key_length_swa",   head_dim);
    add_u32("attention.value_length_swa", head_dim);
    add_f32("attention.layer_norm_rms_epsilon", get_f32({ "rms_norm_eps" }, 1e-6f));
    add_u32("attention.shared_kv_layers", get_u32({ "num_kv_shared_layers" }, 0));
    add_u32("embedding_length_per_layer_input", get_u32({ "hidden_size_per_layer_input" }, 0));
    gguf_set_arr_data(meta, (a + ".attention.sliding_window_pattern").c_str(), GGUF_TYPE_BOOL, swa.data(), swa.size());
    if (t.contains("sliding_window") && !t.at("sliding_window").is_null()) {
        add_u32("attention.sliding_window", t.at("sliding_window").get<uint32_t>());
    }
    if (t.contains("final_logit_softcapping") && !t.at("final_logit_softcapping").is_null()) {
        add_f32("final_logit_softcapping", t.at("final_logit_softcapping").get<float>());
    }

    add_f32("rope.freq_base",     st_json_value(rope_full, "rope_theta", 1000000.0f));
    add_f32("rope.freq_base_swa", st_json_value(rope_swa, "rope_theta", 10000.0f));
    // the proportional rope rotates the whole head, the frequency factors disable what stays unrotated
    add_u32("rope.dimension_count",     head_dim_full);
    add_u32("rope.dimension_count_swa", (uint32_t) (head_dim * st_json_value(rope_swa, "partial_rotary_factor", 1.0f)));
}

static void st_build_plans(st_loader & L, gguf_context * meta, const fs::path & dir, const json & cfg, enum st_mode mode,
        enum st_map_kind kind, const st_generic_arch * gen) {
    const json & tc = cfg.contains("text_config") ? cfg.at("text_config") : cfg;
    const char * expert_key = st_json_set(tc, "num_experts")       ? "num_experts" :
                              st_json_set(tc, "num_local_experts") ? "num_local_experts" :
                              st_json_set(tc, "n_routed_experts")  ? "n_routed_experts" : nullptr;
    const bool   moe     = expert_key != nullptr;
    const int    n_layer = tc.at("num_hidden_layers").get<int>();
    const uint32_t n_expert = moe ? st_json_value(tc, expert_key, 0u) : 0;

    const std::vector<std::string> part_names = st_list_shards(dir);

    if (part_names.empty()) {
        throw std::runtime_error("no safetensors shards found in " + dir.string());
    }

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
    std::unordered_map<std::string, std::vector<st_ref>> exps_scale2;
    std::unordered_map<std::string, std::vector<st_ref>> exps_iscale;
    std::unordered_map<std::string, int>                 exps_nvfp4;
    std::unordered_map<std::string, int>                 exps_nvfp4_recip;
    std::unordered_map<std::string, int>                 exps_f8_mode; // fp8 scale: 0 per 128 columns, 1 per tensor, 2 per row

    auto add_plan = [&](std::unique_ptr<st_plan> plan) {
        if (L.plans.count(plan->name) != 0) {
            throw std::runtime_error("duplicated tensor '" + plan->name + "'");
        }
        L.plans[plan->name] = plan.get();
        L.storage.push_back(std::move(plan));
    };

    size_t n_skipped = 0;

    for (const auto & [src_name, ref] : tensors) {
        std::string name = src_name; // a synthesized name is fed to the architecture below
        if (st_ends_with(name, "_scale_inv")) {
            continue;
        }
        // the scale and the shape of a packed weight are read together with the weight itself
        if (st_ends_with(name, ".weight_scale") || st_ends_with(name, ".weight_shape") ||
            st_ends_with(name, ".weight_zero_point") || st_ends_with(name, ".weight_scale_2") ||
            st_ends_with(name, ".input_scale") ||
            st_ends_with(name, ".scales") || st_ends_with(name, ".biases") || st_ends_with(name, ".signs")) {
            continue;
        }

        // a packed int4 weight is stored as <name>.weight_packed with a _scale and a _shape tensor
        const bool packed = st_ends_with(name, ".weight_packed");
        std::string base_name = packed ? name.substr(0, name.size() - strlen("_packed")) : name;

        // exl3 stores a quantized linear as coded indices plus one factor per input and per output
        // channel: the weight is decoded from those tensors and has no storage of its own
        st_ref  ex_suh, ex_svh;
        int64_t ex_bits = 0, ex_din = 0, ex_dout = 0;
        if (st_ends_with(name, ".trellis")) {
            const std::string pfx = name.substr(0, name.size() - strlen(".trellis"));
            const auto it_su = tensors.find(pfx + ".suh");
            const auto it_sv = tensors.find(pfx + ".svh");
            if (it_su == tensors.end() || it_sv == tensors.end() || ref.ndim != 3 || ref.ne[2] % 16 != 0) {
                throw std::runtime_error("unexpected exl3 trellis tensor '" + name + "'");
            }
            ex_suh  = it_su->second;
            ex_svh  = it_sv->second;
            ex_bits = ref.ne[2] / 16;
            ex_din  = (int64_t) ex_suh.nrows() * ex_suh.ncols();
            ex_dout = (int64_t) ex_svh.nrows() * ex_svh.ncols();
            if (ex_din % 16 != 0 || ex_dout % 16 != 0 ||
                (ref.ne[0] * 16 != ex_din && ref.ne[0] * 16 != ex_dout) ||
                (ref.ne[1] * 16 != ex_din && ref.ne[1] * 16 != ex_dout)) {
                throw std::runtime_error("exl3 tensor '" + name + "' does not match its suh or svh factors");
            }
            // the architecture knows the decoded weight under this name
            name      = pfx + ".weight";
            base_name = name;
        }

        // an MTP head export contains only the MTP block, the trunk layers stay in the target model
        static const char * layers_prefix = "model.language_model.layers.";
        if (mode == ST_MODE_MTP && base_name.rfind(layers_prefix, 0) == 0 &&
            atoi(base_name.c_str() + strlen(layers_prefix)) < n_layer) {
            n_skipped++;
            continue;
        }

        st_map m;
        int64_t expert = -1;
        std::string expert_base;
        bool mapped = false;
        switch (kind) {
            case ST_MAP_QWEN:    mapped = st_map_tensor(base_name, n_layer, m, expert, expert_base); break;
            case ST_MAP_GEMMA4:  mapped = st_map_tensor_gemma4(base_name, n_layer, m, expert, expert_base); break;
            case ST_MAP_GENERIC: mapped = st_map_tensor_generic(base_name, n_layer, *gen, m, expert, expert_base); break;
            case ST_MAP_NEMOTRON: mapped = st_map_tensor_nemotron(base_name, L.ssm_groups, m, expert, expert_base); break;
        }
        if (!mapped) {
            n_skipped++;
            continue;
        }

        if (L.prism_hadamard) {
            m.add_one = 0;
            if (m.reorder == ST_RE_OUT_COLS && L.hadamard_gdn_v_grouped) {
                m.reorder = ST_RE_NONE;
            }
        }

        if (m.fused != 0) {
            // MTP export fuses the experts of the layer: one 3d tensor per projection
            if (ref.ndim != 3 || ref.ne[0] != (int64_t) n_expert ||
                (m.fused == 1 && ref.ne[1] % 2 != 0)) {
                throw std::runtime_error("unexpected fused expert tensor '" + src_name + "'");
            }
            const uint64_t elem = (uint64_t) st_dtype_size(ref.dtype);
            auto add_experts = [&](const std::string & out, int64_t row_offs, int64_t nrows) {
                auto & weights = exps[out];
                if (weights.empty()) {
                    weights.resize(n_expert);
                }
                for (int64_t e = 0; e < (int64_t) n_expert; ++e) {
                    st_ref s = ref;
                    s.offs = ref.offs + (uint64_t) e * ref.ne[1] * ref.ne[2] * elem + (uint64_t) row_offs * ref.ne[2] * elem;
                    s.ndim = 2;
                    s.ne[0] = nrows;
                    s.ne[1] = ref.ne[2];
                    s.ne[2] = 1;
                    weights[e] = s;
                }
            };
            if (m.fused == 1) {
                // gate and up share one 3d tensor, split in half along the rows
                const int64_t half = ref.ne[1] / 2;
                add_experts(expert_base, half, half);
                add_experts(std::string(expert_base.begin(), expert_base.end() - strlen("ffn_up_exps.weight")) + "ffn_gate_exps.weight", 0, half);
            } else {
                add_experts(expert_base, 0, ref.ne[1]);
            }
            continue;
        }

        // quantized weights keep their scales in sibling tensors
        const std::string stem = st_ends_with(base_name, ".weight") ?
            base_name.substr(0, base_name.size() - strlen(".weight")) : std::string();
        const auto find_sibling = [&](const char * suffix) {
            return stem.empty() ? tensors.end() : tensors.find(stem + suffix);
        };

        if (expert >= 0) {
            if (packed && ref.dtype != ST_DT_U8) {
                throw std::runtime_error("packed experts are not supported yet: " + name);
            }
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

            if (ref.dtype == ST_DT_U8) {
                // NVFP4 experts carry their E4M3 block scales and their global scales with each expert
                auto it_scale  = find_sibling(".weight_scale");
                auto it_scale2 = find_sibling(".weight_scale_2");
                auto it_iscale = find_sibling(".input_scale");
                int recip = 0;
                if (it_scale2 == tensors.end() || it_iscale == tensors.end()) {
                    // compressed-tensors calls them the global scales and stores the reciprocal of the factor
                    it_scale2 = find_sibling(".weight_global_scale");
                    it_iscale = find_sibling(".input_global_scale");
                    recip = 1;
                }
                if (it_scale == tensors.end() || it_scale2 == tensors.end() || it_iscale == tensors.end()) {
                    throw std::runtime_error("missing NVFP4 scales of expert " + name);
                }
                scales[expert] = it_scale->second;

                if (exps_scale2[expert_base].empty()) {
                    exps_scale2[expert_base].resize(n_expert);
                    exps_iscale[expert_base].resize(n_expert);
                }
                exps_scale2[expert_base][expert] = it_scale2->second;
                exps_iscale[expert_base][expert] = it_iscale->second;
                exps_nvfp4[expert_base] = 1;
                exps_nvfp4_recip[expert_base] = recip;
            } else {
                const auto it_scale = tensors.find(name + "_scale_inv");
                if (it_scale != tensors.end()) {
                    scales[expert] = it_scale->second;
                } else {
                    // compressed-tensors fp8: one scale per tensor, or one per output channel
                    const auto it_ws = find_sibling(".weight_scale");
                    if (it_ws != tensors.end()) {
                        scales[expert] = it_ws->second;
                        const st_ref & s = it_ws->second;
                        const int64_t n_scale = s.ndim >= 2 ? s.ne[0] * s.ne[1] : s.ne[0];
                        if (n_scale == 1) {
                            exps_f8_mode[expert_base] = 1;
                        } else if (s.ndim >= 2 && s.ne[1] == 1) {
                            if (s.ne[0] != ref.ne[0]) {
                                throw std::runtime_error("unexpected fp8 scale shape of expert " + name);
                            }
                            exps_f8_mode[expert_base] = 2;
                        }
                    }
                }
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

        st_ref nvfp4_scale2;
        st_ref nvfp4_iscale;
        int    nvfp4_reciprocal = 0;

        std::string pfx = name;
        if (st_ends_with(pfx, ".weight")) {
            pfx = pfx.substr(0, pfx.size() - strlen(".weight"));
        }
        const auto it_mlx_scale = tensors.find(pfx + ".scales");
        const bool is_mlx_pq2 = (ref.dtype == ST_DT_U32 && it_mlx_scale != tensors.end());

        if (packed && ref.dtype != ST_DT_U8) {
            // compressed-tensors int4: nibble packed with a separate shape tensor
            const auto it_scale = tensors.find(base_name + "_scale");
            const auto it_shape = tensors.find(base_name + "_shape");
            if (it_scale == tensors.end() || it_shape == tensors.end()) {
                throw std::runtime_error("missing scale or shape of packed tensor " + name);
            }
            int64_t rows = 0;
            int64_t cols = 0;
            st_read_shape(it_shape->second, rows, cols);

            plan->ndim       = 2;
            plan->ne[0]      = cols;
            plan->ne[1]      = rows;
            plan->has_scale  = 1;
            plan->scale      = it_scale->second;
            plan->pack_bits  = L.pack_bits;
            plan->pack_group = L.pack_group;
            plan->type       = L.int4_type;
        } else if (ref.dtype == ST_DT_U8) {
            // NVFP4: 2 values per byte, one E4M3 scale per group of 16 columns
            auto it_scale  = find_sibling(".weight_scale");
            auto it_scale2 = find_sibling(".weight_scale_2");
            auto it_iscale = find_sibling(".input_scale");
            if (it_scale2 == tensors.end() || it_iscale == tensors.end()) {
                // compressed-tensors calls them the global scales and stores the reciprocal of the factor
                it_scale2 = find_sibling(".weight_global_scale");
                it_iscale = find_sibling(".input_global_scale");
                nvfp4_reciprocal = 1;
            }
            if (it_scale == tensors.end() || it_scale2 == tensors.end() || it_iscale == tensors.end()) {
                throw std::runtime_error("missing NVFP4 scales of tensor " + name +
                        " (weight_scale with weight_scale_2/input_scale, or weight_scale with the global scales, are required)");
            }
            if (ref.ne[1] * 2 != it_scale->second.ne[1] * 16) {
                throw std::runtime_error("unexpected NVFP4 scale shape of tensor " + name);
            }

            plan->ndim      = 2;
            plan->ne[0]     = ref.ne[1] * 2;
            plan->ne[1]     = ref.ne[0];
            plan->has_scale = 1;
            plan->scale     = it_scale->second;
            plan->nvfp4     = 1;
            plan->type      = GGML_TYPE_NVFP4;
            nvfp4_scale2    = it_scale2->second;
            nvfp4_iscale    = it_iscale->second;
        } else if (is_mlx_pq2) {
            plan->ndim      = 2;
            plan->ne[0]     = ref.ne[1] * 16;
            plan->ne[1]     = ref.ne[0];
            plan->has_scale = 1;
            plan->scale     = it_mlx_scale->second;
            plan->mlx_pq2   = 1;
            plan->type      = GGML_TYPE_PQ2_0;
        } else {
            if (ref.ndim == 1) {
                plan->ne[0] = ref.ne[0];
            } else {
                plan->ne[0] = ref.ne[1];
                plan->ne[1] = ref.ne[0];
            }

            if (m.split_rows != 0) {
                // a flat source that the architecture reads as rows: the mamba2 group norm, or a column
                const int64_t split = m.split_rows < 0 ? ref.ne[0] : m.split_rows;
                if (ref.ndim != 1 || split <= 0 || ref.ne[0] % split != 0) {
                    throw std::runtime_error("unexpected shape for tensor " + name);
                }
                plan->ndim       = 2;
                plan->ne[0]      = ref.ne[0] / split;
                plan->ne[1]      = split;
                plan->split_rows = split;
            }

            if (ref.dtype == ST_DT_F8) {
                const auto it_block = tensors.find(name + "_scale_inv");
                const auto it_tensor = find_sibling(".weight_scale");
                if (it_block != tensors.end()) {
                    // compressed-tensors fp8: one scale per 128x128 block of the weight
                    plan->has_scale      = 1;
                    plan->scale          = it_block->second;
                    plan->scale_per_block = 1;
                } else if (it_tensor != tensors.end()) {
                    const st_ref & s = it_tensor->second;
                    const int64_t n_scale = s.ndim >= 2 ? s.ne[0] * s.ne[1] : s.ne[0];
                    plan->has_scale = 1;
                    plan->scale     = s;
                    if (n_scale == 1) {
                        // modelopt fp8: one scale for the whole weight
                        plan->scale_per_tensor = 1;
                    } else if (s.ndim >= 2 && s.ne[1] == 1) {
                        // compressed-tensors fp8 with the channel strategy: one scale per output channel
                        if (s.ne[0] != plan->ne[1]) {
                            throw std::runtime_error("unexpected fp8 scale shape of tensor " + name);
                        }
                        plan->scale_per_row = 1;
                    }
                } else {
                    throw std::runtime_error("missing scales for tensor " + name);
                }
            }

            if (ref.dtype == ST_DT_F8 && L.fp8_native) {
                // keep the weights of the checkpoint in fp8, no requantization
                plan->f8_native = 1;
                plan->type      = GGML_TYPE_F8_E4M3;
            } else {
                plan->type = m.role == ST_ROLE_F32 ? GGML_TYPE_F32 :
                             ref.dtype == ST_DT_F8  ? L.fp8_type : st_ggml_type(ref.dtype);
            }
        }

        // a 1d tensor has a single row, so the permutation applies to the element index
        if (plan->ndim == 1 && plan->reorder.dim == 0) {
            plan->reorder.dim = 1;
        }

        // tensors that are stored as-is are read by the loader itself, with mmap like a GGUF file
        plan->verbatim = !packed && !plan->nvfp4 && !plan->f8_native && !plan->mlx_pq2 && plan->split_rows == 0 && ref.dtype != ST_DT_F8 &&
                         st_ggml_type(ref.dtype) == plan->type &&
                         plan->reorder.dim < 0 && !plan->add_one && !plan->neg_exp;

        // a quantized tensor is emitted row by row, so the row length must fit whole blocks
        if (!plan->verbatim && ggml_is_quantized(plan->type) && plan->ne[0] % ggml_blck_size(plan->type) != 0) {
            throw std::runtime_error(string_format("row length %lld of tensor '%s' is not a multiple of the %d value block of %s",
                    (long long) plan->ne[0], plan->name.c_str(), (int) ggml_blck_size(plan->type), ggml_type_name(plan->type)));
        }

        // ggml applies the per tensor scales of an NVFP4 weight separately, like the converter stores them
        if (plan->nvfp4) {
            const std::string base = m.gguf.substr(0, m.gguf.size() - strlen(".weight"));
            const auto add_sidecar = [&](const char * suffix, const st_ref & src) {
                auto side = std::make_unique<st_plan>();
                side->name       = base + suffix;
                side->src        = src;
                side->ndim       = 1;
                side->ne[0]      = 1;
                side->type       = GGML_TYPE_F32;
                side->verbatim   = nvfp4_reciprocal ? 0 : 1;
                side->reciprocal = nvfp4_reciprocal;
                add_plan(std::move(side));
            };
            add_sidecar(".scale",       nvfp4_scale2);
            add_sidecar(".input_scale", nvfp4_iscale);
        }

        if (ex_bits > 0) {
            // the decoded weight is not stored in the checkpoint: keep the trellis and the factors
            plan->exl3      = 1;
            plan->exl3_bits = ex_bits;
            plan->exl3_suh  = ex_suh;
            plan->exl3_svh  = ex_svh;
            plan->ndim      = 2;
            plan->ne[0]     = ex_din;  // input channels, the ggml row length
            plan->ne[1]     = ex_dout; // output channels
            // follow the size of the checkpoint unless the outtype asked for something else:
            // the nearest native type to the bit width of this tensor's trellis
            plan->type      = L.exl3_type == GGML_TYPE_COUNT ?
                (ex_bits <= 2 ? GGML_TYPE_Q2_K : ex_bits == 3 ? GGML_TYPE_Q3_K : GGML_TYPE_Q4_K) : L.exl3_type;
        }
        add_plan(std::move(plan));
    }

    if (kind == ST_MAP_GEMMA4) {
        // the full attention layers rotate a part of the head, the frequency factors disable what stays unrotated
        const json rope = st_json_value(tc, "rope_parameters", json::object());
        const json rope_full = rope.contains("full_attention") ? rope.at("full_attention") : json::object();
        const int64_t head_dim_full = st_json_value(tc, "global_head_dim", st_json_value(tc, "head_dim", 0));
        if (head_dim_full <= 0) {
            throw std::runtime_error("cannot read the head dimensions of this checkpoint");
        }

        auto plan = std::make_unique<st_plan>();
        plan->name           = "rope_freqs.weight";
        plan->ndim           = 1;
        plan->ne[0]          = head_dim_full / 2;
        plan->type           = GGML_TYPE_F32;
        plan->rope_freqs     = 1;
        plan->rope_freqs_rot = (int64_t) (head_dim_full * st_json_value(rope_full, "partial_rotary_factor", 1.0) / 2);
        add_plan(std::move(plan));
    }

    for (auto & [base, weights] : exps) {
        auto plan = std::make_unique<st_plan>();
        plan->name     = base;
        plan->ndim     = 3;
        plan->n_expert = n_expert;
        plan->exps     = std::move(weights);
        plan->exps_scale = std::move(exps_scale[base]);

        if (exps_nvfp4[base]) {
            // every expert is NVFP4 packed, the stacked tensor keeps the packed blocks
            auto & scale2 = exps_scale2[base];
            auto & iscale = exps_iscale[base];
            if (scale2.size() != (size_t) n_expert || iscale.size() != (size_t) n_expert) {
                throw std::runtime_error("incomplete NVFP4 experts of tensor " + base);
            }
            plan->exps_nvfp4  = 1;
            plan->exps_scale2 = scale2;
            plan->exps_iscale = iscale;
            plan->type        = GGML_TYPE_NVFP4;
            plan->ne[0]       = plan->exps_scale[0].ne[1] * 16;
            plan->ne[1]       = plan->exps_scale[0].ne[0];
            plan->ne[2]       = n_expert;

            // ggml applies the per expert scales next to the experts, like the converter stores them
            const std::string prefix = base.substr(0, base.size() - strlen(".weight"));
            const std::pair<const char *, const std::vector<st_ref> *> nvfp4_sides[2] = {
                {".scale",       &scale2},
                {".input_scale", &iscale},
            };
            for (const auto & side : nvfp4_sides) {
                auto sc = std::make_unique<st_plan>();
                sc->name     = prefix + side.first;
                sc->ndim     = 1;
                sc->ne[0]    = n_expert;
                sc->type     = GGML_TYPE_F32;
                sc->stack_src = *side.second;
                sc->stack_reciprocal = exps_nvfp4_recip[base];
                add_plan(std::move(sc));
            }
        } else {
            plan->ne[0]    = plan->exps[0].ne[1];
            plan->ne[1]    = plan->exps[0].ne[0];
            plan->ne[2]    = n_expert;
            plan->type     = plan->exps[0].dtype == ST_DT_F8 ? L.fp8_type : st_ggml_type(plan->exps[0].dtype);
            const int f8_mode = exps_f8_mode[base];
            if (f8_mode == 1) {
                plan->scale_per_tensor = 1;
            } else if (f8_mode == 2) {
                plan->scale_per_row = 1;
            }
        }

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

        const std::string model_type = st_json_value(cfg, "model_type", std::string());
        const bool prism_hadamard = (model_type == "prism_hadamard_qwen35");
        const bool moe = model_type == "qwen3_5_moe";
        const bool native_qwen = model_type == "qwen3_5" || moe || model_type == "agnes" || prism_hadamard;
        const bool gemma4 = !native_qwen && (model_type == "gemma4_unified" || model_type == "gemma4");
        // the nemotron-h family nests its text configuration in llm_config, its values win over the top level
        const bool nemotron = !native_qwen && !gemma4 && model_type.rfind("NemotronH", 0) == 0;
        json cfg_text = cfg;
        if (nemotron && cfg.contains("llm_config") && cfg.at("llm_config").is_object()) {
            for (const auto & [key, value] : cfg.at("llm_config").items()) {
                cfg_text[key] = value;
            }
        }
        const st_generic_arch * generic = (native_qwen || gemma4 || nemotron) ? nullptr : st_generic_lookup(cfg);
        if (!native_qwen && !gemma4 && !nemotron && generic == nullptr) {
            throw std::runtime_error("unsupported safetensors architecture: " + model_type +
                    " (no mapping for it, and no generic decoder mapping either - convert this checkpoint to GGUF first)");
        }

        auto L = std::make_unique<st_loader>();
        if (fp8_type == "auto") {
            L->fp8_type  = GGML_TYPE_Q8_0;
            L->int4_type = GGML_TYPE_Q4_0;
        } else if (fp8_type == "native") {
            L->fp8_native = 1;
            L->int4_type  = GGML_TYPE_Q4_0;
        } else if (fp8_type == "q8_0") {
            L->fp8_type = L->int4_type = L->exl3_type = GGML_TYPE_Q8_0;
        } else if (fp8_type == "q4_0") {
            L->fp8_type = L->int4_type = L->exl3_type = GGML_TYPE_Q4_0;
        } else if (fp8_type == "f16") {
            L->fp8_type = L->int4_type = L->exl3_type = GGML_TYPE_F16;
        } else if (fp8_type == "bf16") {
            L->fp8_type = L->int4_type = L->exl3_type = GGML_TYPE_BF16;
        } else {
            throw std::runtime_error("unsupported --safetensors-outtype: " + fp8_type);
        }

        gguf_context * meta = gguf_init_empty();

        if (model_type == "agnes") {
            st_add_meta_agnes(meta, cfg, dir.filename().string());
            st_add_meta_vocab(meta, dir, cfg, ST_VOCAB_BPE);
            st_build_plans_agnes(*L, meta, dir, cfg, mode);
        } else if (gemma4) {
            st_parse_quant_config(*L, cfg);
            st_add_meta_gemma4(meta, cfg, dir.filename().string());
            st_add_meta_vocab(meta, dir, cfg, ST_VOCAB_GEMMA4);
            st_build_plans(*L, meta, dir, cfg, mode, ST_MAP_GEMMA4, nullptr);
        } else if (nemotron) {
            st_parse_quant_config(*L, cfg_text);
            st_add_meta_nemotron(*L, meta, cfg_text, dir.filename().string());
            st_add_meta_vocab(meta, dir, cfg_text, ST_VOCAB_NEMOTRON);
            st_build_plans(*L, meta, dir, cfg_text, mode, ST_MAP_NEMOTRON, nullptr);
        } else if (generic != nullptr) {
            st_parse_quant_config(*L, cfg);
            st_add_meta_generic(*L, meta, cfg, dir.filename().string(), *generic);
            st_add_meta_vocab(meta, dir, cfg, ST_VOCAB_BPE);
            st_build_plans(*L, meta, dir, cfg, mode, ST_MAP_GENERIC, generic);
        } else {
            st_parse_quant_config(*L, cfg);
            st_add_meta_arch(*L, meta, cfg, dir.filename().string(), moe);
            st_add_meta_vocab(meta, dir, cfg, ST_VOCAB_BPE);
            if (prism_hadamard) {
                L->prism_hadamard = true;
                const json & tc = cfg.at("text_config");
                const int64_t n_layer = tc.at("num_hidden_layers").get<int64_t>();
                st_add_meta_hadamard(*L, meta, dir, n_layer);
            }
            st_build_plans(*L, meta, dir, cfg, mode, ST_MAP_QWEN, nullptr);
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
    const uint32_t n_pos     = st_json_value(v, "num_position_embeddings", 0u);
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
    gguf_set_val_u32 (meta, "clip.vision.spatial_merge_size", st_json_value(v, "spatial_merge_size", 2u));
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
    for (const auto & idx : st_json_value(v, "deepstack_visual_indexes", std::vector<int>())) {
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

    const std::vector<std::string> part_names = st_list_shards(dir);

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

static void st_add_meta_agnes(gguf_context * meta, const json & cfg, const std::string & model_name) {
    const json & t = cfg.at("text_config");

    const uint32_t n_layer  = t.at("num_hidden_layers").get<uint32_t>();
    const uint32_t n_nextn  = st_json_value(t, "mtp_num_hidden_layers", 0u);
    const uint32_t n_embd   = t.at("hidden_size").get<uint32_t>();
    const uint32_t n_head   = t.at("num_attention_heads").get<uint32_t>();
    const uint32_t head_dim = t.at("head_dim").get<uint32_t>();
    const uint32_t n_ff     = t.at("intermediate_size").get<uint32_t>();
    const uint32_t n_ff_par = st_json_value(t, "parallel_ffn_intermediate_size", 0u);

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
    ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       (uint32_t) (head_dim * st_json_value(t, "partial_rotary_factor", 0.25f)));
    ms.add_kv(LLM_KV_ROPE_FREQ_BASE,             rope_theta);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, t.at("rms_norm_eps").get<float>());
    ms.add_kv(LLM_KV_NEXTN_PREDICT_LAYERS,       n_nextn);
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL,    st_json_value(t, "global_attention_interval", 4u));
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
    L.num_k_heads = t.at("linear_num_key_heads").get<uint32_t>();
    L.num_v_heads = t.at("linear_num_value_heads").get<uint32_t>();
    L.head_k_dim  = t.at("linear_key_head_dim").get<uint32_t>();
    L.head_v_dim  = t.at("linear_value_head_dim").get<uint32_t>();

    const std::vector<std::string> part_names = st_list_shards(dir);

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
