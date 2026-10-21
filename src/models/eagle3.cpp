#include "models.h"

void llama_model_eagle3::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    if (!ml.get_arr(LLM_KV_TARGET_LAYERS, target_layer_ids, false)) {
        throw std::runtime_error(arch_name() + " model requires 'target_layers' in GGUF metadata");
    }
    if (target_layer_ids.size() != 3) {
        throw std::runtime_error(arch_name() + " requires exactly 3 entries in 'target_layers'");
    }
    LLAMA_LOG_INFO("%s: %s target_layers = [%d, %d, %d]\n", __func__, arch_name().c_str(),
            target_layer_ids[0],
            target_layer_ids[1],
            target_layer_ids[2]);

    uint32_t n_embd_tgt = 0;

    ml.get_key(LLM_KV_TARGET_HIDDEN_SIZE, n_embd_tgt);
    LLAMA_LOG_INFO("%s: %s n_embd_tgt = %u (draft n_embd = %u)\n", __func__, arch_name().c_str(), n_embd_tgt, hparams.n_embd);

    hparams.n_embd_inp_enc_impl = (uint32_t) target_layer_ids.size() * n_embd_tgt;

    // eagle3 norm_before_residual (optional, default false)
    // compatible with Readhat eagle3 speculator model
    ml.get_key(LLM_KV_NORM_BEFORE_RESIDUAL, hparams.norm_before_residual, false);
    if (hparams.norm_before_residual) {
        LLAMA_LOG_INFO("%s: %s norm_before_residual = true\n", __func__, arch_name().c_str());
    }

    // eagle3 norm_before_fc (optional, default false)
    // compatible with eagle3.1 (e.g. nvidia/gpt-oss-120b-Eagle3-v3)
    ml.get_key(LLM_KV_NORM_BEFORE_FC, hparams.norm_before_fc, false);

    // eagle4: per-aux-hidden-state norm before fc and normed output for the next step
    ml.get_key(LLM_KV_FC_NORM,     hparams.fc_norm,     false);
    ml.get_key(LLM_KV_NORM_OUTPUT, hparams.norm_output, false);

    // eagle4: the draft layer may use a sliding window. The model is one layer, so
    // the pattern is a single entry.
    if (ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false) && hparams.n_swa > 0) {
        if (hparams.n_layer_all != 1) {
            throw std::runtime_error(arch_name() + " sliding-window drafts must have 1 layer");
        }
        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
        if (!ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl)) {
            throw std::runtime_error(arch_name() + " sliding-window drafts require 'sliding_window_pattern'");
        }
        hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
        hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;
        LLAMA_LOG_INFO("%s: %s sliding_window = %u\n", __func__, arch_name().c_str(), hparams.n_swa);
    }

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_eagle3::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_embd_inp = hparams.n_embd_inp_enc();
    const int64_t n_embd_attn_input = 2 * n_embd;

    // Get vocab size from the d2t tensor in the GGUF file (optional - only needed if eagle3 has different vocab_size than target)
    // d2t: draft to target vocabulary mapping
    int64_t n_draft_vocab = n_vocab;  // Default: same as target vocab
    const struct ggml_tensor * d2t_meta = ml->get_tensor_meta("d2t");
    if (d2t_meta) {
        n_draft_vocab = d2t_meta->ne[0]; // update draft vocab size
        d2t = create_tensor(tn(LLM_TENSOR_D2T), {n_draft_vocab}, 0);
        LLAMA_LOG_INFO("%s: %s using d2t mapping (draft_vocab_size = %lld)\n", __func__, arch_name().c_str(), (long long)n_draft_vocab);
    } else {
        d2t = nullptr; // no d2t, use default vocab size
        LLAMA_LOG_INFO("%s: %s without d2t - sharing same vocab_size with target (vocab_size = %lld)\n", __func__, arch_name().c_str(), (long long)n_draft_vocab);
    }

    // Feature fusion layer: projects 3 target layers to draft hidden size
    fc = create_tensor(tn(LLM_TENSOR_FC, "weight"), {n_embd_inp, n_embd}, 0);

    // eagle4: one RMSNorm per aux hidden state, applied before the fc projection
    if (hparams.fc_norm) {
        const int64_t n_embd_tgt = n_embd_inp / 3;
        for (int i = 0; i < 3; ++i) {
            fc_norm[i] = create_tensor(tn(LLM_TENSOR_FC_NORM, "weight", i), {n_embd_tgt}, 0);
        }
    }

    // eagle4: low-rank candidate head. The rank comes from the factor shape, so no
    // extra metadata is needed. The full output head stays loaded for re-scoring.
    {
        const std::string name = tn(LLM_TENSOR_EFFICIENT_LM_HEAD, "weight").str();
        const ggml_tensor * meta = ml->get_tensor_meta(name.c_str());
        if (meta) {
            if (d2t) {
                throw std::runtime_error(arch_name() + " cannot combine a reduced draft vocabulary with the candidate head");
            }
            const int64_t rank = meta->ne[0];
            efficient_lm_head      = create_tensor(tn(LLM_TENSOR_EFFICIENT_LM_HEAD,      "weight"), {rank, n_vocab}, 0);
            efficient_lm_head_down = create_tensor(tn(LLM_TENSOR_EFFICIENT_LM_HEAD_DOWN, "weight"), {n_embd, rank}, 0);
            LLAMA_LOG_INFO("%s: %s low-rank candidate head rank = %lld\n", __func__, arch_name().c_str(), (long long) rank);
        }
    }

    // RMSNorm on the fused target features (input to fc), only when norm_before_fc is set.
    if (hparams.norm_before_fc) {
        output_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_OUTPUT_NORM, "weight"), {n_embd_inp}, 0);
    }

    // Output layer (uses draft vocab size)
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_draft_vocab}, TENSOR_NOT_REQUIRED);

    // Token embeddings (optional - Llama 3.3 70B EAGLE3 has its own)
    const struct ggml_tensor * tok_embd_meta = ml->get_tensor_meta(tn(LLM_TENSOR_TOKEN_EMBD, "weight").str().c_str());
    if (tok_embd_meta) {
        const int64_t n_target_vocab = tok_embd_meta->ne[1];
        tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_target_vocab}, 0);
        LLAMA_LOG_INFO("%s: %s using its own token_embd (vocab = %lld)\n", __func__, arch_name().c_str(), (long long)n_target_vocab);
    }

    // Single decoder layer
    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        // input_layernorm: applied to token embeddings
        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        // eagle3 specific: hidden_norm applied to fused target features
        layer.attn_norm_2 = create_tensor(tn(LLM_TENSOR_ATTN_NORM_2, "weight", i), {n_embd}, 0);

        // Attention takes input_embeds_normed + fused_target_normed as input
        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd_attn_input, n_embd_head_k * n_head}, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd_attn_input, n_embd_k_gqa}, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd_attn_input, n_embd_v_gqa}, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);

        // rope_freqs for llama3 rope scaling (optional - only if eagle3 config has rope_scaling)
        layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED);
    }
}

template <>
ggml_tensor * llama_model_eagle3::graph<true>::build_inp_embd_enc() const {
    ggml_tensor * cur = nullptr;

    // Input: Target model features (3 layers concatenated: low, mid, high)
    // Data will be provided via ubatch->embd in encode_eagle3_features()
    auto inp_target = std::make_unique<llm_graph_input_embd>(hparams.n_embd_inp_enc());
    inp_target->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp_enc(), n_tokens);
    ggml_set_input(inp_target->embd);

    cur = inp_target->embd;
    cb(cur, "inp_embd", -1);

    res->add_input(std::move(inp_target));

    return cur;
}

// eagle3 Encoder: processes target model features through feature fusion layer
// Input: target_features e.g. [12288, n_tokens] from target model layers low, middle, high
// Output: g_embeddings e.g. [4096, n_tokens] stored in context
template <>
llama_model_eagle3::graph<true>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur = nullptr;

    cur = build_inp_embd_enc();

    if (hparams.fc_norm) {
        // eagle4: normalize each aux hidden state separately, then concatenate
        const int64_t n_embd_tgt = hparams.n_embd_inp_enc() / 3;
        ggml_tensor * normed[3] = {};
        for (int k = 0; k < 3; ++k) {
            ggml_tensor * chunk = ggml_view_2d(ctx0, cur, n_embd_tgt, n_tokens, cur->nb[1],
                    (size_t) k * n_embd_tgt * sizeof(float));
            normed[k] = build_norm(chunk, model.fc_norm[k], NULL, LLM_NORM_RMS, -1);
            cb(normed[k], "fc_norm", k);
        }
        cur = ggml_concat(ctx0, ggml_concat(ctx0, normed[0], normed[1], 0), normed[2], 0);
    }

    // RMSNorm on the fused target features before fc
    if (hparams.norm_before_fc) {
        cur = build_norm(cur, model.output_norm_enc, NULL, LLM_NORM_RMS, -1);
        cb(cur, "enc_input_norm", -1);
    }

    // Feature fusion layer
    cur = build_lora_mm(model.fc, cur);
    cb(cur, "fc_out", -1);

    // Output: g_embeddings e.g. [4096, n_tokens]
    // store in t_h_nextn (same as MTP) so can be read via llama_get_embeddings_nextn(ctx_dft)
    ggml_set_output(cur);
    res->t_h_nextn = cur;

    ggml_build_forward_expand(gf, cur);
}

// eagle3 Decoder: processes draft tokens using g_embeddings from encoder
// Input: draft tokens + g_embeddings from encoder
// Output: draft logits
template <>
llama_model_eagle3::graph<false>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_layer == 1);  // eagle3 has only one decoder layer

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // eagle3 Decoder receives:
    // 1. Token embeddings (e.g.from eagle3's own tok_embd for Llama 3.3 70B, or target model for Llama 3.1 8B)
    // 2. g_embeddings from encoder
    auto * tok_embd = model.tok_embd;
    if (model.tok_embd == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(cparams.ctx_other);

        GGML_ASSERT(model_other->tok_embd != nullptr && "EAGLE decoder requires token embeddings (own or from target model)");
        tok_embd = model_other->tok_embd;
    }

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_input(inp->embd);

    ggml_tensor * inp_embd = ggml_get_rows(ctx0, tok_embd, inp->tokens);
    cb(inp_embd, "inp_embd", -1);

    ggml_tensor * inp_g = inp->embd;
    cb(inp_g, "inp_g_embeddings", -1);

    res->add_input(std::move(inp));

    inpL = inp_g;

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn       = (llm_graph_input_attn_kv *)      nullptr;
    auto * inp_attn_iswa = (llm_graph_input_attn_kv_iswa *) nullptr;
    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        inp_attn_iswa = build_attn_inp_kv_iswa();
    } else {
        inp_attn = build_attn_inp_kv();
    }

    const float kq_scale = 1.0f/sqrtf(float(n_embd_head));

    // Single decoder layer (il = 0)
    const int il = 0;
    {
        // Apply input_layernorm to the token embeddings
        ggml_tensor * embd_norm = build_norm(inp_embd,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(embd_norm, "embd_norm", il);

        // Apply hidden_norm to inp_g
        ggml_tensor * g_norm = build_norm(inp_g,
                model.layers[il].attn_norm_2, NULL,
                LLM_NORM_RMS, -1);
        cb(g_norm, "g_norm", il);

        // norm_before_residual: determines what goes into the residual connection (compatible with Readhat eagle3 speculator model)
        // - false (default): use raw inp_g for residual
        // - true: use normalized g_norm for residual
        // inpL is the concatenated input (normalized inp_embd + normalized inp_g)
        ggml_tensor * inpSA = hparams.norm_before_residual ? g_norm : inpL;

        // Concatenate normalized inp_embd and normalized inp_g
        cur = ggml_concat(ctx0, embd_norm, g_norm, il);
        cb(cur, "concat_embd", il);

        // Self-attention with concatenated input
        ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
        cb(Qcur, "Qcur", il);

        ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
        cb(Kcur, "Kcur", il);

        ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
        cb(Vcur, "Vcur", il);

        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

        // rope freq factors, returns nullptr if not available
        ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

        // RoPE
        Qcur = ggml_rope_ext(
                ctx0, Qcur, inp_pos, rope_factors,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
                );
        Kcur = ggml_rope_ext(
                ctx0, Kcur, inp_pos, rope_factors,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
                );

        cb(Qcur, "Qcur_rope", il);
        cb(Kcur, "Kcur_rope", il);

        if (inp_attn_iswa) {
            cur = build_attn(inp_attn_iswa,
                    model.layers[il].wo, NULL, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
        } else {
            cur = build_attn(inp_attn,
                    model.layers[il].wo, NULL, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
        }

        // Add residual and update it
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // Apply FFN norm to the sum
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "post_attn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        // Output norm with residual
        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "eagle3_prenorm", il);

        inpL = cur;
    }

    // only the output rows go to the norm and the lm_head
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);

    // EAGLE-4 passes the normed output to the next draft step, EAGLE-3 the pre-norm state
    ggml_tensor * prenorm = inpL;
    cur = build_norm(prenorm,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    ggml_tensor * h_nextn = hparams.norm_output ? cur : prenorm;
    ggml_set_output(h_nextn);
    res->t_h_nextn = h_nextn;

    // lm_head: the candidate head scores the vocabulary through its rank-r factors.
    // Without the reference candidate-topk kernels the two small matmuls are slower
    // than the full head on current backends, so this is opt-in.
    static const bool use_candidate_head = getenv("LLAMA_EAGLE4_CANDIDATE_HEAD") != nullptr;
    if (use_candidate_head && model.efficient_lm_head && model.efficient_lm_head_down) {
        ggml_tensor * low = build_lora_mm(model.efficient_lm_head_down, cur);
        cb(low, "candidate_low_rank", -1);
        cur = build_lora_mm(model.efficient_lm_head, low);
    } else {
        // if the draft has no own output projection, inherit the target model's lm_head
        auto * output = model.output;
        if (output == nullptr) {
            GGML_ASSERT(cparams.ctx_other != nullptr);
            const auto * model_other = llama_get_model(cparams.ctx_other);

            GGML_ASSERT(model_other->output != nullptr && "EAGLE decoder requires an output projection (own or from target model)");
            output = model_other->output;
        }
        cur = build_lora_mm(output, cur);
    }

    if (model.d2t) {
        const int64_t n_draft_vocab = cur->ne[0];
        const int64_t n_outputs     = cur->ne[1];
        const int64_t n_vocab       = (int64_t) model.vocab.n_tokens();

        GGML_ASSERT(model.d2t->type == GGML_TYPE_I64);
        GGML_ASSERT(model.d2t->ne[0] == n_draft_vocab);

        ggml_tensor * logits = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_vocab, n_outputs), -INFINITY);
        cur = ggml_set_rows(ctx0, logits,
                ggml_reshape_3d(ctx0, cur,       1,             n_draft_vocab, n_outputs),
                ggml_reshape_3d(ctx0, model.d2t, n_draft_vocab, 1,             1));
        cur = ggml_reshape_2d(ctx0, cur, n_vocab, n_outputs);
    }

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

std::unique_ptr<llm_graph_context> llama_model_eagle3::build_arch_graph(const llm_graph_params & params) const {
    switch (params.gtype) {
        case LLM_GRAPH_TYPE_ENCODER:
            return std::make_unique<graph<true>>(*this, params);
        case LLM_GRAPH_TYPE_DEFAULT:
        case LLM_GRAPH_TYPE_DECODER:
            return std::make_unique<graph<false>>(*this, params);
        default:
            GGML_ABORT("invalid graph type");
    };
}
