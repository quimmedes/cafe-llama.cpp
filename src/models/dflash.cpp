#include "models.h"

#include <cmath>

#include "llama-impl.h"
#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"

void llama_model_dflash::load_arch_hparams(llama_model_loader & ml) {

    ml.get_key(LLM_KV_EMBEDDING_SCALE, hparams.f_embedding_scale, false);
    ml.get_key(LLM_KV_ATTENTION_SCALE, hparams.f_attention_scale, false);
    ml.get_key(LLM_KV_ATTENTION_VALUE_SCALE, hparams.f_attn_value_scale, false);

    hparams.llm_ffn_op = LLM_FFN_SILU;
    std::string hidden_act;
    if (ml.get_key(LLM_KV_HIDDEN_ACT, hidden_act, false)) {
        if (hidden_act == "gelu" || hidden_act == "gelu_pytorch_tanh") {
            hparams.llm_ffn_op = LLM_FFN_GELU;
        } else if (hidden_act == "swigluoai") {
            hparams.llm_ffn_op = LLM_FFN_SWIGLU_OAI_MOE;
        } else if (hidden_act != "silu") {
            throw std::runtime_error("unsupported DFlash hidden activation: " + hidden_act);
        }
    }

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_LOGIT_SCALE,                 hparams.f_logit_scale, false);
    hparams.f_final_logit_softcapping = 0.0f;
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING,     hparams.f_final_logit_softcapping, false);

    // drafts for M-RoPE targets carry degenerate sections [n_rot/2, 0, 0, 0]
    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, 4, false);

    ml.get_key(LLM_KV_DFLASH_BLOCK_SIZE,       hparams.dflash_block_size,       false);
    ml.get_key(LLM_KV_DFLASH_CONV_KERNEL_SIZE, hparams.dflash_conv_kernel_size, false);
    ml.get_key(LLM_KV_DFLASH_CONV_GROUP_SIZE,  hparams.dflash_conv_group_size,  false);
    ml.get_key(LLM_KV_DFLASH_SELECTOR_RANK,    hparams.dflash_selector_rank,    false);
    ml.get_key(LLM_KV_DFLASH_SELECTOR_TOP_K,   hparams.dflash_selector_top_k,   false);

    const bool has_markov   = ml.get_tensor_meta("markov_w1.weight")       != nullptr;
    const bool has_selector = ml.get_tensor_meta("selector_hidden.weight") != nullptr;
    const bool has_lilicorr = ml.get_key(LLM_KV_LILICORR_HIDDEN_SIZE, hparams.lilicorr_hidden_size, false);
    if (has_markov) {
        variant = VARIANT_DSPARK;
    } else if (has_selector) {
        variant = VARIANT_DFLASH2;
    } else if (has_lilicorr) {
        variant = VARIANT_LILICORR;
        load_hparams_lilicorr(ml);
    }

    if (!ml.get_arr(LLM_KV_TARGET_LAYERS, target_layer_ids, false)) {
        throw std::runtime_error("DFlash model requires 'target_layers' in GGUF metadata");
    }

    hparams.n_embd_inp_enc_impl = (uint32_t) target_layer_ids.size() * hparams.n_embd;

    // AngelSpec DFly fuses the target context once per DRAFT layer, so the encoder emits
    // [n_embd, n_layer] per token instead of a single [n_embd] row. Detected from the fusion
    // tensor rather than a KV so a DFly export cannot load as plain DFlash.
    if (ml.get_tensor_meta("layer_fusion")) {
        // both ends must widen: a dflash draft context is not MTP-typed, so it sizes its embd batch
        // from n_embd_inp, and setting only n_embd_out gives the graph one layer's context plus junk
        hparams.n_embd_out_impl = hparams.n_layer() * hparams.n_embd;
        hparams.n_embd_inp_impl = hparams.n_layer() * hparams.n_embd;
        LLAMA_LOG_INFO("%s: DFly per-layer context fusion (n_layer = %u, n_embd_out = %u)\n",
                __func__, hparams.n_layer(), hparams.n_embd_out());
    }

    // dspark GIDD log-SNR conditioning (drafters trained with the GIDD bundle);
    // absent on every other drafter, so it must default off
    ml.get_key(LLM_KV_LOG_SNR_CONDITIONING, hparams.dspark_log_snr_conditioning, false);
    if (hparams.dspark_log_snr_conditioning) {
        // required once the flag is set: defaulting either bound to 0 collapses
        // (max - min) in the featurizer and fills the input with NaNs instead of
        // failing to load
        ml.get_key(LLM_KV_MIN_LOG_SNR, hparams.dspark_min_log_snr, true);
        ml.get_key(LLM_KV_MAX_LOG_SNR, hparams.dspark_max_log_snr, true);
        if (!std::isfinite(hparams.dspark_min_log_snr) || !std::isfinite(hparams.dspark_max_log_snr)) {
            throw std::runtime_error("dspark log-SNR conditioning: min/max_log_snr must be finite");
        }
        if (!(hparams.dspark_max_log_snr > hparams.dspark_min_log_snr)) {
            throw std::runtime_error("dspark log-SNR conditioning: max_log_snr must be greater than min_log_snr");
        }
    }

    std::string layers;
    const char * sep = "";
    for (const auto id : target_layer_ids) {
        layers += sep;
        layers += std::to_string(id);
        sep = ", ";
    }
    LLAMA_LOG_INFO("%s: DFlash extract_layers = [%s]\n", __func__, layers.c_str());

    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT, hparams.dsv4_hc_mult, false);
    if (hparams.dsv4_hc_mult > 0) {
        load_hparams_dsv4(ml);
        backbone = hparams.dsv4_hc_single_pass ? BACKBONE_DSV41 : BACKBONE_DSV4;
    } else if (ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK, hparams.n_lora_kv, false)) {
        load_hparams_mistral_large4(ml);
        backbone = BACKBONE_MISTRAL_LARGE4;
    } else if (ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false) && hparams.n_swa > 0) {
        // optional interleaved sliding-window attention with per-layer pattern array.
        // DFlash has a single rope, so the SWA rope == main rope.
        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
        ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl);
        hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
        hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;
    }

    ml.get_key_or_arr(LLM_KV_ATTENTION_ROPE_PATTERN, hparams.rope_pattern, hparams.n_layer(), false);

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_dflash::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_embd_inp = hparams.n_embd_inp_enc();

    tok_embd        = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,       "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);

    // reduced draft vocab (optional): d2t maps draft rows to target token ids
    int64_t n_vocab_draft = n_vocab;
    const struct ggml_tensor * d2t_meta = ml->get_tensor_meta("d2t");
    if (d2t_meta) {
        n_vocab_draft = d2t_meta->ne[0];
        d2t = create_tensor(tn(LLM_TENSOR_D2T), { n_vocab_draft }, 0);
        LLAMA_LOG_INFO("%s: DFlash using d2t mapping (draft_vocab_size = %lld)\n", __func__, (long long) n_vocab_draft);
    }

    // AngelSpec DFly: per-draft-layer context fusion + TreeFlash predecessor correction.
    // Detected from the fusion tensor, and checked before the Markov head below so a
    // checkpoint carrying both is reported as such rather than as a missing tensor.
    const bool is_dfly = ml->get_tensor_meta("layer_fusion") != nullptr;

    if (is_dfly && d2t) {
        throw std::runtime_error("dflash: DFly with a reduced draft vocabulary (d2t) is not supported. "
                                 "The reference chain runs the full target head");
    }

    // A predecessor correction without the per-layer fusion is a third lineage (DSpark plus
    // correction, tap_fusion none). It is not supported here, and the correction weights are
    // only created on the DFly path, so letting it through means an opaque
    // "wrong number of tensors" from done_getting_tensors rather than a reason.
    if (!is_dfly && ml->get_tensor_meta("hidden_correction.down.weight")) {
        throw std::runtime_error("dflash: checkpoint carries hidden_correction weights but no layer_fusion. "
                                 "That lineage (DSpark plus predecessor correction) is not supported; "
                                 "loading it as plain DSpark would drop the trained correction");
    }

    // Reject a declared confidence head when its required Markov head is missing.
    bool kv_confidence_head = false;
    const bool has_kv_confidence_head = ml->get_key(LLM_KV_CONFIDENCE_HEAD, kv_confidence_head, false);

    if (has_kv_confidence_head && kv_confidence_head && variant != VARIANT_DSPARK) {
        throw std::runtime_error("dflash: metadata declares a confidence head, but markov_w1.weight is missing. "
                                 "The export is incomplete; it would load as plain DFlash and read drafts one row late");
    }

    if (variant == VARIANT_DSPARK && is_dfly) {
        throw std::runtime_error("dflash: checkpoint has both a DFly layer_fusion and a DSpark markov_w1. "
                                 "The two draft chains are mutually exclusive; the export is wrong");
    }

    switch (variant) {
        case VARIANT_DSPARK:   load_tensors_dspark(n_vocab_draft); break;
        case VARIANT_DFLASH2:  load_tensors_dflash2();             break;
        case VARIANT_LILICORR: load_tensors_lilicorr();            break;
        case VARIANT_DFLASH:                                       break;
    }

    // dspark GIDD log-SNR conditioning (LogSnrEmbed): unlike the markov head, this
    // is built into the decoder graph and changes the draft embedding every forward
    // pass, so when the GGUF says it is on the weights are REQUIRED. A missing
    // tensor here is a broken conversion, not something to degrade past silently.
    if (hparams.dspark_log_snr_conditioning) {
        const int64_t n_freq = 128; // matches LogSnrEmbed.NUM_FREQ_FEATURES
        dspark_log_snr_fc1_w = create_tensor(tn(LLM_TENSOR_DSPARK_LOG_SNR_FC1, "weight"), { n_freq, n_embd }, 0);
        dspark_log_snr_fc1_b = create_tensor(tn(LLM_TENSOR_DSPARK_LOG_SNR_FC1, "bias"),   { n_embd },         0);
        dspark_log_snr_fc2_w = create_tensor(tn(LLM_TENSOR_DSPARK_LOG_SNR_FC2, "weight"), { n_embd, n_embd }, 0);
        dspark_log_snr_fc2_b = create_tensor(tn(LLM_TENSOR_DSPARK_LOG_SNR_FC2, "bias"),   { n_embd },         0);
        LLAMA_LOG_INFO("%s: DFlash with DSpark log-SNR conditioning (min %.3f, max %.3f)\n",
                __func__, (double) hparams.dspark_min_log_snr, (double) hparams.dspark_max_log_snr);
    }

    fc              = create_tensor(tn(LLM_TENSOR_FC,              "weight"), { n_embd_inp, n_embd }, 0);
    fc_s            = create_tensor(tn(LLM_TENSOR_FC,              "scale"),  { 1 }, TENSOR_NOT_REQUIRED);
    fc_in_s         = create_tensor(tn(LLM_TENSOR_FC,              "input_scale"), { 1 }, TENSOR_NOT_REQUIRED);

    // DFly replaces the encoder's single hidden_norm with a post-fusion context_norm, so
    // exactly one of the two is present.
    if (is_dfly) {
        const int64_t n_ctx_feat = (int64_t) target_layer_ids.size();

        dfly_layer_fusion = create_tensor(tn(LLM_TENSOR_DFLY_LAYER_FUSION), { n_ctx_feat, n_layer }, 0);
        dfly_ctx_norm     = create_tensor(tn(LLM_TENSOR_DFLY_CTX_NORM,     "weight"), { n_embd },             0);

        // TreeFlash predecessor correction. Optional in the reference
        // (enable_hidden_correction), so absence is a valid checkpoint, but a partial
        // set is a broken export rather than something to degrade past.
        const struct ggml_tensor * hc_meta = ml->get_tensor_meta("hidden_correction.down.weight");
        if (hc_meta) {
            const int64_t n_ff_hc = hc_meta->ne[0];

            dfly_hc_hidden_norm = create_tensor(tn(LLM_TENSOR_DFLY_HC_HIDDEN_NORM, "weight"), { n_embd },              0);
            dfly_hc_embed_norm  = create_tensor(tn(LLM_TENSOR_DFLY_HC_EMBED_NORM,  "weight"), { n_embd },              0);
            dfly_hc_gate        = create_tensor(tn(LLM_TENSOR_DFLY_HC_GATE,        "weight"), { 2*n_embd, n_ff_hc },   0);
            dfly_hc_up          = create_tensor(tn(LLM_TENSOR_DFLY_HC_UP,          "weight"), { 2*n_embd, n_ff_hc },   0);
            dfly_hc_down        = create_tensor(tn(LLM_TENSOR_DFLY_HC_DOWN,        "weight"), { n_ff_hc,  n_embd },    0);

            LLAMA_LOG_INFO("%s: DFly predecessor correction (n_ff = %lld)\n", __func__, (long long) n_ff_hc);
        } else {
            LLAMA_LOG_INFO("%s: DFly without predecessor correction\n", __func__);
        }
    } else {
        output_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_OUTPUT_NORM, "weight"), { n_embd }, 0); // encoder hidden_norm (after fc)
    }
    output_norm     = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,    "weight"), { n_embd }, 0); // decoder final norm

    // optional: reduced-vocab drafts ship their own lm head, full-vocab drafts can share the target's via ctx_other
    // a draft with its own embeddings + head references no target tensors and can run on devices the target does not use (e.g. -devd with a tensor-split target)
    output   = create_tensor(tn(LLM_TENSOR_OUTPUT,     "weight"), { n_embd, n_vocab_draft }, TENSOR_NOT_REQUIRED);

    switch (backbone) {
        case BACKBONE_BASE:           load_tensors_layers_base();           break;
        case BACKBONE_DSV4:
        case BACKBONE_DSV41:          load_tensors_layers_dsv4();           break;
        case BACKBONE_MISTRAL_LARGE4: load_tensors_layers_mistral_large4(); break;
    }
}

uint32_t llama_model_dflash::graph_nodes_per_tensor() const {
    if (hparams.dsv4_hc_mult > 0) {
        return 32;
    }
    if (hparams.dflash_selector_rank > 0) {
        // DFlash2's convolutions and selector are shape work rather than matmuls,
        // so they cost ~8.6 nodes per tensor against ~5.9 for a plain DFlash draft.
        // The selector lattice scales with the draft block size; 12u was calibrated
        // on block-8 drafters, so scale the budget for larger blocks (e.g. Ripple's 16).
        return 12u*std::max<uint32_t>(1u, hparams.dflash_block_size/8u);
    }
    return llama_model_base::graph_nodes_per_tensor();
}

void llama_model_dflash::load_tensors_layers_base() {
    LLAMA_LOAD_LOCALS;

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), { n_embd, n_embd_head_k * n_head }, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), { n_embd, n_embd_k_gqa }, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), { n_embd, n_embd_v_gqa }, TENSOR_NOT_REQUIRED);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, 0);

        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), { n_embd_head_k }, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), { n_embd_head_k }, 0);

        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), { n_embd }, TENSOR_NOT_REQUIRED);
        layer.ffn_post_norm  = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM,  "weight", i), { n_embd }, TENSOR_NOT_REQUIRED);
        layer.out_scale      = create_tensor(tn(LLM_TENSOR_LAYER_OUT_SCALE, "weight", i), { 1 }, TENSOR_NOT_REQUIRED);
        layer.rope_freqs     = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), { n_embd_head_k/2 }, TENSOR_NOT_REQUIRED | (i > 0 ? TENSOR_DUPLICATED : 0));

        // optional per-head attention sinks (e.g. Nemotron DSpark)
        layer.attn_sinks = create_tensor(tn(LLM_TENSOR_ATTN_SINKS, "weight", i), { n_head }, TENSOR_NOT_REQUIRED);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff }, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), { n_embd, n_ff }, 0);
    }
}

template <>
ggml_tensor * llama_model_dflash::graph<true>::build_inp_embd_enc() const {
    const int64_t n_embd_inp = hparams.n_embd_inp_enc();
    auto inp_target = std::make_unique<llm_graph_input_embd>(n_embd_inp);

    inp_target->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_inp, n_tokens);
    ggml_set_input(inp_target->embd);

    ggml_tensor * cur = inp_target->embd;
    cb(cur, "inp_embd", -1);

    res->add_input(std::move(inp_target));

    return cur;
}

// DFlash Encoder: processes target model features through feature fusion layer
template <>
llama_model_dflash::graph<true>::graph(const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * inp = build_inp_embd_enc();

    ggml_tensor * cur = build_lora_mm(model.fc, inp, model.fc_s);
    cb(cur, "fc_out", -1);

    if (model.dfly_layer_fusion) {
        // DFly: the shared projection is only the base context. Each draft layer adds its own
        // softmax-weighted mix of the raw per-target-layer features, so the encoder emits one
        // context per draft layer. Reference: Qwen3DFlyModel.project_target_hidden.
        const int64_t n_feat = model.dfly_layer_fusion->ne[0];
        const int64_t n_lyr  = model.dfly_layer_fusion->ne[1];

        GGML_ASSERT(n_feat*n_embd == (int64_t) hparams.n_embd_inp_enc());
        GGML_ASSERT(n_lyr == n_layer);

        // softmax over each draft layer's n_feat mixing logits (torch: softmax(dim=-1) on [L, T])
        ggml_tensor * probs = ggml_soft_max(ctx0, model.dfly_layer_fusion); // [n_feat, n_layer]

        // [n_feat*n_embd, n_tokens] -> [n_feat, n_embd, n_tokens]: put the contracted axis first
        ggml_tensor * feats = ggml_cont(ctx0, ggml_permute(ctx0,
                    ggml_reshape_3d(ctx0, inp, n_embd, n_feat, n_tokens), 1, 0, 2, 3));

        ggml_tensor * resid = ggml_mul_mat(ctx0, probs,
                    ggml_reshape_2d(ctx0, feats, n_feat, n_embd*n_tokens)); // [n_layer, n_embd*n_tokens]

        resid = ggml_cont(ctx0, ggml_permute(ctx0,
                    ggml_reshape_3d(ctx0, resid, n_lyr, n_embd, n_tokens), 1, 0, 2, 3)); // [n_embd, n_layer, n_tokens]

        // broadcast the base context across draft layers
        cur = ggml_add(ctx0, resid, ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens));
        cur = build_norm(cur, model.dfly_ctx_norm, NULL, LLM_NORM_RMS, -1);

        // flatten layer-major within each token: the host round-trips this as one
        // n_embd_out()-wide row per token and the decoder views layer il back out of it
        cur = ggml_reshape_2d(ctx0, cur, n_embd*n_lyr, n_tokens);
        cb(cur, "dfly_ctx_out", -1);
    } else {
        cur = build_norm(cur, model.output_norm_enc, NULL, LLM_NORM_RMS, -1);
        cb(cur, "enc_norm_out", -1);
    }

    ggml_set_output(cur);
    res->t_h_nextn = cur;

    ggml_build_forward_expand(gf, cur);
}

// target features of an injection batch: host embd rows, or rows of the target's layer input device buffer
static ggml_tensor * build_inp_target_features(llm_graph_context & g, llm_graph_input_embd & inp, int64_t n_embd_inp) {
    if (g.ubatch.embd_row) {
        GGML_ASSERT(g.layer_inp_src != nullptr && g.layer_inp_src->ne[0] == n_embd_inp);

        inp.embd_rows = ggml_new_tensor_1d(g.ctx0, GGML_TYPE_I32, g.n_tokens);
        ggml_set_input(inp.embd_rows);

        return ggml_get_rows(g.ctx0, g.layer_inp_src, inp.embd_rows);
    }

    inp.embd = ggml_new_tensor_2d(g.ctx0, GGML_TYPE_F32, n_embd_inp, g.n_tokens);
    ggml_set_input(inp.embd);

    return inp.embd;
}

// injection batch: the target features fused through the encoder
ggml_tensor * llama_model_dflash::build_inject_inp(llm_graph_context & g, const llama_model & model) {
    // DFly ships one fused context per draft layer, so the incoming row is n_layer wide
    const int64_t n_embd_inp = model.dfly_layer_fusion ? (int64_t) g.hparams.n_embd_out() : (int64_t) g.hparams.n_embd_inp_enc();

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd_inp);

    ggml_tensor * inp_target = build_inp_target_features(g, *inp, n_embd_inp);
    g.cb(inp_target, "inp_target_features", -1);

    g.res->add_input(std::move(inp));

    ggml_tensor * inp_g = g.build_lora_mm(model.fc, inp_target, model.fc_s);
    return g.build_norm(inp_g, model.output_norm_enc, nullptr, LLM_NORM_RMS, -1);
}

// tok_embd from the target model (shared via ctx_other) when the draft has none
ggml_tensor * llama_model_dflash::get_tok_embd(const llm_graph_context & g, const llama_model & model) {
    if (model.tok_embd) {
        return model.tok_embd;
    }
    GGML_ASSERT(g.cparams.ctx_other != nullptr);
    const auto * model_other = llama_get_model(g.cparams.ctx_other);

    GGML_ASSERT(model_other->tok_embd != nullptr && "DFlash draft requires the target model's token embeddings");
    return model_other->tok_embd;
}

// lm_head and its scale from the target model (shared via ctx_other) when the draft has none
std::pair<ggml_tensor *, ggml_tensor *> llama_model_dflash::get_lm_head(const llm_graph_context & g, const llama_model & model) {
    if (model.output) {
        return { model.output, model.output_s };
    }
    GGML_ASSERT(g.cparams.ctx_other != nullptr);
    const auto * model_other = llama_get_model(g.cparams.ctx_other);

    GGML_ASSERT(model_other->output != nullptr && "DFlash draft requires the target model's output projection");
    return { model_other->output, model_other->output_s };
}

// DFlash decoder, dual-mode by batch type:
//   * embd batch  -> fused target features: project + inject K/V into the cache.
//   * token batch -> noise-block diffusion: attend over [committed, MASK...] to generate draft tokens
llama_model_dflash::graph_base::graph_base(const llm_graph_params & params) : llm_graph_context(params) {
    n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    inp_pos = build_inp_pos();

    // optional iSWA: pick the matching attention input
    use_iswa = hparams.swa_type != LLAMA_SWA_TYPE_NONE;
    if (use_iswa) {
        inp_attn_iswa = build_attn_inp_kv_iswa();
    } else {
        inp_attn = build_attn_inp_kv();
    }

    // drafts for M-RoPE targets use degenerate sections (temporal dim only)
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);
}

ggml_tensor * llama_model_dflash::graph_base::build_rope(ggml_tensor * cur, ggml_tensor * pos, int il) {
    if (!hparams.has_rope(il)) {
        return cur;
    }

    return rope_type == GGML_ROPE_TYPE_MROPE
        ? ggml_rope_multi(ctx0, cur, pos, nullptr,
                n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow)
        : ggml_rope_ext(ctx0, cur, pos, model.layers[0].rope_freqs,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
}

// KV cache injection
ggml_tensor * llama_model_dflash::graph_base::build_inject() {
    ggml_tensor * inp_g = build_inject_inp(*this, model);
    cb(inp_g, "inp_g_embeddings", -1);

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        // plain DFlash injects one shared context into every layer; DFly slices out this
        // layer's own fused context (encoder writes them layer-major within each token)
        ggml_tensor * ctx_il = inp_g;
        if (model.dfly_layer_fusion) {
            ctx_il = ggml_cont(ctx0, ggml_view_2d(ctx0, inp_g, n_embd, n_tokens,
                        inp_g->nb[1], (size_t) il*n_embd*inp_g->nb[0]));
            cb(ctx_il, "dfly_ctx_layer", il);
        }

        ggml_tensor * Kcur = build_lora_mm(layer.wk, ctx_il, layer.wk_s);
        const bool shared_kv = layer.wv == nullptr;
        ggml_tensor * Vcur = shared_kv ? Kcur : build_lora_mm(layer.wv, ctx_il, layer.wv_s);

        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

        Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);
        if (shared_kv) {
            Vcur = ggml_rms_norm(ctx0, Vcur, hparams.f_norm_rms_eps);
        }
        Kcur = build_rope(Kcur, inp_pos, il);
        cb(Kcur, "Kcur_injected", il);
        cb(Vcur, "Vcur_injected", il);

        if (use_iswa) {
            // route each layer's K/V to its sub-cache: SWA layers -> sliding cache, full -> dense
            const bool    is_swa = hparams.is_swa(il);
            const auto  * kv     = is_swa ? inp_attn_iswa->mctx->get_swa() : inp_attn_iswa->mctx->get_base();
            ggml_tensor * k_idxs = is_swa ? inp_attn_iswa->get_k_idxs_swa() : inp_attn_iswa->get_k_idxs();
            ggml_tensor * v_idxs = is_swa ? inp_attn_iswa->get_v_idxs_swa() : inp_attn_iswa->get_v_idxs();
            // rotate K/V into the cache's rotated space
            ggml_tensor * k_rot  = is_swa ? inp_attn_iswa->self_k_rot_swa[il] : inp_attn_iswa->self_k_rot[il];
            ggml_tensor * v_rot  = is_swa ? inp_attn_iswa->self_v_rot_swa[il] : inp_attn_iswa->self_v_rot[il];
            if (k_rot) {
                Kcur = llama_mul_mat_hadamard(ctx0, Kcur, k_rot);
            }
            if (v_rot) {
                Vcur = llama_mul_mat_hadamard(ctx0, Vcur, v_rot);
            }
            ggml_build_forward_expand(gf, kv->cpy_k(ctx0, Kcur, k_idxs, il));
            ggml_build_forward_expand(gf, kv->cpy_v(ctx0, Vcur, v_idxs, il));
        } else {
            // rotate K/V into the cache's rotated space
            if (inp_attn->self_k_rot[il]) {
                Kcur = llama_mul_mat_hadamard(ctx0, Kcur, inp_attn->self_k_rot[il]);
            }
            if (inp_attn->self_v_rot[il]) {
                Vcur = llama_mul_mat_hadamard(ctx0, Vcur, inp_attn->self_v_rot[il]);
            }
            ggml_build_forward_expand(gf, inp_attn->mctx->cpy_k(ctx0, Kcur, inp_attn->get_k_idxs(), il));
            ggml_build_forward_expand(gf, inp_attn->mctx->cpy_v(ctx0, Vcur, inp_attn->get_v_idxs(), il));
        }
    }

    res->t_embd = inp_g;

    ggml_build_forward_expand(gf, inp_g);

    return inp_g;
}

ggml_tensor * llama_model_dflash::graph_base::build_inp_tokens() {
    ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp_tokens);
    res->t_inp_tokens = inp_tokens;

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd);
    inp->tokens = inp_tokens;
    res->add_input(std::move(inp));

    return inp_tokens;
}

ggml_tensor * llama_model_dflash::graph_base::build_layers(ggml_tensor * inp_tokens) {
    const float kq_scale = hparams.f_attention_scale != 0.0f ? hparams.f_attention_scale : 1.0f/sqrtf(float(n_embd_head));

    ggml_tensor * inpL = ggml_get_rows(ctx0, get_tok_embd(*this, model), inp_tokens);
    if (hparams.f_embedding_scale != 0.0f) {
        inpL = ggml_scale(ctx0, inpL, hparams.f_embedding_scale);
    }
    cb(inpL, "inp_noise_embd", -1);

    inp_noise_embd = inpL;

    // dspark GIDD log-SNR conditioning (LogSnrEmbed): added to the draft noise
    // embedding before the layer loop, matching the training reference. The
    // per-position log-SNR is the fixed round-1 inference convention: each block's
    // anchor (row 0; the ubatch is block-major, as build_dspark_markov_head also
    // relies on) at max_log_snr, every masked position at min_log_snr.
    if (hparams.dspark_log_snr_conditioning) {
        GGML_ASSERT(model.dspark_log_snr_fc1_w && model.dspark_log_snr_fc2_w &&
                    model.dspark_log_snr_fc1_b && model.dspark_log_snr_fc2_b);

        const int64_t n_blocks = ubatch.n_seqs_unq;
        GGML_ASSERT(n_blocks > 0 && n_tokens % n_blocks == 0 && "log-SNR conditioning requires equal-size blocks");
        const int64_t block_drafts = n_tokens / n_blocks;

        const int64_t n_freq  = 128;
        const int64_t half    = n_freq / 2;
        const float   min_snr = hparams.dspark_min_log_snr;
        const float   max_snr = hparams.dspark_max_log_snr;

        // host-side port of LogSnrEmbed.forward's featurization fused with the
        // anchor/mask pattern above. A pure function of the values below, all known
        // here, so precomputing keeps it auditable against the python reference
        // instead of chaining ggml_arange/sin/cos in-graph.
        std::vector<float> feat((size_t) (n_freq * n_tokens));
        for (int64_t pos = 0; pos < n_tokens; ++pos) {
            const float log_snr = (pos % block_drafts == 0) ? max_snr : min_snr;
            const float tt      = (log_snr - min_snr) / (max_snr - min_snr) * 1000.0f;
            for (int64_t i = 0; i < half; ++i) {
                const float freq  = expf(-logf(10000.0f) * (float) i / (float) half);
                const float angle = tt * freq;
                feat[(size_t) (pos * n_freq + i)]        = sinf(angle);
                feat[(size_t) (pos * n_freq + half + i)] = cosf(angle);
            }
        }

        auto logsnr_input = std::make_unique<llm_graph_input_dspark_logsnr>(std::move(feat));
        logsnr_input->feat = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_freq, n_tokens);
        ggml_set_input(logsnr_input->feat);
        ggml_set_name(logsnr_input->feat, "dspark_log_snr_feat");
        ggml_tensor * snr_feat = logsnr_input->feat;
        res->add_input(std::move(logsnr_input));

        ggml_tensor * snr_hidden = build_lora_mm(model.dspark_log_snr_fc1_w, snr_feat);
        snr_hidden = ggml_add(ctx0, snr_hidden, model.dspark_log_snr_fc1_b);
        snr_hidden = ggml_silu(ctx0, snr_hidden);
        cb(snr_hidden, "dspark_log_snr_fc1", -1);

        ggml_tensor * snr_embed = build_lora_mm(model.dspark_log_snr_fc2_w, snr_hidden);
        snr_embed = ggml_add(ctx0, snr_embed, model.dspark_log_snr_fc2_b);
        cb(snr_embed, "dspark_log_snr_fc2", -1);

        inpL = ggml_add(ctx0, inpL, snr_embed);
        cb(inpL, "dspark_draft_embd_snr", -1);
    }

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * noise_norm = build_norm(inpL, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(noise_norm, "noise_norm", il);

        ggml_tensor * attn_dynamic = nullptr;
        if (layer.dflash_attn_conv_proj) {
            attn_dynamic = build_lora_mm(layer.dflash_attn_conv_proj, noise_norm);
            noise_norm = build_dflash2_conv(*this, noise_norm, attn_dynamic, layer.dflash_attn_conv_base, 0);
            cb(noise_norm, "attn_conv_in", il);
        }

        ggml_tensor * Qcur = build_lora_mm(layer.wq, noise_norm, layer.wq_s);
        ggml_tensor * Kcur = build_lora_mm(layer.wk, noise_norm, layer.wk_s);
        const bool shared_kv = layer.wv == nullptr;
        ggml_tensor * Vcur = shared_kv ? Kcur : build_lora_mm(layer.wv, noise_norm, layer.wv_s);

        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

        Qcur = build_norm(Qcur, layer.attn_q_norm, NULL, LLM_NORM_RMS, il);
        Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);
        if (shared_kv) {
            Vcur = ggml_rms_norm(ctx0, Vcur, hparams.f_norm_rms_eps);
        }

        Qcur = build_rope(Qcur, inp_pos, il);
        Kcur = build_rope(Kcur, inp_pos, il);
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        // cache-aware, non-causal attention
        ggml_tensor * cur = use_iswa
            ? build_attn(inp_attn_iswa, layer.wo, NULL, layer.wo_s, Qcur, Kcur, Vcur, nullptr, layer.attn_sinks, nullptr, kq_scale, il)
            : build_attn(inp_attn,      layer.wo, NULL, layer.wo_s, Qcur, Kcur, Vcur, nullptr, layer.attn_sinks, nullptr, kq_scale, il);

        if (hparams.f_attn_value_scale != 0.0f) {
            cur = ggml_scale(ctx0, cur, hparams.f_attn_value_scale);
            cb(cur, "attn_out_scaled", il);
        }

        if (attn_dynamic) {
            cur = build_dflash2_conv(*this, cur, attn_dynamic, layer.dflash_attn_conv_base, 1);
            cb(cur, "attn_conv_out", il);
        }

        if (layer.attn_post_norm) {
            cur = build_norm(cur, layer.attn_post_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "attn_post_norm", il);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        ggml_tensor * ffn_dynamic = nullptr;
        if (layer.dflash_ffn_conv_proj) {
            ffn_dynamic = build_lora_mm(layer.dflash_ffn_conv_proj, cur);
            cur = build_dflash2_conv(*this, cur, ffn_dynamic, layer.dflash_ffn_conv_base, 0);
            cb(cur, "ffn_conv_in", il);
        }

        cur = build_ffn(cur,
                layer.ffn_up,   NULL, layer.ffn_up_s,
                layer.ffn_gate, NULL, layer.ffn_gate_s,
                layer.ffn_down, NULL, layer.ffn_down_s,
                NULL,
                hparams.llm_ffn_op, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        if (ffn_dynamic) {
            cur = build_dflash2_conv(*this, cur, ffn_dynamic, layer.dflash_ffn_conv_base, 1);
            cb(cur, "ffn_conv_out", il);
        }

        if (layer.ffn_post_norm) {
            cur = build_norm(cur, layer.ffn_post_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "ffn_post_norm", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        if (layer.out_scale) {
            cur = ggml_mul(ctx0, cur, layer.out_scale);
        }
        cb(cur, "l_out", il);

        inpL = cur;
    }

    ggml_tensor * cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    res->t_embd = cur;

    return cur;
}

ggml_tensor * llama_model_dflash::graph_base::build_logits(ggml_tensor * cur) {
    const auto [output, output_s] = get_lm_head(*this, model);
    cur = build_lora_mm(output, cur, output_s);

    // DFlash2 feeds these logits to the selector, so they need the target's output
    // transforms; DFlash1 and DSpark read them through the sampler instead
    if (model.dflash_selector_hidden) {
        if (hparams.f_logit_scale != 0.0f) {
            cur = ggml_scale(ctx0, cur, hparams.f_logit_scale);
        }
        if (hparams.f_final_logit_softcapping > 0.0f) {
            cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
            cur = ggml_tanh(ctx0, cur);
            cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);
        }
    }

    // reduced-draft-vocab exports: scatter the draft logits to the target vocabulary via d2t
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

    return cur;
}

// DFly (AngelSpec): TreeFlash predecessor correction chained across a block.
//
// Position i's logits come from the draft hidden state at row i corrected by the embedding
// of the token drafted at row i-1, then projected through the target head. Slot 0 of each
// block is the committed anchor, not a prediction slot, so the chain seeds from the anchor
// TOKEN and runs i = 1..block_drafts-1 -- the same row convention the DFlash reader in
// common/speculative.cpp uses (rows 1..n-1), which is why a DFly drafter must NOT report a
// DSpark Markov head.
//
// Reference: _DflyDraftSampler.__call__ + DFlyHiddenStatesCorrection.forward.
static void build_dfly_correction_head(llm_graph_context & g, const llama_model & model,
        ggml_tensor * inp_embd_raw) {
    ggml_context * ctx0 = g.ctx0;
    auto         & res  = g.res;

    GGML_ASSERT(model.dfly_hc_hidden_norm && model.dfly_hc_embed_norm &&
                model.dfly_hc_gate && model.dfly_hc_up && model.dfly_hc_down &&
                "DFly predecessor-correction weights not loaded");

    ggml_tensor * hidden = res->t_embd;   // [n_embd, n_tokens], after the decoder's final norm
    ggml_tensor * base   = res->t_logits; // [n_vocab, n_tokens], uncorrected

    const int64_t n_embd  = hidden->ne[0];
    const int64_t n_tok   = base->ne[1];
    const int64_t n_vocab = base->ne[0];

    const auto it = model.gguf_kv.find("dflash.block_size");
    GGML_ASSERT(it != model.gguf_kv.end() && "DFly draft requires 'dflash.block_size' in GGUF metadata");
    const int64_t block_size = std::stoi(it->second);
    GGML_ASSERT(block_size > 0);

    const int64_t n_blocks = g.ubatch.n_seqs_unq;
    GGML_ASSERT(n_blocks > 0 && n_tok % n_blocks == 0 && "DFly head requires equal-size blocks");
    const int64_t block_drafts = n_tok / n_blocks;
    if (block_drafts > block_size) {
        return;
    }

    // Only a noise block carries a chain. The draft context also decodes ordinary staging
    // batches, whose slot 0 is the caller's id_last -- LLAMA_TOKEN_NULL until the first token
    // is committed, which would index the embedding table with row -1. A block is
    // [id_last, MASK, ...] per sequence, so check that shape before building anything.
    if (g.ubatch.token == nullptr) {
        return;
    }

    const llama_token mask_id = model.vocab.token_mask();

    for (int64_t b = 0; b < n_blocks; ++b) {
        const llama_token anchor = g.ubatch.token[b*block_drafts];
        if (anchor < 0 || anchor >= (llama_token) model.vocab.n_tokens()) {
            return;
        }
        for (int64_t i = 1; i < block_drafts; ++i) {
            if (g.ubatch.token[b*block_drafts + i] != mask_id) {
                return;
            }
        }
    }

    // the target's head and embeddings when the draft ships none (shared via ctx_other)
    auto * output   = model.output;
    auto * output_s = model.output_s;
    auto * tok_embd = model.tok_embd;
    if (output == nullptr || tok_embd == nullptr) {
        GGML_ASSERT(g.cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(g.cparams.ctx_other);
        if (output == nullptr) {
            GGML_ASSERT(model_other->output != nullptr && "DFly head requires the target output projection");
            output   = model_other->output;
            output_s = model_other->output_s;
        }
        if (tok_embd == nullptr) {
            GGML_ASSERT(model_other->tok_embd != nullptr && "DFly head requires the target token embeddings");
            tok_embd = model_other->tok_embd;
        }
    }


    const size_t hidden_stride = (size_t) block_drafts * hidden->nb[1];
    const size_t base_stride   = (size_t) block_drafts * base->nb[1];

    // position 1 conditions on the block anchor: take its embedding from the rows already
    // gathered by the decoder. prev stays null until the chain produces its first token.
    ggml_tensor * prev      = nullptr;
    ggml_tensor * prev_embd = ggml_cont(ctx0, ggml_view_2d(ctx0, inp_embd_raw, n_embd, n_blocks,
                (size_t) block_drafts * inp_embd_raw->nb[1], 0));

    // the anchor slot is not predicted: pass its uncorrected logits through so the output
    // keeps one row per ubatch token
    ggml_tensor * cat = ggml_cont(ctx0, ggml_view_2d(ctx0, base, n_vocab, n_blocks, base_stride, 0));

    for (int64_t i = 1; i < block_drafts; ++i) {
        // predecessor correction: delta = down(silu(gate(z)) * up(z)),
        //                         z = [rms(h_i); rms(embd(prev))]
        if (prev) {
            // chained positions condition on the previously drafted token: an argmax, always in range
            prev_embd = ggml_get_rows(ctx0, tok_embd, prev); // [n_embd, n_blocks]
        }

        ggml_tensor * h_i = ggml_cont(ctx0, ggml_view_2d(ctx0, hidden, n_embd, n_blocks,
                    hidden_stride, i*hidden->nb[1]));

        ggml_tensor * z = ggml_concat(ctx0,
                g.build_norm(h_i,       model.dfly_hc_hidden_norm, NULL, LLM_NORM_RMS, -1),
                g.build_norm(prev_embd, model.dfly_hc_embed_norm,  NULL, LLM_NORM_RMS, -1), 0);

        ggml_tensor * delta = g.build_ffn(z,
                model.dfly_hc_up,   NULL, NULL,
                model.dfly_hc_gate, NULL, NULL,
                model.dfly_hc_down, NULL, NULL,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, -1);

        ggml_tensor * col = g.build_lora_mm(output, ggml_add(ctx0, h_i, delta), output_s);

        cat = ggml_concat(ctx0, cat, col, 1);

        // greedy chain: the next position conditions on this position's drafted token
        if (i + 1 < block_drafts) {
            prev = ggml_argmax(ctx0, col);
        }
    }

    // cat is position-major; restore ubatch block-major order
    ggml_tensor * out = ggml_reshape_3d(ctx0, cat, n_vocab, n_blocks, block_drafts);
    out = ggml_cont(ctx0, ggml_permute(ctx0, out, 0, 2, 1, 3)); // [n_vocab, block_drafts, n_blocks]
    out = ggml_reshape_2d(ctx0, out, n_vocab, n_tok);

    res->t_logits = out;
    ggml_build_forward_expand(g.gf, out);
}

llama_model_dflash::graph_dflash::graph_dflash(const llm_graph_params & params) : graph_base(params) {
    ASSERT_EMBD_OR_TOKEN(ubatch);
    if (ubatch.embd || ubatch.embd_row) {
        build_inject();
        return;
    }

    ggml_tensor * inp_tokens = build_inp_tokens();
    ggml_tensor * cur        = build_layers(inp_tokens);

    build_logits(cur);

    // DFly: re-derive the draft logits through the chained predecessor correction
    // LLAMA_DFLY_NO_CHAIN drops the correction, for A/B measurement only: DFly's acceptance
    // depends on it, so a run without it is not a meaningful DFly configuration.
    if (model.dfly_hc_down && getenv("LLAMA_DFLY_NO_CHAIN") == nullptr) {
        build_dfly_correction_head(*this, model, inp_noise_embd);
    }
}

std::unique_ptr<llm_graph_context> llama_model_dflash::build_arch_graph(const llm_graph_params & params) const {
    switch (params.gtype) {
        case LLM_GRAPH_TYPE_ENCODER:
            return std::make_unique<graph<true>>(params);
        case LLM_GRAPH_TYPE_DEFAULT:
        case LLM_GRAPH_TYPE_DECODER:
            switch (backbone) {
                case BACKBONE_BASE:           break;
                case BACKBONE_DSV4:           return std::make_unique<graph_dsv4>(params);
                case BACKBONE_DSV41:          return std::make_unique<graph_dsv41>(params);
                case BACKBONE_MISTRAL_LARGE4: return std::make_unique<graph_mistral_large4>(params);
            }
            switch (variant) {
                case VARIANT_DFLASH:   return std::make_unique<graph_dflash>(params);
                case VARIANT_DFLASH2:  return std::make_unique<graph_dflash2>(params);
                case VARIANT_DSPARK:   return std::make_unique<graph_dspark>(params);
                case VARIANT_LILICORR: return std::make_unique<graph_lilicorr>(params);
            }
            GGML_ABORT("invalid DFlash variant");
        default:
            GGML_ABORT("invalid graph type");
    };
}
