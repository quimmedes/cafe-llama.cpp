#include "models.h"

int llama_model_inkling2_loop::layer_phys(int il) const {
    return il < n_phys ? il : core_start + (il - n_phys) % (core_end - core_start);
}

void llama_model_inkling2_loop::load_arch_hparams(llama_model_loader & ml) {
    llama_model_inkling2::load_arch_hparams(ml);

    // Unsloth GGUFs use the core keys and count the layer visits in block_count
    visit_indexed = !ml.get_key(LLM_KV_LOOP_START, core_start, false);
    if (visit_indexed) {
        ml.get_key(LLM_KV_LOOP_CORE_START, core_start);
        ml.get_key(LLM_KV_LOOP_CORE_END,   core_end);
        ml.get_key(LLM_KV_NUM_LOOPS,       n_passes);
    } else {
        ml.get_key(LLM_KV_LOOP_END,   core_end);
        ml.get_key(LLM_KV_LOOP_COUNT, n_passes);
    }
    ml.get_key(LLM_KV_LOOP_GATE_RANK, gate_rank);
    ml.get_key(LLM_KV_LOOP_VARYING,   varying, false);

    auto copy_layer = [&](int dst, int src) {
        hparams.n_head_arr[dst]        = hparams.n_head_arr[src];
        hparams.n_head_kv_arr[dst]     = hparams.n_head_kv_arr[src];
        hparams.n_ff_arr[dst]          = hparams.n_ff_arr[src];
        hparams.n_ff_exp_arr[dst]      = hparams.n_ff_exp_arr[src];
        hparams.n_expert_used_arr[dst] = hparams.n_expert_used_arr[src];
        hparams.is_swa_impl[dst]       = hparams.is_swa_impl[src];
        hparams.is_recr_impl[dst]      = hparams.is_recr_impl[src];
    };

    n_phys = hparams.n_layer();
    if (visit_indexed && core_start < core_end && n_passes > 0) {
        // the visits are [0, core_end), the later passes over the core, then [core_end, n_phys)
        const uint64_t n_repeat = (uint64_t) (n_passes - 1)*(core_end - core_start);
        if (n_repeat >= (uint64_t) n_phys) {
            throw std::runtime_error(format("inkling2_loop: %u passes over [%u, %u) exceed the %d layer visits", n_passes, core_start, core_end, n_phys));
        }
        n_phys -= (int) n_repeat;
        for (int il = (int) core_end; il < n_phys; ++il) {
            copy_layer(il, il + (int) n_repeat);
        }
        hparams.n_layer_all = n_phys;
    }

    const uint32_t block = hparams.attn_res_block_size;
    // a fixed loop needs at least 2 passes, a variable loop at least 1
    if (n_passes < (varying ? 1u : 2u) || !gate_rank || !core_start || core_start >= core_end || core_end > (uint32_t) n_phys ||
        core_start % block || core_end % block || hparams.n_layer_nextn) {
        throw std::runtime_error(format("inkling2_loop: invalid loop core [%u, %u) with block size %u, %u passes or gate rank %u for %d layers",
                core_start, core_end, block, n_passes, gate_rank, n_phys));
    }
    const size_t n_slot = n_phys + (size_t) (n_passes - 1)*(core_end - core_start);
    if (n_slot > LLAMA_MAX_LAYERS) {
        throw std::runtime_error(format("inkling2_loop: %zu cache slots exceed the maximum of %d", n_slot, LLAMA_MAX_LAYERS));
    }

    // expand to the cache slots before load_tensors() allocates the layers and the memory
    hparams.n_layer_all = n_slot;
    for (int il = n_phys; il < (int) n_slot; ++il) {
        copy_layer(il, layer_phys(il));
    }
}

void llama_model_inkling2_loop::load_arch_tensors(llama_model_loader & ml) {
    llama_model_inkling2::load_arch_tensors(ml);

    const int64_t n_embd = hparams.n_embd;
    const int64_t rank   = gate_rank;
    const int64_t n_core = core_end - core_start;

    // down, gain and forget tensors of the q, k, v and ffn controllers; Unsloth GGUFs name them attn, k, v and mlp
    const llm_tensor ctrl[2][3][LOOP_CTRL_COUNT] = {
        {
            { LLM_TENSOR_LOOP_Q_CTRL_DOWN,   LLM_TENSOR_LOOP_K_CTRL_DOWN,   LLM_TENSOR_LOOP_V_CTRL_DOWN,   LLM_TENSOR_LOOP_FFN_CTRL_DOWN   },
            { LLM_TENSOR_LOOP_Q_CTRL_GAIN,   LLM_TENSOR_LOOP_K_CTRL_GAIN,   LLM_TENSOR_LOOP_V_CTRL_GAIN,   LLM_TENSOR_LOOP_FFN_CTRL_GAIN   },
            { LLM_TENSOR_LOOP_Q_CTRL_FORGET, LLM_TENSOR_LOOP_K_CTRL_FORGET, LLM_TENSOR_LOOP_V_CTRL_FORGET, LLM_TENSOR_LOOP_FFN_CTRL_FORGET },
        },
        {
            { LLM_TENSOR_LOOP_ATTN_DOWN,   LLM_TENSOR_LOOP_K_DOWN,   LLM_TENSOR_LOOP_V_DOWN,   LLM_TENSOR_LOOP_MLP_DOWN   },
            { LLM_TENSOR_LOOP_ATTN_GAIN,   LLM_TENSOR_LOOP_K_GAIN,   LLM_TENSOR_LOOP_V_GAIN,   LLM_TENSOR_LOOP_MLP_GAIN   },
            { LLM_TENSOR_LOOP_ATTN_FORGET, LLM_TENSOR_LOOP_K_FORGET, LLM_TENSOR_LOOP_V_FORGET, LLM_TENSOR_LOOP_MLP_FORGET },
        },
    };
    const auto & names = ctrl[visit_indexed];

    if (visit_indexed) {
        loop_res_query    = create_tensor(tn(LLM_TENSOR_LOOP_RES_QUERY,    "weight"), {n_embd, 2*n_core}, 0);
        loop_kv_res_query = create_tensor(tn(LLM_TENSOR_LOOP_KV_RES_QUERY, "weight"), {n_embd, 2*n_core}, 0);
    }

    loop_layers.resize(n_core);
    for (uint32_t il = core_start; il < core_end; ++il) {
        auto & l = loop_layers[il - core_start];
        // the q, k, v and ffn message queries
        if (!visit_indexed) {
            l.msg_query = create_tensor(tn(LLM_TENSOR_LOOP_MSG_QUERY, "weight", il), {n_embd, LOOP_CTRL_COUNT}, 0);
        }
        for (int c = 0; c < LOOP_CTRL_COUNT; ++c) {
            l.ctrl_down[c]   = create_tensor(tn(names[0][c], "weight", il), {2*n_embd, rank}, 0);
            l.ctrl_gain[c]   = create_tensor(tn(names[1][c], "weight", il), {rank, n_embd}, 0);
            l.ctrl_forget[c] = create_tensor(tn(names[2][c], "weight", il), {rank, n_embd}, 0);
        }
    }
}

void llama_model_inkling2_loop::post_arch_tensors(llama_model_loader &) {
    // share the physical weights, each cache slot keeps its own KV and conv state
    for (int il = n_phys; il < (int) hparams.n_layer(); ++il) {
        layers[il] = layers[layer_phys(il)];
    }
}

std::unique_ptr<llm_graph_context> llama_model_inkling2_loop::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(params);
}

llama_model_inkling2_loop::graph::graph(const llm_graph_params & params) : llama_model_inkling2::graph(params, no_build{}) {
    const auto & m = static_cast<const llama_model_inkling2_loop &>(model);
    const int start = m.core_start;
    const int end   = m.core_end;

    // the q, k, v and ffn message queries of each core layer
    std::vector<ggml_tensor *> msg_query(end - start);
    for (int i = 0; i < end - start; ++i) {
        msg_query[i] = m.loop_layers[i].msg_query;
        if (!msg_query[i]) {
            auto rows = [&](ggml_tensor * t, int r, int n) { return ggml_view_2d(ctx0, t, n_embd, n, t->nb[1], r*t->nb[1]); };
            ggml_tensor * q  = m.loop_res_query;
            ggml_tensor * kv = m.loop_kv_res_query;
            msg_query[i] = ggml_concat(ctx0, ggml_concat(ctx0, rows(q, 2*i, 1), rows(kv, 2*i, 2), 1), rows(q, 2*i + 1, 1), 1);
        }
    }

    auto * cur = build_inputs();
    std::vector<ggml_tensor *> blocks;
    for (int il = 0; il < start; ++il) {
        cur = build_layer(il, il, cur, blocks);
    }

    // every pass starts from the prelude output
    auto * prelude = cur;
    const auto prelude_blocks = blocks;
    for (uint32_t pass = 0; pass < m.n_passes; ++pass) {
        // the bank of the previous pass: its blocks after the first core block, and its output
        std::vector<ggml_tensor *> bank;
        ggml_tensor * bank_out = nullptr;
        if (pass > 0) {
            bank.assign(blocks.begin() + prelude_blocks.size() + 1, blocks.end());
            bank_out = cur;
            cur = prelude;
            blocks = prelude_blocks;
        }
        for (int ip = start; ip < end; ++ip) {
            const int il = pass == 0 ? ip : m.n_phys + (pass - 1)*(end - start) + (ip - start);
            if (pass == 0) {
                cur = build_layer(il, ip, cur, blocks);
                continue;
            }
            const auto & loop = m.loop_layers[ip - start];
            cur = build_layer(il, ip, cur, blocks, mix(bank, bank_out, msg_query[ip - start]), &loop);
        }
    }

    for (int il = end; il < m.n_phys; ++il) {
        cur = build_layer(il, il, cur, blocks);
    }
    build_output(cur, blocks);
}
