#include "models.h"

#include "llama-impl.h"
#include "llama-memory-recurrent.h"

llm_build_delta_net_base::llm_build_delta_net_base(const llm_graph_params & params) : llm_graph_context(params) {}

llm_graph_input_tree * llm_build_delta_net_base::build_inp_tree(int64_t d_conv) {
    if (inp_tree) {
        GGML_ASSERT(inp_tree->d_conv == d_conv);
        return inp_tree;
    }

    const int64_t n_tokens = ubatch.n_tokens;

    auto inp = std::make_unique<llm_graph_input_tree>(d_conv);

    inp->parent = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->parent);

    inp->conv_win = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, d_conv*n_tokens);
    ggml_set_input(inp->conv_win);

    inp->conv_snap = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (d_conv - 1)*n_tokens);
    ggml_set_input(inp->conv_snap);

    inp_tree = inp.get();
    res->add_input(std::move(inp));

    return inp_tree;
}

std::pair<ggml_tensor *, ggml_tensor *> llm_build_delta_net_base::build_delta_net(
        ggml_tensor * q,
        ggml_tensor * k,
        ggml_tensor * v,
        ggml_tensor * g,
        ggml_tensor * b,
        ggml_tensor * s) {
    const int64_t S_k      = q->ne[0];
    const int64_t H_k      = q->ne[1];
    const int64_t n_tokens = q->ne[2];
    const int64_t n_seqs   = q->ne[3];

    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];

    GGML_ASSERT(S_k == S_v);
    GGML_ASSERT(H_v % H_k == 0);

    GGML_ASSERT(q->ne[0] == S_k && q->ne[1] == H_k && q->ne[2] == n_tokens && q->ne[3] == n_seqs);
    GGML_ASSERT(k->ne[0] == S_k && k->ne[1] == H_k && k->ne[2] == n_tokens && k->ne[3] == n_seqs);
    GGML_ASSERT(v->ne[0] == S_v && v->ne[1] == H_v && v->ne[2] == n_tokens && v->ne[3] == n_seqs);

    GGML_ASSERT(g->ne[0] == 1   || g->ne[0] == S_v);
    GGML_ASSERT(                   g->ne[1] == H_v && g->ne[2] == n_tokens && g->ne[3] == n_seqs);
    GGML_ASSERT(b->ne[0] == 1   && b->ne[1] == H_v && b->ne[2] == n_tokens && b->ne[3] == n_seqs);
    GGML_ASSERT(s->ne[0] == S_v && s->ne[1] == S_v && s->ne[2] == H_v      && s->ne[3] == n_seqs);

    // K=1: output carries the final state only. state s is 4D [S_v, S_v, H_v, n_seqs].
    const bool raw = gdn_raw_beta && gdn_raw_alpha && gdn_raw_dt_bias && gdn_raw_a;

    ggml_tensor * result = ggml_gated_delta_net(ctx0, q, k, v, raw ? gdn_raw_alpha : g, raw ? gdn_raw_beta : b, s, /*K=*/1);
    if (raw) {
        ggml_gated_delta_net_set_raw_gates(result, gdn_raw_dt_bias, gdn_raw_a);
    }
    ggml_tensor * output = ggml_view_4d(ctx0, result,
            S_v, H_v, n_tokens, n_seqs,
            ggml_row_size(result->type, S_v),
            ggml_row_size(result->type, S_v * H_v),
            ggml_row_size(result->type, S_v * H_v * n_tokens), 0);

    ggml_tensor * new_state = ggml_view_4d(ctx0, result,
            S_v, S_v, H_v, n_seqs,
            ggml_row_size(result->type, S_v),
            ggml_row_size(result->type, S_v * S_v),
            ggml_row_size(result->type, S_v * S_v * H_v),
            ggml_row_size(result->type, S_v * H_v * n_tokens * n_seqs));

    return {output, new_state};
}

ggml_tensor * llm_build_delta_net_base::build_conv_state(
        llm_graph_input_rs * inp,
        ggml_tensor *        conv_states_all,
        ggml_tensor *        qkv_mixed,
        int64_t              conv_kernel_size,
        int64_t              conv_channels,
        int                  il) {
    const int64_t n_seqs = ubatch.n_seqs;

    ggml_tensor * conv_states = build_rs(inp, conv_states_all, hparams.n_embd_r(), n_seqs);
    cb(conv_states, "conv_states", il);

    conv_states = ggml_reshape_3d(ctx0, conv_states, conv_kernel_size - 1, conv_channels, n_seqs);
    cb(conv_states, "conv_states_reshaped", il);

    if (ubatch.tree_parent) {
        // token tree: every token convolves over its own ancestor chain, so gather a window per token
        // and run them as n_tokens sequences of one token
        const int64_t n_tokens = ubatch.n_tokens;

        GGML_ASSERT(n_seqs == 1);
        GGML_ASSERT((int64_t) cparams.n_rs_seq + 1 >= n_tokens);

        llm_graph_input_tree * tree = build_inp_tree(conv_kernel_size);

        const auto     kv_head  = inp->mctx->get_head();
        const uint32_t mem_size = inp->mctx->get_size();

        ggml_tensor * qkv_2d = ggml_is_contiguous(qkv_mixed)
            ? ggml_reshape_2d(ctx0, qkv_mixed, conv_channels, n_tokens)
            : ggml_cont_2d(ctx0, qkv_mixed, conv_channels, n_tokens);

        // one row per time step: [conv_channels, d_conv - 1 + n_tokens]
        ggml_tensor * steps = ggml_concat(ctx0,
                ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_2d(ctx0, conv_states, conv_kernel_size - 1, conv_channels))),
                qkv_2d, 1);
        cb(steps, "conv_steps", il);

        ggml_tensor * win = ggml_get_rows(ctx0, steps, tree->conv_win);
        win = ggml_reshape_3d(ctx0, win, conv_channels, conv_kernel_size, n_tokens);
        win = ggml_cont(ctx0, ggml_permute(ctx0, win, 1, 0, 2, 3));
        cb(win, "conv_input_tree", il);

        // snapshot slot s gets the conv state after token n_tokens - 1 - s
        ggml_tensor * snap = ggml_get_rows(ctx0, steps, tree->conv_snap);
        snap = ggml_reshape_3d(ctx0, snap, conv_channels, conv_kernel_size - 1, n_tokens);
        snap = ggml_permute(ctx0, snap, 1, 0, 2, 3);

        const size_t row_size_tree = ggml_row_size(conv_states_all->type, (conv_kernel_size - 1) * conv_channels);

        ggml_tensor * snap_dst = ggml_view_2d(ctx0, conv_states_all,
                (conv_kernel_size - 1) * conv_channels, n_tokens,
                (size_t) mem_size * row_size_tree, (size_t) kv_head * row_size_tree);

        ggml_build_forward_expand(gf, ggml_cpy(ctx0, snap, snap_dst));

        return win;
    }

    qkv_mixed = ggml_transpose(ctx0, qkv_mixed);
    cb(qkv_mixed, "qkv_mixed_transposed", il);

    ggml_tensor * conv_input = ggml_concat(ctx0, conv_states, qkv_mixed, 0);
    cb(conv_input, "conv_input", il);

    build_conv_state_store(inp, conv_states_all, conv_input, conv_kernel_size - 1, 0);

    return conv_input;
}

ggml_tensor * llm_build_delta_net_base::build_recurrent_attn(
        llm_graph_input_rs * inp,
        ggml_tensor *        ssm_states_all,
        ggml_tensor *        q,
        ggml_tensor *        k,
        ggml_tensor *        v,
        ggml_tensor *        g,
        ggml_tensor *        b,
        ggml_tensor *        s,
        int                  il,
        ggml_tensor *        state_rows) {
    const auto * mctx_cur   = inp->mctx;
    const auto   kv_head    = mctx_cur->get_head();
    const uint32_t mem_size = mctx_cur->get_size();

    // dims from v (always (S_v, H_v, T, B)): in rows mode `s` is the 2D cache
    // view, so its shape no longer carries them
    const int64_t S_v          = v->ne[0];
    const int64_t H_v          = v->ne[1];
    const int64_t n_seqs       = v->ne[3];
    const int64_t n_seq_tokens = q->ne[2];

    const bool keep = cparams.n_rs_seq > 0;

    GGML_ASSERT(state_rows == nullptr || keep); // rows mode is a ring-path optimization

    if (!keep) {
        auto attn_out = build_delta_net(q, k, v, g, b, s);
        ggml_tensor * output    = attn_out.first;
        ggml_tensor * new_state = attn_out.second;
        cb(output, "attn_output", il);
        cb(new_state, "new_state", il);

        ggml_build_forward_expand(gf,
                ggml_cpy(ctx0, new_state,
                    ggml_view_2d(ctx0, ssm_states_all, hparams.n_embd_s(), n_seqs, ssm_states_all->nb[1],
                        kv_head * hparams.n_embd_s() * ggml_element_size(ssm_states_all))));

        return output;
    }

    const int64_t D = S_v * S_v * H_v;
    const int64_t K = cparams.n_rs_seq + 1;

    const bool raw = gdn_raw_beta && gdn_raw_alpha && gdn_raw_dt_bias && gdn_raw_a;
    ggml_tensor * gg = raw ? gdn_raw_alpha : g;
    ggml_tensor * bb = raw ? gdn_raw_beta  : b;

    ggml_tensor * gdn_out;
    if (state_rows) {
        // rows mode: the fused op reads each seq's live state directly from the
        // 2D cache view at row state_rows[seq] -- no gather
        gdn_out = ggml_gated_delta_net_rows(ctx0, q, k, v, gg, bb, s, state_rows, (int) K);
    } else {
        // state s is 4D [S_v, S_v, H_v, n_seqs]; K snapshot slots are written into the output.
        gdn_out = ggml_gated_delta_net(ctx0, q, k, v, gg, bb, s, K);
    }
    if (raw) {
        ggml_gated_delta_net_set_raw_gates(gdn_out, gdn_raw_dt_bias, gdn_raw_a);
    }
    if (ubatch.tree_parent) {
        GGML_ASSERT(inp_tree && state_rows == nullptr);
        ggml_gated_delta_net_set_tree(gdn_out, inp_tree->parent);
    }
    const int64_t attn_score_elems    = S_v * H_v * n_seq_tokens * n_seqs;
    const int64_t state_size_per_snap = S_v * S_v * H_v * n_seqs;

    ggml_tensor * output = ggml_view_4d(ctx0, gdn_out,
        S_v, H_v, n_seq_tokens, n_seqs,
        ggml_row_size(gdn_out->type, S_v),
        ggml_row_size(gdn_out->type, S_v * H_v),
        ggml_row_size(gdn_out->type, S_v * H_v * n_seq_tokens),
        0);
    cb(output, "attn_output", il);

    const size_t row_size = hparams.n_embd_s() * ggml_element_size(ssm_states_all);

    // op writes the last min(n_seq_tokens, K) snapshots; trailing slots are left unwritten
    const int64_t n_written = std::min<int64_t>(n_seq_tokens, K);

    if (state_rows) {
        // rows mode: scatter the compact snapshot region [D, n_written*n_seqs]
        // (slot-major, starting right after the attention scores) into the
        // cache rows slot*mem_size + kv_head + seq -- same destinations as the
        // strided cpy below, but expressed as SET_ROWS so the Metal backend
        // can fold the scatter into the fused op's epilogue.
        ggml_tensor * snaps = ggml_view_2d(ctx0, gdn_out,
            D, n_written * n_seqs,
            ggml_row_size(gdn_out->type, D),
            ggml_row_size(gdn_out->type, attn_score_elems));

        ggml_tensor * write_rows = build_rs_write_rows(inp, K, n_seq_tokens, n_seqs);

        ggml_build_forward_expand(gf, ggml_set_rows(ctx0, ssm_states_all, snaps, write_rows));

        return output;
    }

    // write the produced snapshots into the recurrent cache (snapshot slot i -> rollback group i)
    ggml_tensor * src = ggml_view_3d(ctx0, gdn_out,
        D, n_seqs, n_written,
        ggml_row_size(gdn_out->type, D),
        ggml_row_size(gdn_out->type, state_size_per_snap),
        ggml_row_size(gdn_out->type, attn_score_elems));

    ggml_tensor * dst = ggml_view_3d(ctx0, ssm_states_all,
        D, n_seqs, n_written,
        ssm_states_all->nb[1],
        (size_t) mem_size * row_size,
        (size_t) kv_head * row_size);

    ggml_build_forward_expand(gf, ggml_cpy(ctx0, src, dst));

    return output;
}
