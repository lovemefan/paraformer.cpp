//
// Created by lovemefan on 2023/10/1.
//

#include "decoder.h"

// pre-compute cross-attention memory
static struct ggml_cgraph * paraformer_build_graph_cross(
        paraformer_context & wctx,
        paraformer_state & wstate) {
    const auto & model   = wctx.model;
    const auto & hparams = model.hparams;

    const int n_ctx   = wstate.exp_n_audio_ctx > 0 ? wstate.exp_n_audio_ctx : hparams.n_audio_ctx;
    const int n_state = hparams.n_audio_state;
    const int n_head  = hparams.n_audio_head;

    struct ggml_init_params params = {
            /*.mem_size   =*/ wstate.alloc_cross.meta.size(),
            /*.mem_buffer =*/ wstate.alloc_cross.meta.data(),
            /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);

    ggml_cgraph * gf = ggml_new_graph(ctx0);

    ggml_allocr * alloc = wstate.alloc_cross.alloc;

    struct ggml_tensor * cur = ggml_view_tensor(ctx0, wstate.embd_enc);

    struct ggml_tensor * Kscale = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    ggml_allocr_alloc(alloc, Kscale);

    if (!ggml_allocr_is_measure(alloc)) {
        ggml_set_f32(Kscale, pow(float(n_state) / n_head, -0.25));
    }

    for (int il = 0; il < model.hparams.n_text_layer; ++il) {
        auto & layer = model.layers_decoder[il];

        struct ggml_tensor* Kcross = ggml_mul_mat(ctx0,
                                                  layer.cross_attn_k_w,
                                                  cur);

        Kcross = ggml_scale(ctx0, Kcross, Kscale);

        struct ggml_tensor* Vcross = ggml_mul_mat(ctx0,
                                                  layer.cross_attn_v_w,
                                                  cur);

        Vcross = ggml_add(ctx0,
                          Vcross,
                          layer.cross_attn_v_b);

        Vcross = ggml_transpose(ctx0, ggml_reshape_2d(ctx0, Vcross, n_state, n_ctx));

        struct ggml_tensor * k = ggml_view_1d(ctx0, wstate.kv_cross.k,
                                              n_state*n_ctx,
                                              (ggml_element_size(wstate.kv_cross.k)*n_state)*(il*n_ctx));

        struct ggml_tensor * v = ggml_view_2d(ctx0, wstate.kv_cross.v, n_ctx, n_state,
                                              (   n_ctx)*ggml_element_size(wstate.kv_cross.v),
                                              (il*n_ctx)*ggml_element_size(wstate.kv_cross.v)*n_state);

        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcross, k));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcross, v));
    }

    //ggml_graph_print(gf);

    ggml_free(ctx0);

    return gf;
}

static struct ggml_cgraph * paraformer_build_graph_decoder(
        paraformer_context & wctx,
        paraformer_state   & wstate,
        paraformer_decoder & decoder,
        const paraformer_token * tokens,
        int   n_tokens,
        int   n_past) {
    const auto & model   = wctx.model;
    const auto & hparams = model.hparams;

    auto & kv_self = decoder.kv_self;

    PARAFORMER_ASSERT(!!kv_self.ctx);

    const int n_ctx   = hparams.n_text_ctx;
    const int n_state = hparams.n_text_state;
    const int n_head  = hparams.n_text_head;
    const int n_layer = hparams.n_text_layer;

    const int N = n_tokens;
    const int M = wstate.exp_n_audio_ctx > 0 ? wstate.exp_n_audio_ctx : hparams.n_audio_ctx;

    //PARAFORMER_PRINT_DEBUG("%s: n_past = %d, N = %d, M = %d, n_ctx = %d\n", __func__, n_past, N, M, n_ctx);

    struct ggml_init_params params = {
            /*.mem_size   =*/ wstate.alloc_decode.meta.size(),
            /*.mem_buffer =*/ wstate.alloc_decode.meta.data(),
            /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);

    ggml_cgraph * gf = ggml_new_graph(ctx0);

    ggml_allocr * alloc = wstate.alloc_decode.alloc;

    struct ggml_tensor * embd = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    ggml_allocr_alloc(alloc, embd);

    if (!ggml_allocr_is_measure(alloc)) {
        memcpy(embd->data, tokens, N*ggml_element_size(embd));
    }

    struct ggml_tensor * position = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    ggml_allocr_alloc(alloc, position);

    if (!ggml_allocr_is_measure(alloc)) {
        for (int i = 0; i < N; ++i) {
            ((int32_t *) position->data)[i] = n_past + i;
        }
    }

    struct ggml_tensor * KQscale = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    ggml_allocr_alloc(alloc, KQscale);

    if (!ggml_allocr_is_measure(alloc)) {
        ggml_set_f32(KQscale, pow(float(n_state)/n_head, -0.25));
    }

    // token encoding + position encoding
    struct ggml_tensor * cur =
            ggml_add(ctx0,
                     ggml_get_rows(ctx0, model.d_te, embd),
                     ggml_get_rows(ctx0, model.d_pe, position));

    struct ggml_tensor * inpL = cur;

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers_decoder[il];

        // norm
        {
            cur = ggml_norm(ctx0, inpL, hparams.eps);

            // cur = ln_0_w*cur + ln_0_b
            cur = ggml_add(ctx0,
                           ggml_mul(ctx0,
                                    cur,
                                    layer.attn_ln_0_w),
                           layer.attn_ln_0_b);
        }

        // self-attention
        {
            struct ggml_tensor * Qcur = ggml_mul_mat(ctx0,
                                                     layer.attn_q_w,
                                                     cur);

            Qcur = ggml_add(ctx0,
                            Qcur,
                            layer.attn_q_b);

            Qcur = ggml_scale(ctx0, Qcur, KQscale);

            // note: no bias for Key
            struct ggml_tensor * Kcur = ggml_mul_mat(ctx0,
                                                     layer.attn_k_w,
                                                     cur);

            Kcur = ggml_scale(ctx0, Kcur, KQscale);

            // store key and value to memory
            {
                struct ggml_tensor * Vcur = ggml_mul_mat(ctx0,
                                                         layer.attn_v_w,
                                                         cur);

                Vcur = ggml_add(ctx0,
                                Vcur,
                                layer.attn_v_b);

                Vcur = ggml_transpose(ctx0, ggml_reshape_2d(ctx0, Vcur, n_state, N));

                struct ggml_tensor * k = ggml_view_1d(ctx0, kv_self.k, N*n_state, (ggml_element_size(kv_self.k)*n_state)*(il*n_ctx + n_past));
                struct ggml_tensor * v = ggml_view_2d(ctx0, kv_self.v, N, n_state,
                                                      (   n_ctx)*ggml_element_size(kv_self.v),
                                                      (il*n_ctx)*ggml_element_size(kv_self.v)*n_state + n_past*ggml_element_size(kv_self.v));

                ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k));
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v));
            }

            // ------

            struct ggml_tensor * Q =
                    ggml_permute(ctx0,
                                 ggml_reshape_3d(ctx0, Qcur, n_state/n_head, n_head, N),
                                 0, 2, 1, 3);

            struct ggml_tensor * K =
                    ggml_view_3d(ctx0, kv_self.k,
                                 n_state/n_head, n_past + N, n_head,
                                 ggml_element_size(kv_self.k)*n_state,
                                 ggml_element_size(kv_self.k)*n_state/n_head,
                                 ggml_element_size(kv_self.k)*n_state*n_ctx*il);

            // K * Q
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);

            //struct ggml_tensor * KQ_scaled = ggml_scale(ctx0, KQ, KQ_scale);

            struct ggml_tensor * KQ_masked = ggml_diag_mask_inf(ctx0, KQ, n_past);

            struct ggml_tensor * KQ_soft_max = ggml_soft_max(ctx0, KQ_masked);

            struct ggml_tensor * V =
                    ggml_view_3d(ctx0, kv_self.v,
                                 n_past + N, n_state/n_head, n_head,
                                 n_ctx*ggml_element_size(kv_self.v),
                                 n_ctx*ggml_element_size(kv_self.v)*n_state/n_head,
                                 il*n_ctx*ggml_element_size(kv_self.v)*n_state);

            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);

            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

            cur = ggml_cpy(ctx0,
                           KQV_merged,
                           ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_state, N));
        }

        // projection
        {
            cur = ggml_mul_mat(ctx0,
                               layer.attn_ln_1_w,
                               cur);

            cur = ggml_add(ctx0,
                           cur,
                           layer.attn_ln_1_b);
        }

        // add the input
        struct ggml_tensor * inpCA = ggml_add(ctx0, cur, inpL);

        // norm
        {
            cur = ggml_norm(ctx0, inpCA, hparams.eps); // note: we use inpCA here

            // cur = ln_0_w*cur + ln_0_b
            cur = ggml_add(ctx0,
                           ggml_mul(ctx0,
                                    cur,
                                    layer.cross_attn_ln_0_w),
                           layer.cross_attn_ln_0_b);
        }

        // cross-attention
        {
            struct ggml_tensor * Qcur = ggml_mul_mat(ctx0,
                                                     layer.cross_attn_q_w,
                                                     cur);

            Qcur = ggml_add(ctx0,
                            Qcur,
                            layer.cross_attn_q_b);

            Qcur = ggml_scale(ctx0, Qcur, KQscale);

            // Kcross is already scaled
            struct ggml_tensor * Kcross =
                    ggml_view_3d(ctx0, wstate.kv_cross.k,
                                 n_state/n_head, M, n_head,
                                 ggml_element_size(wstate.kv_cross.k)*n_state,
                                 ggml_element_size(wstate.kv_cross.k)*n_state/n_head,
                                 ggml_element_size(wstate.kv_cross.k)*n_state*M*il);

            //struct ggml_tensor * Vcross =
            //    ggml_reshape_3d(ctx0,
            //            ggml_view_1d(ctx0, wstate.kv_cross.v, M*n_state, il*M*ggml_element_size(wstate.kv_cross.v)*n_state),
            //            n_state/n_head, n_head, M);

            //struct ggml_tensor * V_trans =
            //    ggml_cpy(ctx0,
            //            ggml_permute(ctx0, Vcross, 1, 2, 0, 3),
            //            ggml_new_tensor_3d(ctx0, Vcross->type, M, n_state/n_head, n_head));

            struct ggml_tensor * V =
                    ggml_view_3d(ctx0, wstate.kv_cross.v,
                                 M, n_state/n_head, n_head,
                                 M*ggml_element_size(wstate.kv_cross.v),
                                 M*ggml_element_size(wstate.kv_cross.v)*n_state/n_head,
                                 il*M*ggml_element_size(wstate.kv_cross.v)*n_state);

            // ------

            struct ggml_tensor * Q =
                    ggml_permute(ctx0,
                                 ggml_reshape_3d(ctx0, Qcur, n_state/n_head, n_head, N),
                                 0, 2, 1, 3);

            // K * Q
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, Kcross, Q);

            //struct ggml_tensor * KQ_scaled =
            //    ggml_scale(ctx0,
            //            KQ,
            //            ggml_new_f32(ctx0, 1.0f/sqrt(float(n_state)/n_head))
            //            );

            // no masking for cross-attention
            //struct ggml_tensor * KQ_masked = ggml_diag_mask_inf(ctx0, KQ_scaled, n_past);

            struct ggml_tensor * KQ_soft_max = ggml_soft_max(ctx0, KQ);

            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);

            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

            // cur = KQV_merged.contiguous().view(n_state, N)
            cur = ggml_cpy(ctx0,
                           KQV_merged,
                           ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_state, N));
        }

        // projection
        {
            cur = ggml_mul_mat(ctx0,
                               layer.cross_attn_ln_1_w,
                               cur);

            cur = ggml_add(ctx0,
                           cur,
                           layer.cross_attn_ln_1_b);
        }

        // add the input
        cur = ggml_add(ctx0, cur, inpCA);

        struct ggml_tensor * inpFF = cur;

        // feed-forward network
        {
            // norm
            {
                cur = ggml_norm(ctx0, inpFF, hparams.eps);

                // cur = mlp_ln_w*cur + mlp_ln_b
                cur = ggml_add(ctx0,
                               ggml_mul(ctx0,
                                        cur,
                                        layer.mlp_ln_w),
                               layer.mlp_ln_b);
            }

            // fully connected
            cur = ggml_mul_mat(ctx0,
                               layer.mlp_0_w,
                               cur);

            cur = ggml_add(ctx0,
                           cur,
                           layer.mlp_0_b);

            // GELU activation
            cur = ggml_gelu(ctx0, cur);

            // projection
            cur = ggml_mul_mat(ctx0,
                               layer.mlp_1_w,
                               cur);

            cur = ggml_add(ctx0,
                           cur,
                           layer.mlp_1_b);
        }

        inpL = ggml_add(ctx0, cur, inpFF);
    }

    cur = inpL;

    // norm
    {
        cur = ggml_norm(ctx0, cur, hparams.eps);

        cur = ggml_add(ctx0,
                       ggml_mul(ctx0,
                                cur,
                                model.d_ln_w),
                       model.d_ln_b);
    }

    // compute logits only for the last token
    // comment this line to compute logits for all N tokens
    // might be useful in the future
    cur = ggml_view_2d(ctx0, cur, cur->ne[0], 1, cur->nb[1], (cur->ne[1] - 1)*cur->nb[1]);

    struct ggml_tensor * logits = ggml_mul_mat(ctx0, model.d_te, cur);

    ggml_build_forward_expand(gf, logits);

    ggml_free(ctx0);

    return gf;
}