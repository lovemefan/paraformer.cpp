//
// Created by lovemefan on 2023/10/1.
//

#include "encoder.h"

void build_encoder(paraformer_encoder &model) {
    model.layers_encoder.resize(n_audio_layer);
    model.layers_decoder.resize(n_text_layer);

    // encoder
    {
        model.e_pe       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_audio_state, n_audio_ctx);

        model.e_conv_1_w = ggml_new_tensor_3d(ctx, vtype,         3, n_mels, n_audio_state);
        model.e_conv_1_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n_audio_state);

        model.e_conv_2_w = ggml_new_tensor_3d(ctx, vtype,         3, n_audio_state, n_audio_state);
        model.e_conv_2_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n_audio_state);

        model.e_ln_w     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state);
        model.e_ln_b     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state);

        // map by name
        model.tensors["encoder.positional_embedding"] = model.e_pe;

        model.tensors["encoder.conv1.weight"]         = model.e_conv_1_w;
        model.tensors["encoder.conv1.bias"]           = model.e_conv_1_b;

        model.tensors["encoder.conv2.weight"]         = model.e_conv_2_w;
        model.tensors["encoder.conv2.bias"]           = model.e_conv_2_b;

        model.tensors["encoder.ln_post.weight"]       = model.e_ln_w;
        model.tensors["encoder.ln_post.bias"]         = model.e_ln_b;

        for (int i = 0; i < n_audio_layer; ++i) {
            auto & layer = model.layers_encoder[i];

            layer.mlp_ln_w    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_audio_state);
            layer.mlp_ln_b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_audio_state);

            layer.mlp_0_w     = ggml_new_tensor_2d(ctx, wtype,           n_audio_state, 4*n_audio_state);
            layer.mlp_0_b     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*n_audio_state);

            layer.mlp_1_w     = ggml_new_tensor_2d(ctx, wtype,         4*n_audio_state, n_audio_state);
            layer.mlp_1_b     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_audio_state);

            layer.attn_ln_0_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_audio_state);
            layer.attn_ln_0_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_audio_state);

            layer.attn_q_w    = ggml_new_tensor_2d(ctx, wtype,           n_audio_state, n_audio_state);
            layer.attn_q_b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_audio_state);

            layer.attn_k_w    = ggml_new_tensor_2d(ctx, wtype,           n_audio_state, n_audio_state);

            layer.attn_v_w    = ggml_new_tensor_2d(ctx, wtype,           n_audio_state, n_audio_state);
            layer.attn_v_b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_audio_state);

            layer.attn_ln_1_w = ggml_new_tensor_2d(ctx, wtype,           n_audio_state, n_audio_state);
            layer.attn_ln_1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_audio_state);

            // map by name
            model.tensors["encoder.blocks." + std::to_string(i) + ".mlp_ln.weight"]     = layer.mlp_ln_w;
            model.tensors["encoder.blocks." + std::to_string(i) + ".mlp_ln.bias"]       = layer.mlp_ln_b;

            model.tensors["encoder.blocks." + std::to_string(i) + ".mlp.0.weight"]      = layer.mlp_0_w;
            model.tensors["encoder.blocks." + std::to_string(i) + ".mlp.0.bias"]        = layer.mlp_0_b;

            model.tensors["encoder.blocks." + std::to_string(i) + ".mlp.2.weight"]      = layer.mlp_1_w;
            model.tensors["encoder.blocks." + std::to_string(i) + ".mlp.2.bias"]        = layer.mlp_1_b;

            model.tensors["encoder.blocks." + std::to_string(i) + ".attn_ln.weight"]    = layer.attn_ln_0_w;
            model.tensors["encoder.blocks." + std::to_string(i) + ".attn_ln.bias"]      = layer.attn_ln_0_b;

            model.tensors["encoder.blocks." + std::to_string(i) + ".attn.query.weight"] = layer.attn_q_w;
            model.tensors["encoder.blocks." + std::to_string(i) + ".attn.query.bias"]   = layer.attn_q_b;

            model.tensors["encoder.blocks." + std::to_string(i) + ".attn.key.weight"]   = layer.attn_k_w;

            model.tensors["encoder.blocks." + std::to_string(i) + ".attn.value.weight"] = layer.attn_v_w;
            model.tensors["encoder.blocks." + std::to_string(i) + ".attn.value.bias"]   = layer.attn_v_b;

            model.tensors["encoder.blocks." + std::to_string(i) + ".attn.out.weight"]   = layer.attn_ln_1_w;
            model.tensors["encoder.blocks." + std::to_string(i) + ".attn.out.bias"]     = layer.attn_ln_1_b;
        }
    }
}


static struct ggml_cgraph * paraformer_build_graph_encoder(
        paraformer_context & wctx,
        paraformer_state & wstate) {
    const auto & model   = wctx.model;
    const auto & hparams = model.hparams;

    const int n_ctx   = wstate.exp_n_audio_ctx > 0 ? wstate.exp_n_audio_ctx : hparams.n_audio_ctx;
    const int n_state = hparams.n_audio_state;
    const int n_head  = hparams.n_audio_head;
    const int n_layer = hparams.n_audio_layer;

    struct ggml_init_params params = {
            /*.mem_size   =*/ wstate.alloc_encode.meta.size(),
            /*.mem_buffer =*/ wstate.alloc_encode.meta.data(),
            /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);

    ggml_cgraph * gf = ggml_new_graph(ctx0);

    ggml_allocr * alloc = wstate.alloc_encode.alloc;

    struct ggml_tensor * KQscale = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    ggml_allocr_alloc(alloc, KQscale);

    if (!ggml_allocr_is_measure(alloc)) {
        ggml_set_f32(KQscale, 1.0f/sqrt(float(n_state)/n_head));
    }

    struct ggml_tensor * cur = ggml_view_tensor(ctx0, wstate.embd_conv);

    // ===================================================================
    // NOTE: experimenting with partial evaluation of the encoder (ignore)
    //static int iter = -1;
    //const int n_iter = 1500/n_ctx;

    //iter = (iter + 1) % n_iter;

    //if (iter == 0) {
    //    memset(model.memory_cross_k->data, 0, ggml_nbytes(model.memory_cross_k));
    //    memset(model.memory_cross_v->data, 0, ggml_nbytes(model.memory_cross_v));
    //}

    static int iter = 0;

    const size_t e_pe_stride = model.e_pe->ne[0]*ggml_element_size(model.e_pe);
    const size_t e_pe_offset = model.e_pe->ne[0]*ggml_element_size(model.e_pe)*n_ctx*iter;

    struct ggml_tensor * e_pe = ggml_view_2d(ctx0, model.e_pe, model.e_pe->ne[0], n_ctx, e_pe_stride, e_pe_offset);

    cur = ggml_add(ctx0, e_pe, ggml_cont(ctx0, ggml_transpose(ctx0, cur)));

    // ===================================================================

    // original:
    //cur = ggml_add(ctx0, model.e_pe, ggml_transpose(ctx0, cur));

    struct ggml_tensor * inpL = cur;

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers_encoder[il];

        // norm
        {
            cur = ggml_norm(ctx0, inpL, hparams.eps);

            // cur = ln_0_w*cur + ln_0_b
            cur = ggml_add(ctx0,
                           ggml_mul(ctx0, cur, layer.attn_ln_0_w),
                           layer.attn_ln_0_b);
        }

        // self-attention
        {
            struct ggml_tensor * Qcur = ggml_mul_mat(ctx0,
                                                     layer.attn_q_w,
                                                     cur);

            Qcur = ggml_add(ctx0, Qcur, layer.attn_q_b);

            //Qcur = ggml_scale(ctx0, Qcur, ggml_new_f32(ctx0, pow(float(n_state)/n_head, -0.25)));

            // note: no bias for Key
            struct ggml_tensor * Kcur = ggml_mul_mat(ctx0,
                                                     layer.attn_k_w,
                                                     cur);

            //Kcur = ggml_scale(ctx0, Kcur, ggml_new_f32(ctx0, pow(float(n_state)/n_head, -0.25)));

            struct ggml_tensor * Vcur = ggml_mul_mat(ctx0,
                                                     layer.attn_v_w,
                                                     cur);

            Vcur = ggml_add(ctx0, Vcur, layer.attn_v_b);

            // ------

#ifdef PARAFORMER_USE_FLASH_ATTN
            struct ggml_tensor * Q =
                ggml_permute(ctx0,
                        ggml_cpy(ctx0,
                            Qcur,
                            ggml_new_tensor_3d(ctx0, wctx.itype, n_state/n_head, n_head, n_ctx)),
                        0, 2, 1, 3);

            struct ggml_tensor * K =
                ggml_permute(ctx0,
                        ggml_cpy(ctx0,
                            Kcur,
                            ggml_new_tensor_3d(ctx0, wctx.itype, n_state/n_head, n_head, n_ctx)),
                        0, 2, 1, 3);

            struct ggml_tensor * V =
                ggml_cpy(ctx0,
                        ggml_permute(ctx0,
                            ggml_reshape_3d(ctx0,
                                Vcur,
                                n_state/n_head, n_head, n_ctx),
                            1, 2, 0, 3),
                        ggml_new_tensor_3d(ctx0, wctx.itype, n_ctx, n_state/n_head, n_head));

            struct ggml_tensor * KQV = ggml_flash_attn(ctx0, Q, K, V, false);
#else
            struct ggml_tensor * Q =
                    ggml_permute(ctx0,
                                 ggml_cpy(ctx0,
                                          Qcur,
                                          ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_state/n_head, n_head, n_ctx)),
                                 0, 2, 1, 3);

            struct ggml_tensor * K =
                    ggml_permute(ctx0,
                                 ggml_cpy(ctx0,
                                          Kcur,
                                          ggml_new_tensor_3d(ctx0, wctx.itype, n_state/n_head, n_head, n_ctx)),
                                 0, 2, 1, 3);

            // K * Q
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);

            struct ggml_tensor * KQ_scaled = ggml_scale(ctx0, KQ, KQscale);

            struct ggml_tensor * KQ_soft_max = ggml_soft_max(ctx0, KQ_scaled);

            struct ggml_tensor * V =
                    ggml_cpy(ctx0,
                             ggml_permute(ctx0,
                                          ggml_reshape_3d(ctx0,
                                                          Vcur,
                                                          n_state/n_head, n_head, n_ctx),
                                          1, 2, 0, 3),
                             ggml_new_tensor_3d(ctx0, wctx.itype, n_ctx, n_state/n_head, n_head)
                    );

            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);
#endif
            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

            cur = ggml_cpy(ctx0,
                           KQV_merged,
                           ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_state, n_ctx));
        }

        // projection
        {
            cur = ggml_mul_mat(ctx0,
                               layer.attn_ln_1_w,
                               cur);

            cur = ggml_add(ctx0, cur, layer.attn_ln_1_b);
        }

        // add the input
        cur = ggml_add(ctx0, cur, inpL);

        struct ggml_tensor * inpFF = cur;

        // feed-forward network
        {
            // norm
            {
                cur = ggml_norm(ctx0, inpFF, hparams.eps);

                // cur = mlp_ln_w*cur + mlp_ln_b
                cur = ggml_add(ctx0,
                               ggml_mul(ctx0, cur, layer.mlp_ln_w),
                               layer.mlp_ln_b);
            }

#ifdef PARAFORMER_USE_FLASH_FF
            cur = ggml_flash_ff(ctx0,
                    ggml_cpy(ctx0, cur, ggml_new_tensor_2d(ctx0, wstate.itype, n_state, n_ctx)),
                    layer.mlp_0_w, layer.mlp_0_b, layer.mlp_1_w, layer.mlp_1_b);
#else
            // fully connected
            cur = ggml_mul_mat(ctx0,
                               layer.mlp_0_w,
                               cur);

            cur = ggml_add(ctx0, cur, layer.mlp_0_b);

            // GELU activation
            cur = ggml_gelu(ctx0, cur);

            // projection
            cur = ggml_mul_mat(ctx0,
                               layer.mlp_1_w,
                               cur);

            cur = ggml_add(ctx0, cur, layer.mlp_1_b);
#endif
        }

        inpL = ggml_add(ctx0, cur, inpFF);
    }

    cur = inpL;

    // norm
    {
        cur = ggml_norm(ctx0, cur, hparams.eps);

        // cur = ln_f_g*cur + ln_f_b
        cur = ggml_add(ctx0,
                       ggml_mul(ctx0, cur, model.e_ln_w),
                       model.e_ln_b);
    }

    ggml_build_forward_expand(gf, cur);

    wstate.embd_enc = cur;

    //ggml_graph_print(gf);

    ////////////////////////////////////////////////////////////////////////////

    //printf("%s: used_mem = %f MB, %f MB, %f MB %f MB %f MB\n", __func__,
    //        ggml_used_mem(ctx0)/1024.0/1024.0,
    //        wstate.get_buf_max_mem(0)/1024.0/1024.0,
    //        wstate.get_buf_max_mem(1)/1024.0/1024.0,
    //        wstate.get_buf_max_mem(2)/1024.0/1024.0,
    //        wstate.get_buf_max_mem(3)/1024.0/1024.0);

    ggml_free(ctx0);

    return gf;
}