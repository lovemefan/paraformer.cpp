//
// Created by lovemefan on 2023/10/1.
//

#ifndef PARAFORMER_CPP_ENCODER_H
#define PARAFORMER_CPP_ENCODER_H

#include <vector>
#include "hparams.h"

struct paraformer_layer_encoder {
    // encoder_attn.linear_out.weight
    struct ggml_tensor * e_attn_ln_out_w;
    struct ggml_tensor * e_attn_ln_out_b;

    // encoder.self_attn.linear_q_k_v.weight
    struct ggml_tensor * e_attn_ln_qkv_w;
    struct ggml_tensor * e_attn_ln_qkv_b;

    // encoder.self_attn.fsmn_block.weight
    struct ggml_tensor * e_attn_fsmn_w;


    // encoder.feed_forward.w_1.weight
    struct ggml_tensor * e_mlp_w1;
    struct ggml_tensor * e_mlp_b1;

    // encoder.feed_forward.w_2.weight
    struct ggml_tensor * e_mlp_w2;
    struct ggml_tensor * e_mlp_b2;

    // encoder.norm1.weight
    struct ggml_tensor * e_norm_w1;
    struct ggml_tensor * e_norm_b1;

    // encoder.norm2.weight
    struct ggml_tensor * e_norm_w2;
    struct ggml_tensor * e_norm_b2;

};


struct paraformer_encoder {
    ggml_type wtype = ggml_type::GGML_TYPE_F16; // weight type (FP32 / FP16 / QX)
    ggml_type itype = ggml_type::GGML_TYPE_F16; // intermediate type (FP32 or FP16)

    std::vector<paraformer_layer_encoder> encoder_layer;
    // encoder.after_norm.weight
    struct ggml_tensor * e_after_norm_w;
    struct ggml_tensor * e_after_norm_b;

};

void build_encoder(paraformer_encoder & encoder, ggml_context & ctx, paraformer_hparams & hparams);
//static struct ggml_cgraph * paraformer_build_graph_encoder(
//        paraformer_context & wctx,
//        paraformer_state & wstate);

#endif //PARAFORMER_CPP_ENCODER_H