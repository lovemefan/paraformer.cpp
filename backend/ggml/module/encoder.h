//
// Created by lovemefan on 2023/10/1.
//

#ifndef PARAFORMER_CPP_ENCODER_H
#define PARAFORMER_CPP_ENCODER_H

#include <vector>
#include "hparams.h"

// audio encoding layer
struct paraformer_layer_encoder {
    // encoder_attn.linear_out.weight
    struct ggml_tensor * attn_ln_out_w;
    struct ggml_tensor * attn_ln_out_b;

    // encoder.self_attn.linear_q_k_v.weight
    struct ggml_tensor * attn_ln_qkv_w;
    struct ggml_tensor * attn_ln_qkv_b;

    // encoder.self_attn.fsmn_block.weight
    struct ggml_tensor * attn_fsmn_w;


    // encoder.feed_forward.w_1.weight
    struct ggml_tensor * mlp_w1;
    struct ggml_tensor * mlp_b1;

    // encoder.feed_forward.w_2.weight
    struct ggml_tensor * mlp_w2;
    struct ggml_tensor * mlp_b2;

    // encoder.norm1.weight
    struct ggml_tensor * norm_w1;
    struct ggml_tensor * norm_b1;

    // encoder.norm2.weight
    struct ggml_tensor * norm_w2;
    struct ggml_tensor * norm_b2;

};


struct paraformer_encoder {
    std::vector<paraformer_layer_encoder> encoder_layer;
};

void build_encoder(paraformer_encoder &model);
static struct ggml_cgraph * paraformer_build_graph_encoder(
        paraformer_context & wctx,
        paraformer_state & wstate);

#endif //PARAFORMER_CPP_ENCODER_H