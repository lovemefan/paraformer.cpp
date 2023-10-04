//
// Created by lovemefan on 2023/10/1.
//

#include <vector>

#ifndef PARAFORMER_CPP_DECODER_H
#define PARAFORMER_CPP_DECODER_H

#include "hparams.h"
// token decoding layer
struct paraformer_layer_decoder {

    // decoder.embed.0.weight
    struct ggml_tensor * d_emb;
    // decoder.after_norm.weight
    struct ggml_tensor * d_norm_w;
    struct ggml_tensor * d_norm_b;

    // decoder.output_layer.weight
    struct ggml_tensor * d_ln_out_w;
    struct ggml_tensor * d_ln_out_b;

    // decoder.self_attn.fsmn_block.weight
    struct ggml_tensor * d_attn_fsmn_w;
    struct ggml_tensor * d_attn_fsmn_b;

    // decoder.src_attn.linear_q.weight
    struct ggml_tensor * d_src_attn_ln_q_w;
    struct ggml_tensor * d_src_attn_ln_q_b;

    // decoder.src_attn.linear_k_v.weight
    struct ggml_tensor * d_src_attn_ln_kv_w;
    struct ggml_tensor * d_src_attn_ln_kv_b;

    // decoder.src_attn.linear_out.weight
    struct ggml_tensor * d_src_attn_ln_o_w;
    struct ggml_tensor * d_src_attn_ln_o_b;


    // decoder.feed_forward.w_1.weight
    struct ggml_tensor * d_mlp_ln_w1;
    struct ggml_tensor * d_mlp_ln_b1;

    // decoder.feed_forward.w_2.weight
    struct ggml_tensor * d_mlp_ln_w2;

    // decoder.feed_forward.norm.weight
    struct ggml_tensor * d_mlp_norm_w;
    struct ggml_tensor * d_mlp_norm_b;

    //decoder.norm1.weight
    struct ggml_tensor * d_norm_w1;
    struct ggml_tensor * d_norm_b1;

    //decoder.norm2.weight
    struct ggml_tensor * d_norm_w2;
    struct ggml_tensor * d_norm_b2;

    //decoder.norm3.weight
    struct ggml_tensor * d_norm_w3;
    struct ggml_tensor * d_norm_b3;
};


typedef int16_t paraformer_token;
typedef struct paraformer_token_data {
    paraformer_token id;  // token id
    paraformer_token tid; // forced timestamp token id

    float p;           // probability of the token
    float plog;        // log probability of the token
    float pt;          // probability of the timestamp token
    float ptsum;       // sum of probabilities of all timestamp tokens

    // token-level timestamp data
    // do not use if you haven't computed token-level timestamps
    int64_t t0;        // start time of the token
    int64_t t1;        //   end time of the token

    float vlen;        // voice length of the token
} paraformer_token_data;

struct paraformer_sequence {
    std::vector<paraformer_token_data> tokens;

    // the accumulated transcription in the current iteration (used to truncate the tokens array)
    int result_len;

    double sum_logprobs_all; // the sum of the log probabilities of the tokens
    double sum_logprobs;     // the sum of the log probabilities of the tokens (first result_len tokens)
    double avg_logprobs;     // the average log probability of the tokens
    double entropy;          // the entropy of the tokens
    double score;            // likelihood rank score
};

// TAGS: PARAFORMER_DECODER_INIT
struct paraformer_decoder {

    std::vector<paraformer_layer_decoder> decoder_layers;
    //-------decoder.decoder3.bias_decoder------

    // decoder.decoder3.feed_forward.w_1.weight
    struct ggml_tensor * d3_mlp_ln_w1;
    struct ggml_tensor * d3_mlp_ln_b1;

    // decoder.decoder3.feed_forward.w_2.weight
    struct ggml_tensor * d3_mlp_ln_w2;

    // decoder.decoder3.feed_forward.norm.weight
    struct ggml_tensor * d3_mlp_norm_w;
    struct ggml_tensor * d3_mlp_norm_b;

    // decoder.decoder3.norm1.weight
    struct ggml_tensor * d3_norm_w1;
    struct ggml_tensor * d3_norm_b1;

    //-------decoder.bias_decoder------

    // decoder.bias_decoder.src_attn.linear_q.weight
    struct ggml_tensor * d_bias_src_attn_ln_q_w;
    struct ggml_tensor * d_bias_src_attn_ln_q_b;

    // decoder.bias_decoder.src_attn.linear_k_v.weight
    struct ggml_tensor * d_bias_src_attn_ln_kv_w;
    struct ggml_tensor * d_bias_src_attn_ln_kv_b;

    // decoder.bias_decoder.src_attn.linear_out.weight
    struct ggml_tensor * d_bias_src_attn_ln_o_w;
    struct ggml_tensor * d_bias_src_attn_ln_o_b;

    // decoder.bias_decoder.norm3.weight
    struct ggml_tensor * d_bias_norm_w3;
    struct ggml_tensor * d_bias_norm_b3;

    // decoder.bias_output.weight
    struct ggml_tensor * d_bias_ln_o_w;
    struct ggml_tensor * d_bias_ln_o_b;

    //-------last_decoder.bias_decoder------

    // decoder.last_decoder.self_attn.fsmn_block.weight
    struct ggml_tensor * d_last_attn_fsmn_w;
    struct ggml_tensor * d_last_attn_fsmn_b;

    // decoder.last_decoder.src_attn.linear_q.weight
    struct ggml_tensor * d_last_src_attn_ln_q_w;
    struct ggml_tensor * d_last_src_attn_ln_q_b;

    // decoder.last_decoder.src_attn.linear_k_v.weight
    struct ggml_tensor * d_last_src_attn_ln_kv_w;
    struct ggml_tensor * d_last_src_attn_ln_kv_b;

    // decoder.last_decoder.src_attn.linear_out.weight
    struct ggml_tensor * d_last_src_attn_ln_o_w;
    struct ggml_tensor * d_last_src_attn_ln_o_b;

    // decoder.last_decoder.feed_forward.w_1.weight
    struct ggml_tensor * d_last_mlp_ln_w1;
    struct ggml_tensor * d_last_mlp_ln_b1;

    // decoder.last_decoder.feed_forward.w_2.weight
    struct ggml_tensor * d_last_mlp_ln_w2;

    // decoder.last_decoder.feed_forward.norm.weight
    struct ggml_tensor * d_last_mlp_norm_w;
    struct ggml_tensor * d_last_mlp_norm_b;

    // decoder.last_decoder.norm1.weight
    struct ggml_tensor * d_last_norm_w1;
    struct ggml_tensor * d_last_norm_b1;

    //decoder.last_decoder.norm2.weight
    struct ggml_tensor * d_last_norm_w2;
    struct ggml_tensor * d_last_norm_b2;

    //decoder.last_decoder.norm3.weight
    struct ggml_tensor * d_last_norm_w3;
    struct ggml_tensor * d_last_norm_b3;


};


typedef int16_t paraformer_token;

static struct ggml_cgraph * paraformer_build_graph_cross(
        paraformer_context & wctx,
        paraformer_state & wstate);

static struct ggml_cgraph * paraformer_build_graph_decoder(
        paraformer_context & wctx,
        paraformer_state   & wstate,
        paraformer_decoder & decoder,
        const paraformer_token * tokens,
        int   n_tokens,
        int   n_past);

#endif //PARAFORMER_CPP_DECODER_H