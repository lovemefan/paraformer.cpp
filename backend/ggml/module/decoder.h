//
// Created by lovemefan on 2023/10/1.
//

#include <vector>

#ifndef PARAFORMER_CPP_DECODER_H
#define PARAFORMER_CPP_DECODER_H





// token decoding layer
struct paraformer_layer_decoder {
    // decoder.blocks.*.attn_ln
    struct ggml_tensor * attn_ln_0_w;
    struct ggml_tensor * attn_ln_0_b;

    // decoder.blocks.*.attn.out
    struct ggml_tensor * attn_ln_1_w;
    struct ggml_tensor * attn_ln_1_b;

    // decoder.blocks.*.attn.query
    struct ggml_tensor * attn_q_w;
    struct ggml_tensor * attn_q_b;

    // decoder.blocks.*.attn.key
    struct ggml_tensor * attn_k_w;

    // decoder.blocks.*.attn.value
    struct ggml_tensor * attn_v_w;
    struct ggml_tensor * attn_v_b;

    // decoder.blocks.*.cross_attn_ln
    struct ggml_tensor * cross_attn_ln_0_w;
    struct ggml_tensor * cross_attn_ln_0_b;

    // decoder.blocks.*.cross_attn.out
    struct ggml_tensor * cross_attn_ln_1_w;
    struct ggml_tensor * cross_attn_ln_1_b;

    // decoder.blocks.*.cross_attn.query
    struct ggml_tensor * cross_attn_q_w;
    struct ggml_tensor * cross_attn_q_b;

    // decoder.blocks.*.cross_attn.key
    struct ggml_tensor * cross_attn_k_w;

    // decoder.blocks.*.cross_attn.value
    struct ggml_tensor * cross_attn_v_w;
    struct ggml_tensor * cross_attn_v_b;

    // decoder.blocks.*.mlp_ln
    struct ggml_tensor * mlp_ln_w;
    struct ggml_tensor * mlp_ln_b;

    // decoder.blocks.*.mlp.0
    struct ggml_tensor * mlp_0_w;
    struct ggml_tensor * mlp_0_b;

    // decoder.blocks.*.mlp.2
    struct ggml_tensor * mlp_1_w;
    struct ggml_tensor * mlp_1_b;
};

struct paraformer_kv_cache {
    struct ggml_tensor * k;
    struct ggml_tensor * v;

    struct ggml_context * ctx;

    // buf points to the memory allocated for both ggml_tensor 'k' and 'v' (see kv_cache_init)
    std::vector<uint8_t> buf;

    int n; // number of tokens currently in the cache
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
    // each decoder keeps its own KV-cache
    paraformer_kv_cache kv_self;

    // the currently generated sequence of tokens
    paraformer_sequence sequence;

    int seek_delta; // the window shift found so far based on the decoded timestamp tokens

    bool failed;    // has the current segment failed to decode?
    bool completed; // has the decoder completed the current segment?
    bool has_ts;    // have we already sampled a non-beg timestamp token for the current segment?

    // new token probs, logits and logprobs after the last paraformer_decode (1-dimensional array: [n_vocab])
    std::vector<float> probs;
    std::vector<float> logits;
    std::vector<float> logprobs;

    std::vector<paraformer_token> tokens_tmp; // used for paraformer_decode calls
};

#endif //PARAFORMER_CPP_DECODER_H