//
// Created by lovemefan on 2023/11/4.
//

#pragma once
#include <ggml.h>

#include <cstddef>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#ifdef GGML_USE_CUBLAS
#include <ggml-cuda.h>
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#ifdef PARAFORMER_SHARED
#ifdef _WIN32
#ifdef PARAFORMER_BUILD
#define PARAFORMER_API __declspec(dllexport)
#else
#define PARAFORMER_API __declspec(dllimport)
#endif
#else
#define PARAFORMER_API __attribute__((visibility("default")))
#endif
#else
#define PARAFORMER_API
#endif

#define VOCAB_SIZE 8404
#define PARAFORMER_MAX_NODES 4096

// available paraformer models
enum e_model { MODEL_ONLINE, MODEL_OFFLINE, MODEL_CONTEXTUAL_OFFLINE };

struct paraformer_vocab {
    using id = int16_t;
    using token = std::string;
    int n_vocab = 8404;
    std::map<token, id> token_to_id;
    std::map<id, token> id_to_token;
    // other special tokens
    id token_blank = 0;
    id token_sot = 1;
    id token_eot = 2;
    id token_unk = 8403;
    token unk_symbol = "unk";
};

struct paraformer_hparams {
    int n_vocab = VOCAB_SIZE;          // number of vocab
    int n_max_audio_length = 20000;    //
    int n_encoder_hidden_state = 512;  // dim of hidden state
    int n_encoder_linear_units = 2048;
    int n_encoder_attention_heads = 4;  // head of self attention
    int n_encoder_layers = 50;          // num block of encoder
    int n_encoder_0_norm_size = 560;
    int n_decoder_hidden_state = 512;
    int n_decoder_linear_units = 2048;
    int n_decoder_attention_heads = 4;
    int n_decoder_layers = 14;
    int fsmn_kernel_size = 11;

    int n_predictor_dim = 512;
    float predictor_tail_threshold = 0.45;

    int n_mels = 80;  // dim of mels
    std::string window = "hamming";
    int frame_length = 25;
    int frame_shift = 10;
    int lfr_m = 7;
    int lfr_n = 6;
    int ftype = 1;
    float eps = 1e-5f;
    e_model model_type = e_model::MODEL_CONTEXTUAL_OFFLINE;
    int n_audio_ctx;
};

// ############ model structure #############
struct paraformer_bias_encoder {
    // bias encoder is a lstm model

    struct ggml_tensor *bias_embed;

    // bias_encoder.weight_ih_l0
    struct ggml_tensor *be_ih_l_w;
    struct ggml_tensor *be_ih_l_b;

    // bias_encoder.weight_hh_l0
    struct ggml_tensor *be_hh_l_w;
    struct ggml_tensor *be_hh_l_b;
};

struct paraformer_layer_encoder {
    // encoder_attn.linear_out.weight
    struct ggml_tensor *e_attn_ln_out_w;
    struct ggml_tensor *e_attn_ln_out_b;

    // encoder.self_attn.linear_q_k_v.weight
    struct ggml_tensor *e_attn_ln_qkv_w;
    struct ggml_tensor *e_attn_ln_qkv_b;

    // encoder.self_attn.fsmn_block.weight
    struct ggml_tensor *e_attn_fsmn_w;

    // encoder.feed_forward.w_1.weight
    struct ggml_tensor *e_mlp_w1;
    struct ggml_tensor *e_mlp_b1;

    // encoder.feed_forward.w_2.weight
    struct ggml_tensor *e_mlp_w2;
    struct ggml_tensor *e_mlp_b2;

    // encoder.norm1.weight
    struct ggml_tensor *e_norm_w1;
    struct ggml_tensor *e_norm_b1;

    // encoder.norm2.weight
    struct ggml_tensor *e_norm_w2;
    struct ggml_tensor *e_norm_b2;
};

struct paraformer_encoder {
    ggml_type wtype = ggml_type::GGML_TYPE_F16;  // weight type (FP32 / FP16 / QX)
    ggml_type itype = ggml_type::GGML_TYPE_F16;  // intermediate type (FP32 or FP16)

    std::vector<paraformer_layer_encoder> encoder_layer;
    // encoder.after_norm.weight
    struct ggml_tensor *e_after_norm_w;
    struct ggml_tensor *e_after_norm_b;
};

// token decoding layer
struct paraformer_layer_decoder {
    // decoder.self_attn.fsmn_block.weight
    struct ggml_tensor *d_attn_fsmn_w;

    // decoder.src_attn.linear_q.weight
    struct ggml_tensor *d_src_attn_ln_q_w;
    struct ggml_tensor *d_src_attn_ln_q_b;

    // decoder.src_attn.linear_k_v.weight
    struct ggml_tensor *d_src_attn_ln_kv_w;
    struct ggml_tensor *d_src_attn_ln_kv_b;

    // decoder.src_attn.linear_out.weight
    struct ggml_tensor *d_src_attn_ln_o_w;
    struct ggml_tensor *d_src_attn_ln_o_b;

    // decoder.feed_forward.w_1.weight
    struct ggml_tensor *d_mlp_ln_w1;
    struct ggml_tensor *d_mlp_ln_b1;

    // decoder.feed_forward.w_2.weight
    struct ggml_tensor *d_mlp_ln_w2;

    // decoder.feed_forward.norm.weight
    struct ggml_tensor *d_mlp_norm_w;
    struct ggml_tensor *d_mlp_norm_b;

    // decoder.norm1.weight
    struct ggml_tensor *d_norm_w1;
    struct ggml_tensor *d_norm_b1;

    // decoder.norm2.weight
    struct ggml_tensor *d_norm_w2;
    struct ggml_tensor *d_norm_b2;

    // decoder.norm3.weight
    struct ggml_tensor *d_norm_w3;
    struct ggml_tensor *d_norm_b3;
};

struct paraformer_decoder {
    // decoder embedding
    struct ggml_tensor *embed;

    // decoder after norm
    struct ggml_tensor *d_after_norm_w;
    struct ggml_tensor *d_after_norm_b;

    // decoder .output_layer
    struct ggml_tensor *d_output_w;
    struct ggml_tensor *d_output_b;

    std::vector<paraformer_layer_decoder> decoder_layers;

    //-------decoder.decoder3.bias_decoder------
    // decoder.decoder3.feed_forward.w_1.weight
    struct ggml_tensor *d3_mlp_ln_w1;
    struct ggml_tensor *d3_mlp_ln_b1;

    // decoder.decoder3.feed_forward.w_2.weight
    struct ggml_tensor *d3_mlp_ln_w2;

    // decoder.decoder3.feed_forward.norm.weight
    struct ggml_tensor *d3_mlp_norm_w;
    struct ggml_tensor *d3_mlp_norm_b;

    // decoder.decoder3.norm1.weight
    struct ggml_tensor *d3_norm_w1;
    struct ggml_tensor *d3_norm_b1;

    //-------decoder.bias_decoder------
    // decoder.bias_decoder.src_attn.linear_q.weight
    struct ggml_tensor *d_bias_src_attn_ln_q_w;
    struct ggml_tensor *d_bias_src_attn_ln_q_b;

    // decoder.bias_decoder.src_attn.linear_k_v.weight
    struct ggml_tensor *d_bias_src_attn_ln_kv_w;
    struct ggml_tensor *d_bias_src_attn_ln_kv_b;

    // decoder.bias_decoder.src_attn.linear_out.weight
    struct ggml_tensor *d_bias_src_attn_ln_o_w;
    struct ggml_tensor *d_bias_src_attn_ln_o_b;

    // decoder.bias_decoder.norm3.weight
    struct ggml_tensor *d_bias_norm_w3;
    struct ggml_tensor *d_bias_norm_b3;

    // decoder.bias_output.weight
    struct ggml_tensor *d_bias_ln_o_w;

    //-------last_decoder.bias_decoder------

    // decoder.last_decoder.self_attn.fsmn_block.weight
    struct ggml_tensor *d_last_attn_fsmn_w;

    // decoder.last_decoder.src_attn.linear_q.weight
    struct ggml_tensor *d_last_src_attn_ln_q_w;
    struct ggml_tensor *d_last_src_attn_ln_q_b;

    // decoder.last_decoder.src_attn.linear_k_v.weight
    struct ggml_tensor *d_last_src_attn_ln_kv_w;
    struct ggml_tensor *d_last_src_attn_ln_kv_b;

    // decoder.last_decoder.src_attn.linear_out.weight
    struct ggml_tensor *d_last_src_attn_ln_o_w;
    struct ggml_tensor *d_last_src_attn_ln_o_b;

    // decoder.last_decoder.feed_forward.w_1.weight
    struct ggml_tensor *d_last_mlp_ln_w1;
    struct ggml_tensor *d_last_mlp_ln_b1;

    // decoder.last_decoder.feed_forward.w_2.weight
    struct ggml_tensor *d_last_mlp_ln_w2;

    // decoder.last_decoder.feed_forward.norm.weight
    struct ggml_tensor *d_last_mlp_norm_w;
    struct ggml_tensor *d_last_mlp_norm_b;

    // decoder.last_decoder.norm1.weight
    struct ggml_tensor *d_last_norm_w1;
    struct ggml_tensor *d_last_norm_b1;

    // decoder.last_decoder.norm2.weight
    struct ggml_tensor *d_last_norm_w2;
    struct ggml_tensor *d_last_norm_b2;

    // decoder.last_decoder.norm3.weight
    struct ggml_tensor *d_last_norm_w3;
    struct ggml_tensor *d_last_norm_b3;
};

struct paraformer_predictor {
    // predictor.cif_conv1d.weight
    struct ggml_tensor *cif_conv1d_w;
    struct ggml_tensor *cif_conv1d_b;

    struct ggml_tensor *cif_ln_out_w;
    struct ggml_tensor *cif_ln_out_b;
};

// contextual bias paraformer contains bias, encoder, predict and decoder
// more detail in https://arxiv.org/pdf/2308.03266.pdf
struct paraformer_model {
    e_model type = MODEL_CONTEXTUAL_OFFLINE;
    paraformer_hparams hparams;

    paraformer_bias_encoder bias_encoder;
    paraformer_encoder encoder;
    paraformer_decoder decoder;
    paraformer_predictor predictor;
    // context
    struct ggml_context *ctx;

    // the model memory buffer is read-only and can be shared between processors
    std::vector<uint8_t> *buf;

    // tensors
    int n_loaded;
    std::map<std::string, struct ggml_tensor *> tensors;
};

struct paraformer_context {
    int64_t t_load_ms = 0;
    int64_t t_start_ms = 0;

    ggml_type wtype = ggml_type::GGML_TYPE_F16;  // weight type (FP32 / FP16 / QX)
    ggml_type itype = ggml_type::GGML_TYPE_F16;  // intermediate type (FP32 or FP16)

    paraformer_model model;
    paraformer_vocab vocab;

    struct paraformer_state *state = nullptr;
    std::string path_model;
};

struct paraformer_kv_cache {
    struct ggml_tensor *k;
    struct ggml_tensor *v;

    struct ggml_context *ctx;

    // buf points to the memory allocated for both ggml_tensor 'k' and 'v' (see
    // kv_cache_init)
    std::vector<uint8_t> buf;

    int n;  // number of tokens currently in the cache
};

struct paraformer_model_loader {
    void *context;
    size_t (*read)(void *ctx, void *output, size_t read_size);
    bool (*eof)(void *ctx);
    void (*close)(void *ctx);
};

struct paraformer_feature {
    int n_len;
    int lfr_m = 7;
    int lfr_n = 6;

    std::vector<float> data;
};

// replace std::pair by using customized pair struct (reason: std::pair is very slow)
template <typename A, typename B>
struct paraformer_pair {
    A first;
    B second;

    // Define a constructor that takes two arguments.
    paraformer_pair(const A &a, const B &b) : first(a), second(b) {}
    // Define a constructor that takes no argument.
    paraformer_pair() : first(A()), second(B()) {}
};

// ggml_allocr wrapper for whisper usage
struct paraformer_allocr {
    ggml_allocr *alloc = nullptr;
    std::vector<uint8_t> meta;
    std::vector<uint8_t> data;
};

struct whisper_token_data {
    int id;   // token id
    int tid;  // forced timestamp token id

    float p;      // probability of the token
    float plog;   // log probability of the token
    float pt;     // probability of the timestamp token
    float ptsum;  // sum of probabilities of all timestamp tokens

    // token-level timestamp data
    // do not use if you haven't computed token-level timestamps
    int64_t t0;  // start time of the token
    int64_t t1;  //   end time of the token

    float vlen;  // voice length of the token
};

struct paraformer_segment {
    int64_t t0;
    int64_t t1;
    std::string text;
    std::vector<whisper_token_data> tokens;
    bool speaker_turn_next;
};

struct paraformer_state {
    int64_t t_sample_us = 0;
    int64_t t_encode_us = 0;
    int64_t t_decode_us = 0;
    int64_t t_prompt_us = 0;
    int64_t t_mel_us = 0;

    int32_t n_sample = 0;  // number of tokens sampled
    int32_t n_encode = 0;  // number of encoder calls
    int32_t n_decode = 0;  // number of decoder calls with n_tokens == 1 (text-generation)
    int32_t n_prompt = 0;  // number of decoder calls with n_tokens >  1 (prompt encoding)
    int32_t n_fail_p = 0;  // number of logprob threshold failures
    int32_t n_fail_h = 0;  // number of entropy threshold failures

    // shared between all decoders
    paraformer_feature feature;

    // reusable buffer for `struct ggml_graph_plan.work_data`
    std::vector<uint8_t> work_buffer;

    // ggml-alloc:
    // - stores meta info about the intermediate tensors into the `meta` buffers
    // - stores the actual tensor data into the `data` buffers
    paraformer_allocr alloc_bias_encoder;
    paraformer_allocr alloc_encode;
    paraformer_allocr alloc_predict;
    paraformer_allocr alloc_decode;

    // result of the encoder
    struct ggml_tensor *embd_enc = nullptr;

    // decode output (2-dimensional array: [n_tokens][n_vocab])
    std::vector<float> logits;

    std::vector<paraformer_segment> result_all;
    std::vector<int> prompt_past;

    // work container used to avoid memory allocations
    std::vector<paraformer_pair<double, paraformer_vocab::id>> logits_id;

    int lang_id = 0;  // english by default

    std::string path_model;  // populated by whisper_init_from_file()
#ifdef USE_COREML
    whisper_coreml_context *ctx_coreml = nullptr;
#endif

#ifdef GGML_USE_METAL
    ggml_metal_context *ctx_metal = nullptr;
#endif

    // [EXPERIMENTAL] token-level timestamps data
    int64_t t_beg = 0;
    int64_t t_last = 0;
    int tid_last;
    std::vector<float> energy;  // PCM signal energy

    // [EXPERIMENTAL] speed-up techniques
    int32_t exp_n_audio_ctx = 0;  // 0 - use default
};

bool paraformer_model_load(struct paraformer_model_loader *loader, paraformer_context &wctx);

// Various functions for loading a ggml paraformer model.
// Allocate (almost) all memory needed for the model.
// Return NULL on failure
PARAFORMER_API struct paraformer_context *paraformer_init_from_file(const char *path_model);
PARAFORMER_API struct paraformer_context *paraformer_init_from_buffer(void *buffer, size_t buffer_size);
PARAFORMER_API struct paraformer_context *paraformer_init(struct paraformer_model_loader *loader);
PARAFORMER_API struct paraformer_state *paraformer_init_state(paraformer_context *ctx);
PARAFORMER_API struct ggml_cgraph *paraformer_build_graph_bias_encoder(paraformer_context &wctx,
                                                                       paraformer_state &wstate);
PARAFORMER_API struct ggml_cgraph *paraformer_build_graph_encoder(paraformer_context &wctx, paraformer_state &wstate);
PARAFORMER_API struct ggml_cgraph *paraformer_build_graph_predict(paraformer_context &wctx, paraformer_state &wstate);
PARAFORMER_API struct ggml_cgraph *paraformer_build_graph_decoder(paraformer_context &wctx, paraformer_state &wstate);

// Frees all allocated memory
PARAFORMER_API void paraformer_free(struct paraformer_context *ctx);
PARAFORMER_API void paraformer_free_params(struct paraformer_full_params *params);
