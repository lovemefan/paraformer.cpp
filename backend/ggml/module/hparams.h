//
// Created by lovemefan on 2023/10/4.
//

#ifndef PARAFORMER_CPP_HPARAMS_H
#define PARAFORMER_CPP_HPARAMS_H

#include "ggml.h"
#include "ggml-alloc.h"
#include <vector>
#include <string>
#include <random>
#include <map>

#ifdef PARAFORMER_SHARED
#    ifdef _WIN32
#        ifdef PARAFORMER_BUILD
#            define PARAFORMER_API __declspec(dllexport)
#        else
#            define PARAFORMER_API __declspec(dllimport)
#        endif
#    else
#        define PARAFORMER_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define PARAFORMER_API
#endif

#define PARAFORMER_SAMPLE_RATE 16000
#define PARAFORMER_N_FFT       400
#define PARAFORMER_N_MEL       80
#define PARAFORMER_HOP_LENGTH  160
#define PARAFORMER_CHUNK_SIZE  30

// Control logging output; default behavior is to print to stderr
static void log(const char * fmt, ...);
typedef void (*paraformer_log_callback)(const char * line);
static void paraformer_default_log(const char * text) {
    fprintf(stderr, "%s", text);
}
static paraformer_log_callback paraformer_log = paraformer_default_log;

#ifdef __GNUC__
#ifdef __MINGW32__
__attribute__((gnu_format(printf, 1, 2)))
#else
__attribute__((format(printf, 1, 2)))
#endif
#endif
static void log(const char * fmt, ...) {
    if (!paraformer_log) return;
    char buf[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    paraformer_log(buf);
}


// available paraformer models
enum e_model {
    MODEL_ONLINE,
    MODEL_OFFLINE,
    MODEL_CONTEXTUAL_OFFLINE
};


static const size_t MB = 1ull*1024*1024;

// TODO: avoid using GGUF
static const std::map<ggml_type, std::map<e_model, size_t>> MEM_REQ_MODEL = {
        { GGML_TYPE_F32,
                {
                        { MODEL_ONLINE,     0ull*MB },
                        { MODEL_OFFLINE,    0ull*MB },
                        { MODEL_CONTEXTUAL_OFFLINE,   228ull*4*MB },
                },
        },
        { GGML_TYPE_F16,
                {
                        { MODEL_ONLINE,     0ull*MB },
                        { MODEL_OFFLINE,    0ull*MB },
                        { MODEL_CONTEXTUAL_OFFLINE,   228ull*2*MB },
                },
        },
        { GGML_TYPE_Q4_0,
                {
                        { MODEL_ONLINE,     0ull*MB },
                        { MODEL_OFFLINE,    0ull*MB },
                        { MODEL_CONTEXTUAL_OFFLINE,   114ull*MB },
                },
        },

        { GGML_TYPE_I8,
                {
                        { MODEL_ONLINE,     0ull*MB },
                        { MODEL_OFFLINE,    0ull*MB },
                        { MODEL_CONTEXTUAL_OFFLINE,   228ull*MB },
                },
        },
};


struct paraformer_mel {
    int n_len;
    int n_len_org;
    int n_mel;

    std::vector<float> data;
};

struct paraformer_filters {
    int32_t n_mel;
    int32_t n_fft;

    std::vector<float> data;
};

struct paraformer_vocab {
    using id    = int16_t;
    using token = std::string;

    int n_vocab = 8404;

    std::map<token, id> token_to_id;
    std::map<id, token> id_to_token;

    // other special tokens
    id token_blank       = 0;
    id token_sot        = 1;
    id token_eot        = 2;
    id token_unk       = 8403;
    token unk_symbol = "unk";
};


struct paraformer_hparams {
    int16_t n_vocab       = 8404; // number of vocab
    int16_t n_max_audio_length   = 20000; //
    int16_t n_encoder_hidden_state = 512; // dim of hidden state
    int16_t n_encoder_linear_units = 2048;
    int8_t n_encoder_attention_heads  = 4; // head of self attention
    int8_t n_encoder_layers = 50; // num block of encoder
    int16_t n_encoder_0_norm_size = 560;

    int16_t n_decoder_hidden_state = 512;
    int16_t n_decoder_linear_units = 2048;
    int8_t n_decoder_attention_heads =  4;
    int8_t n_decoder_layers = 16;
    int8_t fsmn_kernel_size = 11;

    int16_t n_predictor_dim = 512;
    float predictor_tail_threshold = 0.45;

    int8_t n_mels   = 80; // dim of mels
    std::string window = "hamming";
    int8_t frame_length = 25;
    int8_t frame_shift = 10;
    int8_t lfr_m = 7;
    int8_t lfr_n = 6;
    int8_t ftype         = 1;
    float   eps           = 1e-5f;
    e_model model_type = e_model::MODEL_CONTEXTUAL_OFFLINE;
};






// replace std::pair by using customized pair struct (reason: std::pair is very slow)
template<typename A, typename B>
struct paraformer_pair {
    A first;
    B second;

    // Define a constructor that takes two arguments.
    paraformer_pair(const A& a, const B& b) : first(a), second(b) {}
    // Define a constructor that takes no argument.
    paraformer_pair() : first(A()), second(B()) {}
};

// beam-search helpers
struct kv_buf {
    std::vector<uint8_t> k;
    std::vector<uint8_t> v;
};

struct paraformer_kv_cache {
    struct ggml_tensor * k;
    struct ggml_tensor * v;

    struct ggml_context * ctx;

    // buf points to the memory allocated for both ggml_tensor 'k' and 'v' (see kv_cache_init)
    std::vector<uint8_t> buf;

    int n; // number of tokens currently in the cache
};

// ggml_allocr wrapper for paraformer usage
struct paraformer_allocr {
    ggml_allocr * alloc = nullptr;

    std::vector<uint8_t> meta;
    std::vector<uint8_t> data;
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

struct paraformer_segment {
    int64_t t0;
    int64_t t1;
    std::string text;
    std::vector<paraformer_token_data> tokens;
    bool speaker_turn_next;
};

//#define PARAFORMER_USE_FLASH_ATTN
//#define PARAFORMER_USE_FLASH_FF
#define PARAFORMER_MAX_DECODERS 16

struct paraformer_state {
    int64_t t_sample_us = 0;
    int64_t t_encode_us = 0;
    int64_t t_decode_us = 0;
    int64_t t_prompt_us = 0;
    int64_t t_mel_us = 0;

    int32_t n_sample = 0; // number of tokens sampled
    int32_t n_encode = 0; // number of encoder calls
    int32_t n_decode = 0; // number of decoder calls with n_tokens == 1 (text-generation)
    int32_t n_prompt = 0; // number of decoder calls with n_tokens >  1 (prompt encoding)
    int32_t n_fail_p = 0; // number of logprob threshold failures
    int32_t n_fail_h = 0; // number of entropy threshold failures

    // cross-attention KV cache for the decoders
    // shared between all decoders
    paraformer_kv_cache kv_cross;
    paraformer_mel mel;

    // buffer for swapping KV caches between decoders during beam-search
    std::vector<kv_buf> kv_swap_bufs;

    // reusable buffer for `struct ggml_graph_plan.work_data`
    std::vector<uint8_t> work_buffer;

    // ggml-alloc:
    // - stores meta info about the intermediate tensors into the `meta` buffers
    // - stores the actual tensor data into the `data` buffers
    paraformer_allocr alloc_conv;
    paraformer_allocr alloc_encode;
    paraformer_allocr alloc_cross;
    paraformer_allocr alloc_decode;

    // result of the encoder
    struct ggml_tensor * embd_conv = nullptr;
    struct ggml_tensor * embd_enc  = nullptr;

    // decode output (2-dimensional array: [n_tokens][n_vocab])
    std::vector<float> logits;

    std::vector<paraformer_segment> result_all;
    std::vector<paraformer_token>   prompt_past;

    // work container used to avoid memory allocations
    std::vector<paraformer_pair<double, paraformer_vocab::id>> logits_id;

    mutable std::mt19937 rng; // used for sampling at t > 0.0

    int lang_id = 0; // english by default

    std::string path_model; // populated by paraformer_init_from_file()
#ifdef PARAFORMER_USE_COREML
    paraformer_coreml_context * ctx_coreml = nullptr;
#endif

#ifdef GGML_USE_METAL
    ggml_metal_context * ctx_metal = nullptr;
#endif


    // [EXPERIMENTAL] token-level timestamps data
    int64_t t_beg = 0;
    int64_t t_last = 0;
    paraformer_token tid_last;
    std::vector<float> energy; // PCM signal energy

    // [EXPERIMENTAL] speed-up techniques
    int32_t exp_n_audio_ctx = 0; // 0 - use default
};

struct paraformer_context {
    int64_t t_load_us  = 0;
    int64_t t_start_us = 0;

    ggml_type wtype = ggml_type::GGML_TYPE_F16; // weight type (FP32 / FP16 / QX)
    ggml_type itype = ggml_type::GGML_TYPE_F16; // intermediate type (FP32 or FP16)

    paraformer_model model;
    paraformer_vocab vocab;
    paraformer_state * state = nullptr;

    std::string path_model; // populated by paraformer_init_from_file()
};



#endif //PARAFORMER_CPP_HPARAMS_H
