//
// Created by lovemefan on 2024/3/23.
//

#include "paraformer-offline.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#include <assert.h>
#include <math.h>
#include <stdarg.h>

#include <cstddef>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#if defined(_MSC_VER)
#pragma warning(disable : 4244 4267)  // possible loss of data
#endif

#if defined(GGML_BIG_ENDIAN)
#include <bit>

template <typename T>
static T byteswap(T value) {
  return std::byteswap(value);
}

template <>
float byteswap(float value) {
  return std::bit_cast<float>(byteswap(std::bit_cast<std::uint32_t>(value)));
}

template <typename T>
static void byteswap_tensor_data(ggml_tensor *tensor) {
  T *datum = reinterpret_cast<T *>(tensor->data);
  for (int i = 0; i < ggml_nelements(tensor); i++) {
    datum[i] = byteswap(datum[i]);
  }
}

static void byteswap_tensor(ggml_tensor *tensor) {
  switch (tensor->type) {
    case GGML_TYPE_I16: {
      byteswap_tensor_data<int16_t>(tensor);
      break;
    }
    case GGML_TYPE_F16: {
      byteswap_tensor_data<ggml_fp16_t>(tensor);
      break;
    }
    case GGML_TYPE_I32: {
      byteswap_tensor_data<int32_t>(tensor);
      break;
    }
    case GGML_TYPE_F32: {
      byteswap_tensor_data<float>(tensor);
      break;
    }
    default: {  // GML_TYPE_I8
      break;
    }
  }
}

#define BYTESWAP_VALUE(d) d = byteswap(d)
#define BYTESWAP_FILTERS(f)      \
  do {                           \
    for (auto &datum : f.data) { \
      datum = byteswap(datum);   \
    }                            \
  } while (0)
#define BYTESWAP_TENSOR(t) \
  do {                     \
    byteswap_tensor(t);    \
  } while (0)
#else
#define BYTESWAP_VALUE(d) \
  do {                    \
  } while (0)
#define BYTESWAP_FILTERS(f) \
  do {                      \
  } while (0)
#define BYTESWAP_TENSOR(t) \
  do {                     \
  } while (0)
#endif

#ifdef __GNUC__
#ifdef __MINGW32__
#define PARAFORMER_ATTRIBUTE_FORMAT(...) \
  __attribute__((format(gnu_printf, __VA_ARGS__)))
#else
#define PARAFORMER_ATTRIBUTE_FORMAT(...) \
  __attribute__((format(printf, __VA_ARGS__)))
#endif
#else
#define PARAFORMER_ATTRIBUTE_FORMAT(...)
#endif

//
// logging
//

PARAFORMER_ATTRIBUTE_FORMAT(2, 3)
static void paraformer_log_internal(ggml_log_level level, const char *format,
                                    ...);
static void paraformer_log_callback_default(ggml_log_level level,
                                            const char *text, void *user_data);

#define PARAFORMER_LOG_ERROR(...) \
  paraformer_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define PARAFORMER_LOG_WARN(...) \
  paraformer_log_internal(GGML_LOG_LEVEL_WARN, __VA_ARGS__)
#define PARAFORMER_LOG_INFO(...) \
  paraformer_log_internal(GGML_LOG_LEVEL_INFO, __VA_ARGS__)

// define this to enable verbose trace logging - useful for debugging purposes
#define PARAFORMER_DEBUG

#if defined(PARAFORMER_DEBUG)
#define PARAFORMER_LOG_DEBUG(...) \
  paraformer_log_internal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#else
#define PARAFORMER_LOG_DEBUG(...)
#endif

#define PARAFORMER_ASSERT(x)                                           \
  do {                                                                 \
    if (!(x)) {                                                        \
      PARAFORMER_LOG_ERROR("PARAFORMER_ASSERT: %s:%d: %s\n", __FILE__, \
                           __LINE__, #x);                              \
      abort();                                                         \
    }                                                                  \
  } while (0)

template <typename T>
static void read_safe(paraformer_model_loader *loader, T &dest) {
  loader->read(loader->context, &dest, sizeof(T));
  BYTESWAP_VALUE(dest);
}

static const size_t MB = 1ull * 1024 * 1024;

static void paraformer_log_callback_default(ggml_log_level level,
                                            const char *text, void *user_data) {
  (void)level;
  (void)user_data;
  fputs(text, stderr);
  fflush(stderr);
}

struct paraformer_global {
  // We save the log callback globally
  ggml_log_callback log_callback = paraformer_log_callback_default;
  void *log_callback_user_data = nullptr;
};

static paraformer_global g_state;

GGML_ATTRIBUTE_FORMAT(2, 3)
static void paraformer_log_internal(ggml_log_level level, const char *format,
                                    ...) {
  va_list args;
  va_start(args, format);
  char buffer[1024];
  int len = vsnprintf(buffer, 1024, format, args);
  if (len < 1024) {
    g_state.log_callback(level, buffer, g_state.log_callback_user_data);
  } else {
    char *buffer2 = new char[len + 1];
    vsnprintf(buffer2, len + 1, format, args);
    buffer2[len] = 0;
    g_state.log_callback(level, buffer2, g_state.log_callback_user_data);
    delete[] buffer2;
  }
  va_end(args);
}

static const std::map<ggml_type, std::map<e_model, size_t>> MEM_REQ_MODEL = {
    {
        GGML_TYPE_F32,
        {
            {MODEL_ONLINE, 0ull * MB},
            {MODEL_SEACO_OFFLINE, 236ull * 4 * MB},
            {MODEL_CONTEXTUAL_OFFLINE, 228ull * 4 * MB},
        },
    },
    {
        GGML_TYPE_F16,
        {
            {MODEL_ONLINE, 0ull * MB},
            {MODEL_SEACO_OFFLINE, 236ull * 2 * MB},
            {MODEL_CONTEXTUAL_OFFLINE, 228ull * 2 * MB},
        },
    },
    {
        GGML_TYPE_Q4_0,
        {
            {MODEL_ONLINE, 0ull * MB},
            {MODEL_SEACO_OFFLINE, 0ull * MB},
            {MODEL_CONTEXTUAL_OFFLINE, 114ull * MB},
        },
    },

    {
        GGML_TYPE_I8,
        {
            {MODEL_ONLINE, 0ull * MB},
            {MODEL_SEACO_OFFLINE, 0ull * MB},
            {MODEL_CONTEXTUAL_OFFLINE, 228ull * MB},
        },
    },
};

struct paraformer_hparams {
  int n_vocab = 8404;                // number of vocab
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
  e_model model_type = e_model::MODEL_SEACO_OFFLINE;
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
  ggml_type itype =
      ggml_type::GGML_TYPE_F16;  // intermediate type (FP32 or FP16)
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
  e_model type = MODEL_SEACO_OFFLINE;
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

struct paraformer_vocab {
  using id = int32_t;
  using token = std::string;

  int n_vocab = 8404;

  std::map<token, id> token_to_id;
  std::map<id, token> id_to_token;

  id token_eot = 2;
  id token_sot = 1;
};

// replace std::pair by using customized pair struct (reason: std::pair is very
// slow)
template <typename A, typename B>
struct paraformer_pair {
  A first;
  B second;

  // Define a constructor that takes two arguments.
  paraformer_pair(const A &a, const B &b) : first(a), second(b) {}
  // Define a constructor that takes no argument.
  paraformer_pair() : first(A()), second(B()) {}
};

// ggml_allocr wrapper for PARAFORMER usage
struct paraformer_allocr {
  ggml_gallocr_t alloc = nullptr;
  std::vector<uint8_t> meta;
};

struct paraformer_context {
  int64_t t_load_ms = 0;
  int64_t t_start_ms = 0;

  ggml_type wtype = ggml_type::GGML_TYPE_F16;  // weight type (FP32 / FP16 / QX)
  ggml_type itype =
      ggml_type::GGML_TYPE_F16;  // intermediate type (FP32 or FP16)

  paraformer_model model;
  paraformer_vocab vocab;

  paraformer_context_params params;

  struct paraformer_state *state = nullptr;
  ggml_backend_t backend = nullptr;
  std::string path_model;
};

struct paraformer_token_data {
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
  std::vector<paraformer_token_data> tokens;
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
  int32_t n_decode =
      0;  // number of decoder calls with n_tokens == 1 (text-generation)
  int32_t n_prompt =
      0;  // number of decoder calls with n_tokens >  1 (prompt encoding)
  int32_t n_fail_p = 0;  // number of logprob threshold failures
  int32_t n_fail_h = 0;  // number of entropy threshold failures

  // shared between all decoders
  paraformer_feature feature;

  // reusable buffer for `struct ggml_graph_plan.work_data`
  std::vector<uint8_t> work_buffer;

  ggml_backend_t backend = nullptr;

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

  std::string path_model;  // populated by PARAFORMER_init_from_file()
#ifdef USE_COREML
  PARAFORMER_coreml_context *ctx_coreml = nullptr;
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

static size_t paraformer_allocr_size(struct paraformer_allocr &allocr) {
  return allocr.meta.size() + ggml_gallocr_get_buffer_size(allocr.alloc, 0);
}

// measure the memory usage of a graph and prepare the allocr's internal data
// buffer
static bool paraformer_allocr_graph_init(
    struct paraformer_allocr &allocr, ggml_backend_t backend,
    std::function<struct ggml_cgraph *()> &&get_graph) {
  auto &alloc = allocr.alloc;
  auto &meta = allocr.meta;

  alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

  meta.resize(ggml_tensor_overhead() * PARAFORMER_MAX_NODES +
              ggml_graph_overhead());

  // since there are dependencies between the different graphs,
  // we need to allocate them instead of only reserving to get the correct
  // compute buffer size
  if (!ggml_gallocr_alloc_graph(alloc, get_graph())) {
    // failed to allocate the compute buffer
    PARAFORMER_LOG_ERROR("%s: failed to allocate the compute buffer\n",
                         __func__);
    return false;
  }
  return true;
}

struct paraformer_context *paraformer_init(struct gguf_context *g_context) {
  ggml_time_init();

  struct paraformer_context *ctx = new paraformer_context;

  return ctx;
}

struct paraformer_context_params paraformer_context_default_params() {
  struct paraformer_context_params result = {
      /*.use_gpu              =*/true,
      /*.gpu_device           =*/0,
  };
  return result;
}

static ggml_backend_t paraformer_backend_init(
    const paraformer_context_params &params) {
  ggml_backend_t backend_gpu = NULL;

  // initialize the backends
#ifdef GGML_USE_CUDA
  if (params.use_gpu) {
    WHISPER_LOG_INFO("%s: using CUDA backend\n", __func__);
    backend_gpu = ggml_backend_cuda_init(params.gpu_device);
    if (!backend_gpu) {
      WHISPER_LOG_ERROR("%s: ggml_backend_cuda_init() failed\n", __func__);
    }
  }
#endif

#ifdef GGML_USE_METAL
  if (params.use_gpu) {
    WHISPER_LOG_INFO("%s: using Metal backend\n", __func__);
    ggml_backend_metal_log_set_callback(g_state.log_callback,
                                        g_state.log_callback_user_data);
    backend_gpu = ggml_backend_metal_init();
    if (!backend_gpu) {
      WHISPER_LOG_ERROR("%s: ggml_backend_metal_init() failed\n", __func__);
    } else if (!ggml_backend_metal_supports_family(backend_gpu, 7)) {
      WHISPER_LOG_ERROR(
          "%s: Metal GPU does not support family 7 - falling back to CPU\n",
          __func__);
      ggml_backend_free(backend_gpu);
      backend_gpu = NULL;
    }
  }
#endif

#ifdef GGML_USE_SYCL
  if (params.use_gpu) {
    WHISPER_LOG_INFO("%s: using SYCL backend\n", __func__);
    backend_gpu = ggml_backend_sycl_init(params.gpu_device);
    if (!backend_gpu) {
      WHISPER_LOG_ERROR("%s: ggml_backend_sycl_init() failed\n", __func__);
    }
  }
#endif

  if (backend_gpu) {
    return backend_gpu;
  }
  return ggml_backend_cpu_init();
}

void whisper_free_state(struct paraformer_state *state) {
  if (state) {
#ifdef WHISPER_USE_COREML
    if (state->ctx_coreml != nullptr) {
      whisper_coreml_free(state->ctx_coreml);
      state->ctx_coreml = nullptr;
    }
#endif

#ifdef WHISPER_USE_OPENVINO
    if (state->ctx_openvino != nullptr) {
      whisper_openvino_free(state->ctx_openvino);
      state->ctx_openvino = nullptr;
    }
#endif
    ggml_gallocr_free(state->alloc_encode.alloc);
    ggml_gallocr_free(state->alloc_decode.alloc);

    ggml_backend_free(state->backend);

    delete state;
  }
}

struct paraformer_context *paraformer_init_with_params_no_state(
    const char *path_model, paraformer_context_params params) {
  ggml_time_init();

  auto *ctx = new paraformer_context;
  ctx->params = params;

  if (!paraformer_model_load(path_model, *ctx)) {
    PARAFORMER_LOG_ERROR("%s: failed to load model\n", __func__);
    delete ctx;
    return nullptr;
  }

  return ctx;
}

struct paraformer_context *paraformer_init_from_file_with_params(
    const char *path_model, struct paraformer_context_params params) {
  PARAFORMER_LOG_INFO("%s: loading model from '%s'\n", __func__, path_model);

  auto *pctx = paraformer_init_with_params_no_state(path_model, params);
  struct paraformer_context_params backend = {
      /*.use_gpu              =*/params.use_gpu,
      /*.gpu_device           =*/0,
  };
  pctx->backend = paraformer_backend_init(backend);
  paraformer_init_state(pctx);

  if (pctx) {
    pctx->path_model = path_model;
  }

  return pctx;
}

struct paraformer_context *paraformer_init_from_file(const char *path_model) {
  return paraformer_init_from_file_with_params(
      path_model, paraformer_context_default_params());
}

// load the model from a gguf file
// see the convert-pt-to-ggml.py script for details
//
bool paraformer_model_load(const char *path_model, paraformer_context &pctx) {
  struct ggml_context *ctx_data = NULL;

  struct gguf_init_params gguf_params = {
      /*.no_alloc = */ false,
      /*.ctx      = */ &ctx_data,
  };
  struct gguf_context *gguf_ctx = gguf_init_from_file(path_model, gguf_params);

  // kv data
  {
    PARAFORMER_LOG_INFO("%s: version:      %d\n", __func__,
                        gguf_get_version(gguf_ctx));
    PARAFORMER_LOG_INFO("%s: alignment:   %zu\n", __func__,
                        gguf_get_alignment(gguf_ctx));
    PARAFORMER_LOG_INFO("%s: data offset: %zu\n", __func__,
                        gguf_get_data_offset(gguf_ctx));

    const int n_kv = gguf_get_n_kv(gguf_ctx);

    PARAFORMER_LOG_DEBUG("%s: n_kv: %d\n", __func__, n_kv);

    for (int i = 0; i < n_kv; ++i) {
      const char *key = gguf_get_key(gguf_ctx, i);
      PARAFORMER_LOG_DEBUG("%s: %s        \n", __func__, key);
    }
  }

  PARAFORMER_LOG_INFO("%s: loading model\n", __func__);

  const int64_t t_start_ms = ggml_time_ms();

  pctx.t_start_ms = t_start_ms;

  auto &model = pctx.model;
  auto &vocab = pctx.vocab;
  auto &hparams = model.hparams;
  // load hparams
  {
    hparams.n_vocab = gguf_get_val_i32(
        gguf_ctx, gguf_find_key(gguf_ctx, "tokenizer.vocab_size"));
    hparams.n_encoder_hidden_state =
        gguf_get_val_i32(gguf_ctx, gguf_find_key(gguf_ctx, "model.inner_dim"));
    hparams.n_encoder_linear_units = gguf_get_val_i32(
        gguf_ctx, gguf_find_key(gguf_ctx, "encoder.linear_units"));
    hparams.n_encoder_attention_heads = gguf_get_val_i32(
        gguf_ctx, gguf_find_key(gguf_ctx, "encoder.attention_heads"));
    hparams.n_encoder_layers = gguf_get_val_i32(
        gguf_ctx, gguf_find_key(gguf_ctx, "encoder.num_blocks"));
    hparams.n_encoder_0_norm_size = gguf_get_val_i32(
        gguf_ctx, gguf_find_key(gguf_ctx, "encoder.num_blocks"));
    hparams.n_decoder_hidden_state =
        gguf_get_val_i32(gguf_ctx, gguf_find_key(gguf_ctx, "model.inner_dim"));
    hparams.n_decoder_linear_units = gguf_get_val_i32(
        gguf_ctx, gguf_find_key(gguf_ctx, "decoder.linear_units"));
    hparams.n_decoder_attention_heads = gguf_get_val_i32(
        gguf_ctx, gguf_find_key(gguf_ctx, "decoder.attention_heads"));
    hparams.n_decoder_layers = gguf_get_val_i32(
        gguf_ctx, gguf_find_key(gguf_ctx, "decoder.num_blocks"));
    hparams.n_predictor_dim =
        gguf_get_val_i32(gguf_ctx, gguf_find_key(gguf_ctx, "predictor.idim"));
    hparams.fsmn_kernel_size = gguf_get_val_i32(
        gguf_ctx, gguf_find_key(gguf_ctx, "seaco_decoder.kernel_size"));
    hparams.predictor_tail_threshold = gguf_get_val_f32(
        gguf_ctx, gguf_find_key(gguf_ctx, "predictor.tail_threshold"));

    // for the big tensors, we have the option to store the data in 16-bit
    // floats or quantized in order to save memory and also to speed up the
    // computation
    pctx.wtype = ggml_ftype_to_ggml_type((ggml_ftype)(model.hparams.ftype));
    if (pctx.wtype == GGML_TYPE_COUNT) {
      PARAFORMER_LOG_INFO("%s: invalid model (bad ftype value %d)\n", __func__,
                          model.hparams.ftype);
      return false;
    }

    const size_t scale = model.hparams.ftype ? 1 : 2;

    PARAFORMER_LOG_INFO("%s: n_vocab       = %d\n", __func__, hparams.n_vocab);
    PARAFORMER_LOG_INFO("%s: n_encoder_hidden_state   =%d\n", __func__,
                        hparams.n_encoder_hidden_state);
    PARAFORMER_LOG_INFO("%s: n_encoder_linear_units = %d\n", __func__,
                        hparams.n_encoder_linear_units);
    PARAFORMER_LOG_INFO("%s: n_encoder_attention_heads  = %d\n", __func__,
                        hparams.n_encoder_attention_heads);
    PARAFORMER_LOG_INFO("%s: n_encoder_layers = %d\n", __func__,
                        hparams.n_encoder_layers);
    PARAFORMER_LOG_INFO("%s: n_decoder_hidden_state    = %d\n", __func__,
                        hparams.n_decoder_hidden_state);
    PARAFORMER_LOG_INFO("%s: n_decoder_linear_units  = %d\n", __func__,
                        hparams.n_decoder_linear_units);
    PARAFORMER_LOG_INFO("%s: n_decoder_attention_heads   = %d\n", __func__,
                        hparams.n_decoder_attention_heads);
    PARAFORMER_LOG_INFO("%s: n_decoder_layers  = %d\n", __func__,
                        hparams.n_decoder_layers);
    PARAFORMER_LOG_INFO("%s: n_predictor_dim  = %d\n", __func__,
                        hparams.n_predictor_dim);
    PARAFORMER_LOG_INFO("%s: predictor_tail_threshold  = %f\n", __func__,
                        hparams.predictor_tail_threshold);
    PARAFORMER_LOG_INFO("%s: n_mels        = %d\n", __func__, hparams.n_mels);
    PARAFORMER_LOG_INFO("%s: ftype         = %d\n", __func__,
                        model.hparams.ftype);
    PARAFORMER_LOG_INFO("%s: type          = %d\n", __func__, model.type);

    // initialize all memory buffers
    // always have at least one decoder

    pctx.model.buf = new std::vector<uint8_t>();
    pctx.model.buf->resize(scale * MEM_REQ_MODEL.at(pctx.wtype).at(model.type));

    // we skip initialization of the state until it is needed
    // because it might be that state will always be provided externally.
  }

  // load vocab
  {
    std::string word;

    const int token_idx = gguf_find_key(gguf_ctx, "tokenizer.ggml.tokens");
    const int n_vocab = gguf_get_arr_n(gguf_ctx, token_idx);

    if (n_vocab != model.hparams.n_vocab) {
      PARAFORMER_LOG_ERROR(
          "%s: vocabulary loaded from model file error - vaocabulary size is "
          "%d, but got %d .\n",
          __func__, model.hparams.n_vocab, n_vocab);
    }
    std::vector<const char *> tokens;
    tokens.resize(n_vocab);
    for (uint32_t i = 0; i < n_vocab; i++) {
      word = gguf_get_arr_str(gguf_ctx, token_idx, i);
      vocab.token_to_id[word] = i;
      vocab.id_to_token[i] = word;
    }

    PARAFORMER_LOG_INFO("%s: vocab[%d] loaded\n", __func__, n_vocab);
  }

  vocab.n_vocab = model.hparams.n_vocab;

  size_t ctx_size = 0;

  const ggml_type wtype = pctx.wtype;
  const ggml_type vtype =
      pctx.wtype == GGML_TYPE_F32 ? GGML_TYPE_F32 : GGML_TYPE_F16;  // conv

  {
    const auto &hparams = model.hparams;

    const int n_vocab = hparams.n_vocab;
    const int n_encoder_layers = hparams.n_encoder_layers;
    const int n_decoder_layers = hparams.n_decoder_layers;

    model.encoder.encoder_layer.resize(n_encoder_layers);
    model.decoder.decoder_layers.resize(n_decoder_layers);

    // load weights
    {
      const int n_tensors = gguf_get_n_tensors(gguf_ctx);
      PARAFORMER_LOG_INFO("%s: n_tensors: %d\n", __func__, n_tensors);
      model.n_loaded = 0;

      for (int i = 0; i < n_tensors; ++i) {
        const std::string name = gguf_get_tensor_name(gguf_ctx, i);
        struct ggml_tensor *cur = ggml_get_tensor(ctx_data, name.c_str());
        model.tensors[name] = cur;

        auto n_dim = ggml_n_dims(cur);
        std::stringstream shape;
        if (n_dim == 1)
          shape << cur->ne[0];
        else if (n_dim == 2)
          shape << cur->ne[0] << ',' << cur->ne[1];
        else if (n_dim == 3)
          shape << cur->ne[0] << ',' << cur->ne[1] << ',' << cur->ne[2];
        else
          shape << cur->ne[0] << ',' << cur->ne[1] << ',' << cur->ne[2] << ','
                << cur->ne[3];

        PARAFORMER_LOG_DEBUG(
            "%s: tensor[%d]: n_dims = %d, shape = (%s), name = %s, "
            "data = %p\n",
            __func__, i, ggml_n_dims(cur), shape.str().c_str(), cur->name,
            cur->data);
      }
    }
  }

  // add encoders, multi layers of EncoderLayerSANM
  {
    for (int i = 0; i < hparams.n_encoder_layers; ++i) {
      auto &layer = model.encoder.encoder_layer[i];

      // map by name
      layer.e_attn_ln_out_w =
          model.tensors[(i == 0
                             ? ("encoder.encoders0." + std::to_string(i))
                             : ("encoder.encoders." + std::to_string(i - 1))) +
                        ".self_attn.linear_out.weight"];
      layer.e_attn_ln_out_b =
          model.tensors[(i == 0
                             ? ("encoder.encoders0." + std::to_string(i))
                             : ("encoder.encoders." + std::to_string(i - 1))) +
                        ".self_attn.linear_out.bias"];

      layer.e_attn_ln_qkv_w =
          model.tensors[(i == 0
                             ? ("encoder.encoders0." + std::to_string(i))
                             : ("encoder.encoders." + std::to_string(i - 1))) +
                        ".self_attn.linear_q_k_v.weight"];
      layer.e_attn_ln_qkv_b =
          model.tensors[(i == 0
                             ? ("encoder.encoders0." + std::to_string(i))
                             : ("encoder.encoders." + std::to_string(i - 1))) +
                        ".self_attn.linear_q_k_v.bias"];

      layer.e_attn_fsmn_w =
          model.tensors[(i == 0
                             ? ("encoder.encoders0." + std::to_string(i))
                             : ("encoder.encoders." + std::to_string(i - 1))) +
                        ".self_attn.fsmn_block.weight"];

      layer.e_mlp_w1 =
          model.tensors[(i == 0
                             ? ("encoder.encoders0." + std::to_string(i))
                             : ("encoder.encoders." + std::to_string(i - 1))) +
                        ".feed_forward.w_1.weight"];
      layer.e_mlp_b1 =
          model.tensors[(i == 0
                             ? ("encoder.encoders0." + std::to_string(i))
                             : ("encoder.encoders.") + std::to_string(i - 1)) +
                        ".feed_forward.w_1.bias"];

      layer.e_mlp_w2 =
          model.tensors[(i == 0
                             ? ("encoder.encoders0." + std::to_string(i))
                             : ("encoder.encoders.") + std::to_string(i - 1)) +
                        ".feed_forward.w_2.weight"];
      layer.e_mlp_b2 =
          model.tensors[(i == 0
                             ? ("encoder.encoders0." + std::to_string(i))
                             : ("encoder.encoders.") + std::to_string(i - 1)) +
                        ".feed_forward.w_2.bias"];

      layer.e_norm_w1 =
          model.tensors[(i == 0
                             ? ("encoder.encoders0." + std::to_string(i))
                             : ("encoder.encoders.") + std::to_string(i - 1)) +
                        ".norm1.weight"];
      layer.e_norm_b1 =
          model.tensors[(i == 0
                             ? ("encoder.encoders0." + std::to_string(i))
                             : ("encoder.encoders.") + std::to_string(i - 1)) +
                        ".norm1.bias"];

      layer.e_norm_w2 =
          model.tensors[(i == 0
                             ? ("encoder.encoders0." + std::to_string(i))
                             : ("encoder.encoders.") + std::to_string(i - 1)) +
                        ".norm2.weight"];
      layer.e_norm_b2 =
          model.tensors[(i == 0
                             ? ("encoder.encoders0." + std::to_string(i))
                             : ("encoder.encoders.") + std::to_string(i - 1)) +
                        ".norm2.bias"];
    }
    model.encoder.e_after_norm_w = model.tensors["encoder.after_norm.weight"];
    model.encoder.e_after_norm_b = model.tensors["encoder.after_norm.bias"];
  }
  // decoder layers

  {
    model.decoder.embed = model.tensors["decoder.embed.0.weight"];

    // decoder layers
    for (int i = 0; i < hparams.n_decoder_layers; ++i) {
      auto &layer = model.decoder.decoder_layers[i];
      layer.d_attn_fsmn_w =
          model.tensors["decoder.decoders." + std::to_string(i) +
                        ".self_attn.fsmn_block.weight"];

      layer.d_src_attn_ln_q_w =
          model.tensors["decoder.decoders." + std::to_string(i) +
                        ".src_attn.linear_q.weight"];
      layer.d_src_attn_ln_q_b =
          model.tensors["decoder.decoders." + std::to_string(i) +
                        ".src_attn.linear_q.bias"];

      layer.d_src_attn_ln_kv_w =
          model.tensors["decoder.decoders." + std::to_string(i) +
                        ".src_attn.linear_k_v.weight"];
      layer.d_src_attn_ln_kv_b =
          model.tensors["decoder.decoders." + std::to_string(i) +
                        ".src_attn.linear_k_v.bias"];

      layer.d_src_attn_ln_o_w =
          model.tensors["decoder.decoders." + std::to_string(i) +
                        ".src_attn.linear_out.weight"];
      layer.d_src_attn_ln_o_b =
          model.tensors["decoder.decoders." + std::to_string(i) +
                        ".src_attn.linear_out.bias"];

      layer.d_mlp_ln_w1 =
          model.tensors["decoder.decoders." + std::to_string(i) +
                        ".feed_forward.w_1.weight"];
      layer.d_mlp_ln_b1 =
          model.tensors["decoder.decoders." + std::to_string(i) +
                        ".feed_forward.w_1.bias"];

      layer.d_mlp_ln_w2 =
          model.tensors["decoder.decoders." + std::to_string(i) +
                        ".feed_forward.w_2.weight"];

      layer.d_mlp_norm_w =
          model.tensors["decoder.decoders." + std::to_string(i) +
                        ".feed_forward.norm.weight"];
      layer.d_mlp_norm_b =
          model.tensors["decoder.decoders." + std::to_string(i) +
                        ".feed_forward.norm.bias"];

      layer.d_norm_w1 = model.tensors["decoder.decoders." + std::to_string(i) +
                                      ".norm1.weight"];
      layer.d_norm_b1 =
          model
              .tensors["decoder.decoders." + std::to_string(i) + ".norm1.bias"];

      layer.d_norm_w2 = model.tensors["decoder.decoders." + std::to_string(i) +
                                      ".norm2.weight"];
      layer.d_norm_b2 =
          model
              .tensors["decoder.decoders." + std::to_string(i) + ".norm2.bias"];

      layer.d_norm_w3 = model.tensors["decoder.decoders." + std::to_string(i) +
                                      ".norm3.weight"];
      layer.d_norm_b3 =
          model
              .tensors["decoder.decoders." + std::to_string(i) + ".norm3.bias"];
    }
    model.decoder.d_after_norm_w = model.tensors["decoder.after_norm.weight"];
    model.decoder.d_after_norm_b = model.tensors["decoder.after_norm.bias"];
    model.decoder.d_output_w = model.tensors["decoder.output_layer.weight"];
    model.decoder.d_output_b = model.tensors["decoder.output_layer.bias"];

    // decoders3.0.feed_forward.norm
    model.decoder.d3_mlp_norm_w =
        model.tensors["decoder.decoders3.0.feed_forward.norm.weight"];
    model.decoder.d3_mlp_norm_b =
        model.tensors["decoder.decoders3.0.feed_forward.norm.bias"];

    // decoders3.0.norm1
    model.decoder.d3_norm_w1 =
        model.tensors["decoder.decoders3.0.norm1.weight"];
    model.decoder.d3_norm_b1 = model.tensors["decoder.decoders3.0.norm1.bias"];
  }

  // predictor

  // bias encoder

  // bias decoder

  //
  pctx.t_load_ms = ggml_time_ms() - t_start_ms;
  PARAFORMER_LOG_INFO("%s: load paraformer model takes %f second \n", __func__,
                      pctx.t_load_ms * 1.0 / 1000);
  return true;
}

// static struct ggml_cgraph *paraformer_build_bias_encoder(paraformer_context
// &pctx, paraformer_state &pstate,
//                                                          const int
//                                                          mel_offset) {
//     const auto &model = pctx.model;
//     const auto &mel_inp = pstate.feature;
//     const auto &hparams = model.hparams;
//
//     const int n_ctx = pstate.exp_n_audio_ctx > 0 ? pstate.exp_n_audio_ctx :
//     hparams.n_audio_ctx; const int n_state = hparams.n_audio_state;
//     GGML_UNUSED(n_state);
//
//     const int n_mels = hparams.n_mels;
//
//     struct ggml_init_params params = {
//         /*.mem_size   =*/pstate.alloc_conv.meta.size(),
//         /*.mem_buffer =*/pstate.alloc_conv.meta.data(),
//         /*.no_alloc   =*/true,
//     };
//
//     struct ggml_context *ctx0 = ggml_init(params);
//
//     ggml_cgraph *gf = ggml_new_graph(ctx0);
//
//     ggml_allocr *alloc = pstate.alloc_conv.alloc;
//
//     struct ggml_tensor *mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 2 *
//     n_ctx, n_mels); ggml_allocr_alloc(alloc, mel);
//
//     assert(mel->type == GGML_TYPE_F32);
//     if (!ggml_allocr_is_measure(alloc)) {
//         assert(mel_inp.n_mel == n_mels);
//
//         float *dst = (float *)mel->data;
//         memset(dst, 0, ggml_nbytes(mel));
//
//         const int i0 = std::min(mel_offset, mel_inp.n_len);
//         const int i1 = std::min(mel_offset + 2 * n_ctx, mel_inp.n_len);
//
//         for (int j = 0; j < mel_inp.n_mel; ++j) {
//             for (int i = i0; i < i1; ++i) {
//                 dst[j * 2 * n_ctx + (i - i0)] = mel_inp.data[j *
//                 mel_inp.n_len + i];
//             }
//         }
//     }
//
//     struct ggml_tensor *cur = nullptr;
//
//     // convolution + gelu
//     {
//         cur = ggml_conv_1d_ph(ctx0, model.e_conv_1_w, mel, 1, 1);
//         cur = ggml_add(ctx0, ggml_repeat(ctx0, model.e_conv_1_b, cur),
//         cur);
//
//         cur = ggml_gelu(ctx0, cur);
//
//         cur = ggml_conv_1d_ph(ctx0, model.e_conv_2_w, cur, 2, 1);
//         cur = ggml_add(ctx0, ggml_repeat(ctx0, model.e_conv_2_b, cur),
//         cur);
//
//         cur = ggml_gelu(ctx0, cur);
//     }
//
//     pstate.embd_conv = cur;
//
//     ggml_build_forward_expand(gf, cur);
//
//     ggml_free(ctx0);
//
//     return gf;
// }

struct paraformer_state *paraformer_init_state(paraformer_context *ctx) {
  ctx->state = new paraformer_state;
  auto state = ctx->state;

  state->backend = paraformer_backend_init(ctx->params);
  if (!state->backend) {
    PARAFORMER_LOG_ERROR("%s: whisper_backend_init() failed\n", __func__);
    whisper_free_state(state);
    return nullptr;
  }

#ifdef USE_COREML
  const auto path_coreml = PARAFORMER_get_coreml_path_encoder(ctx->path_model);

  PARAFORMER_LOG_INFO("%s: loading Core ML model from '%s'\n", __func__,
                      path_coreml.c_str());
  PARAFORMER_LOG_INFO("%s: first run on a device may take a while ...\n",
                      __func__);

  state->ctx_coreml = PARAFORMER_coreml_init(path_coreml.c_str());
  if (!state->ctx_coreml) {
    PARAFORMER_LOG_INFO("%s: failed to load Core ML model from '%s'\n",
                        __func__, path_coreml.c_str());
#ifndef COREML_ALLOW_FALLBACK
    delete state;
    return nullptr;
#endif
  } else {
    PARAFORMER_LOG_INFO("%s: Core ML model loaded\n", __func__);
  }
#endif

  //    state->logits.reserve(ctx->vocab.n_vocab *
  //    ctx->model.hparams.n_text_ctx);

  state->logits_id.reserve(ctx->model.hparams.n_vocab);

  //  {
  //      paraformer_allocr_graph_init(state->alloc_bias_encoder, [&]() {
  //      return paraformer_build_graph_bias(*ctx, *state, 0); });
  //
  //      PARAFORMER_LOG_INFO("%s: compute buffer (conv)   = %7.2f MB\n",
  //      __func__,
  //          PARAFORMER_allocr_size(state->alloc_bias_encoder) / 1024.0 /
  //          1024.0);
  //  }

  // encoder allocator
  {
    bool ok = paraformer_allocr_graph_init(
        state->alloc_encode, ctx->backend,
        [&]() { return paraformer_build_graph_encoder(*ctx, *state); });

    if (!ok) {
      PARAFORMER_LOG_ERROR("%s: failed to init encode allocator\n", __func__);
      whisper_free_state(state);
      return nullptr;
    }

    PARAFORMER_LOG_INFO("%s: compute buffer (conv)   = %7.2f MB\n", __func__,
                        paraformer_allocr_size(state->alloc_encode) / 1e6);
  }

  // todo predict allocator
  {
    //        paraformer_allocr_graph_init(state->alloc_predict,
    //                                     [&]() { return
    //                                     paraformer_build_graph_predict(*ctx,
    //                                     *state); });

    //    PARAFORMER_LOG_INFO(
    //        "%s: compute buffer (cross)  = %7.2f MB\n", __func__,
    //        paraformer_allocr_size(state->alloc_predict) / 1024.0 / 1024.0);
  }

  // todo decoder allocator
  {
    //    PARAFORMER_LOG_INFO(
    //        "%s: compute buffer (decode) = %7.2f MB\n", __func__,
    //        paraformer_allocr_size(state->alloc_decode) / 1024.0 / 1024.0);
  }

#ifdef GGML_USE_METAL
  state->ctx_metal = ggml_metal_init(1);
  if (!state->ctx_metal) {
    PARAFORMER_LOG_INFO("%s: ggml_metal_init() failed\n", __func__);
    delete state;
    return nullptr;
  }

  PARAFORMER_LOG_INFO("%s: Metal context initialized\n", __func__);

  // this allocates all Metal resources and memory buffers

  void *data_ptr = NULL;
  size_t data_size = 0;

  // TODO: add mmap support
  // if (params.use_mmap) {
  //    data_ptr  = ctx->model.mapping->addr;
  //    data_size = ctx->model.mapping->size;
  //} else {
  //    data_ptr  = ggml_get_mem_buffer(ctx->model.ctx);
  //    data_size = ggml_get_mem_size  (ctx->model.ctx);
  //}

  data_ptr = ggml_get_mem_buffer(ctx->model.ctx);
  data_size = ggml_get_mem_size(ctx->model.ctx);

  const size_t max_size = ggml_get_max_tensor_size(ctx->model.ctx);

  PARAFORMER_LOG_INFO("%s: max tensor size = %8.2f MB\n", __func__,
                      max_size / 1024.0 / 1024.0);

#define PARAFORMER_METAL_CHECK_BUF(result)                             \
  if (!(result)) {                                                     \
    PARAFORMER_LOG_INFO("%s: failed to add metal buffer\n", __func__); \
    delete state;                                                      \
    return nullptr;                                                    \
  }

  PARAFORMER_METAL_CHECK_BUF(ggml_metal_add_buffer(
      state->ctx_metal, "data", data_ptr, data_size, max_size));

  PARAFORMER_METAL_CHECK_BUF(ggml_metal_add_buffer(
      state->ctx_metal, "meta_conv", state->alloc_conv.meta.data(),
      state->alloc_conv.meta.size(), 0));
  PARAFORMER_METAL_CHECK_BUF(ggml_metal_add_buffer(
      state->ctx_metal, "meta_encode", state->alloc_encode.meta.data(),
      state->alloc_encode.meta.size(), 0));
  PARAFORMER_METAL_CHECK_BUF(ggml_metal_add_buffer(
      state->ctx_metal, "meta_cross", state->alloc_cross.meta.data(),
      state->alloc_cross.meta.size(), 0));
  PARAFORMER_METAL_CHECK_BUF(ggml_metal_add_buffer(
      state->ctx_metal, "meta_decode", state->alloc_decode.meta.data(),
      state->alloc_decode.meta.size(), 0));

  PARAFORMER_METAL_CHECK_BUF(ggml_metal_add_buffer(
      state->ctx_metal, "data_conv", state->alloc_conv.data.data(),
      state->alloc_conv.data.size(), 0));
  PARAFORMER_METAL_CHECK_BUF(ggml_metal_add_buffer(
      state->ctx_metal, "data_encode", state->alloc_encode.data.data(),
      state->alloc_encode.data.size(), 0));
  PARAFORMER_METAL_CHECK_BUF(ggml_metal_add_buffer(
      state->ctx_metal, "data_cross", state->alloc_cross.data.data(),
      state->alloc_cross.data.size(), 0));
  PARAFORMER_METAL_CHECK_BUF(ggml_metal_add_buffer(
      state->ctx_metal, "data_decode", state->alloc_decode.data.data(),
      state->alloc_decode.data.size(), 0));

  PARAFORMER_METAL_CHECK_BUF(ggml_metal_add_buffer(
      state->ctx_metal, "kv_cross", state->kv_cross.buf.data(),
      state->kv_cross.buf.size(), 0));

  PARAFORMER_METAL_CHECK_BUF(ggml_metal_add_buffer(
      state->ctx_metal, "kv_self_0", state->decoders[0].kv_self.buf.data(),
      state->decoders[0].kv_self.buf.size(), 0));
#undef PARAFORMER_METAL_CHECK_BUF
#endif

  return state;
}

struct ggml_cgraph *paraformer_build_graph_encoder(paraformer_context &pctx,
                                                   paraformer_state &pstate) {
  const auto &model = pctx.model;
  const auto &hparams = model.hparams;
  const int n_ctx =
      pstate.exp_n_audio_ctx > 0 ? pstate.exp_n_audio_ctx : hparams.n_audio_ctx;
  const int n_state = hparams.n_encoder_hidden_state;
  const int n_head = hparams.n_encoder_attention_heads;
  const int n_layer = hparams.n_encoder_layers;

  struct ggml_init_params params = {
      /*.mem_size   =*/pstate.alloc_encode.meta.size(),
      /*.mem_buffer =*/pstate.alloc_encode.meta.data(),
      /*.no_alloc   =*/true,
  };

  struct ggml_context *ctx0 = ggml_init(params);

  ggml_cgraph *gf = ggml_new_graph(ctx0);

  struct ggml_tensor *fbank = ggml_new_tensor_2d(
      ctx0, GGML_TYPE_F32, 2 * n_ctx, hparams.n_mels * hparams.lfr_m);

  struct ggml_tensor *position = ggml_new_tensor_2d(
      ctx0, GGML_TYPE_F32, 2 * n_ctx, hparams.n_mels * hparams.lfr_m);

  ggml_set_name(fbank, "fbank");
  ggml_set_input(fbank);

  ggml_set_name(position, "position");
  ggml_set_input(position);

  // token encoding + position encoding
  struct ggml_tensor *cur = ggml_add(ctx0, fbank, position);

  static int iter = 0;

  // norm before attention
  struct ggml_tensor *inpL = cur;

  for (auto layer : model.encoder.encoder_layer) {
    cur = ggml_norm(ctx0, inpL, hparams.eps);

    // layer norm
    // cur = ln_0_w*cur + ln_0_b
    cur = ggml_add(ctx0, ggml_mul(ctx0, cur, layer.e_norm_w1), layer.e_norm_b1);

    // self attention
    {
      // self attention linear qkv
      cur = ggml_add(ctx0, ggml_mul(ctx0, cur, layer.e_attn_ln_qkv_w),
                     layer.e_attn_ln_qkv_b);

      // split qkv into separate tensors
      //  ref:
      //  https://github.com/alibaba-damo-academy/FunASR/blob/main/funasr/modules/attention.py#L391-L396
      struct ggml_tensor *Q;
      struct ggml_tensor *K;
      struct ggml_tensor *V;

      Q = ggml_view_2d(ctx0, cur, n_state, n_ctx, cur->nb[1], 0 * cur->nb[2]);
      Q = ggml_reshape_4d(ctx0, Q, n_state / n_head, n_head, n_ctx, 1);
      Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
      Q = ggml_reshape_3d(ctx0, Q, n_state, n_ctx, n_head);

      K = ggml_view_2d(ctx0, cur, n_state, n_ctx, cur->nb[1], 1 * cur->nb[2]);
      K = ggml_reshape_4d(ctx0, K, n_state / n_head, n_head, n_ctx, 1);
      K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
      K = ggml_reshape_3d(ctx0, K, n_state, n_ctx, n_head);

      V = ggml_view_2d(ctx0, cur, n_state, n_ctx, cur->nb[1], 2 * cur->nb[2]);
      V = ggml_reshape_4d(ctx0, V, n_state / n_head, n_head, n_ctx, 1);
      V = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 2, 0, 3));  // transposed
      V = ggml_reshape_3d(ctx0, V, n_state, n_ctx, n_head);

      // fsmn forward with V
      // todo transpose(1,2)
      V = ggml_conv_1d_ph(ctx0, V, layer.e_attn_fsmn_w, 1, 1);

#ifdef USE_FLASH_ATTN

      struct ggml_tensor *KQV = ggml_flash_attn(ctx0, Q, K, V, false);
#else

      // K * Q
      struct ggml_tensor *KQ = ggml_mul_mat(ctx0, K, Q);

      float KQscale = 1.0f / sqrtf(float(n_state) / n_head);

      struct ggml_tensor *KQ_scaled = ggml_scale(ctx0, KQ, KQscale);

      struct ggml_tensor *KQ_soft_max = ggml_soft_max(ctx0, KQ_scaled);

      struct ggml_tensor *KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);
#endif
      struct ggml_tensor *KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
      cur = ggml_cpy(ctx0, KQV_merged,
                     ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_state, n_ctx));
    }
    // norm after attention
  }

  ggml_build_forward_expand(gf, cur);

  ggml_free(ctx0);

  return gf;
}

static bool paraformer_encode_internal(paraformer_context &ctx,
                                       paraformer_state &state,
                                       const int n_threads) {
  const int64_t t_start_us = ggml_time_us();

  const auto &model = ctx.model;
  const auto &hparams = model.hparams;
  const int n_vocab = hparams.n_vocab;

  auto &logits_out = state.logits;

  struct ggml_tensor *logits;

  // encoder
  {
    auto &alloc = state.alloc_encode.alloc;

    ggml_cgraph *gf = paraformer_build_graph_encoder(ctx, state);

    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
      // should never happen as we pre-allocate the memory
      return false;
    }
  }
}

int paraformer_full_with_state(struct paraformer_context *ctx,
                               struct paraformer_state *state,
                               struct paraformer_full_params params,
                               const paraformer_feature &feature,
                               int n_threads) {
  if (!paraformer_encode_internal(*ctx, *state, n_threads)) {
    fprintf(stderr, "%s: failed to encode\n", __func__);
    return -6;
  }
}

struct paraformer_full_params paraformer_full_default_params(
    enum paraformer_decoding_strategy strategy) {
  struct paraformer_full_params result = {
      /*.strategy          =*/strategy,

      /*.n_threads         =*/
      std::min(4, (int32_t)std::thread::hardware_concurrency()),
      /*.n_max_text_ctx    =*/16384,
      /*.offset_ms         =*/0,
      /*.duration_ms       =*/0,

      /*.no_timestamps     =*/false,
      /*.single_segment    =*/false,
      /*.print_progress    =*/true,
      /*.print_timestamps  =*/true,

      /*.debug_mode        =*/false,

      /*.greedy            =*/
      {
          /*.best_of   =*/-1,
      },

      /*.beam_search      =*/
      {
          /*.beam_size =*/-1,
      },

      /*.progress_callback           =*/nullptr,
      /*.progress_callback_user_data =*/nullptr};

  switch (strategy) {
    case PARAFORMER_SAMPLING_GREEDY: {
      result.greedy = {
          /*.best_of   =*/5,
      };
    } break;
    case PARAFORMER_SAMPLING_BEAM_SEARCH: {
      result.beam_search = {/*.beam_size =*/5};
    } break;
  }

  return result;
}