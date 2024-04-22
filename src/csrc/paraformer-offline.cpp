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
#include <string>
#include <vector>

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
// #define PARAFORMER_DEBUG

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
            {MODEL_OFFLINE, 0ull * MB},
            {MODEL_CONTEXTUAL_OFFLINE, 228ull * 4 * MB},
        },
    },
    {
        GGML_TYPE_F16,
        {
            {MODEL_ONLINE, 0ull * MB},
            {MODEL_OFFLINE, 0ull * MB},
            {MODEL_CONTEXTUAL_OFFLINE, 228ull * 2 * MB},
        },
    },
    {
        GGML_TYPE_Q4_0,
        {
            {MODEL_ONLINE, 0ull * MB},
            {MODEL_OFFLINE, 0ull * MB},
            {MODEL_CONTEXTUAL_OFFLINE, 114ull * MB},
        },
    },

    {
        GGML_TYPE_I8,
        {
            {MODEL_ONLINE, 0ull * MB},
            {MODEL_OFFLINE, 0ull * MB},
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
  ggml_gallocr_t *alloc = nullptr;
  std::vector<uint8_t> meta;
  std::vector<uint8_t> data;
};

struct paraformer_context {
  int64_t t_load_ms = 0;
  int64_t t_start_ms = 0;

  ggml_type wtype = ggml_type::GGML_TYPE_F16;  // weight type (FP32 / FP16 / QX)
  ggml_type itype =
      ggml_type::GGML_TYPE_F16;  // intermediate type (FP32 or FP16)

  paraformer_model model;
  paraformer_vocab vocab;

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
  return allocr.meta.size() + allocr.data.size();
}

// measure the memory usage of a graph and prepare the allocr's internal data
// buffer
static bool paraformer_allocr_graph_init(
    struct paraformer_allocr &allocr, ggml_backend_t backend,
    std::function<struct ggml_cgraph *()> &&get_graph) {
  auto &alloc = allocr.alloc;
  auto &meta = allocr.meta;

  alloc = reinterpret_cast<ggml_gallocr_t *>(
      ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)));

  meta.resize(ggml_tensor_overhead() * PARAFORMER_MAX_NODES +
              ggml_graph_overhead());

  // since there are dependencies between the different graphs,
  // we need to allocate them instead of only reserving to get the correct
  // compute buffer size
  if (!ggml_gallocr_alloc_graph(*alloc, get_graph())) {
    // failed to allocate the compute buffer
    PARAFORMER_LOG_ERROR("%s: failed to allocate the compute buffer\n",
                         __func__);
    return false;
  }
  return true;
}

struct paraformer_context *paraformer_init(
    struct paraformer_model_loader *loader) {
  ggml_time_init();

  struct paraformer_context *ctx = new paraformer_context;

  if (!paraformer_model_load(loader, *ctx)) {
    loader->close(loader->context);
    PARAFORMER_LOG_INFO("%s: failed to load model\n", __func__);
    delete ctx;
    return nullptr;
  }

  loader->close(loader->context);

  return ctx;
}

struct paraformer_context_params paraformer_context_default_params() {
  struct paraformer_context_params result = {
      /*.use_gpu              =*/true,
      /*.gpu_device           =*/0,
  };
  return result;
}

struct paraformer_context *paraformer_init_from_file_with_params(
    const char *path_model, struct paraformer_context_params params) {
  PARAFORMER_LOG_INFO("%s: loading model from '%s'\n", __func__, path_model);

  auto fin = std::ifstream(path_model, std::ios::binary);
  if (!fin) {
    PARAFORMER_LOG_INFO("%s: failed to open '%s'\n", __func__, path_model);
    return nullptr;
  }

  paraformer_model_loader loader = {};

  loader.context = &fin;

  loader.read = [](void *ctx, void *output, size_t read_size) {
    auto fin = (std::ifstream *)ctx;
    fin->read((char *)output, read_size);
    return read_size;
  };

  loader.eof = [](void *ctx) {
    auto *fin = (std::ifstream *)ctx;
    return fin->eof();
  };

  loader.close = [](void *ctx) {
    auto *fin = (std::ifstream *)ctx;
    fin->close();
  };

  auto ctx = paraformer_init(&loader);
  paraformer_init_state(ctx);

  if (ctx) {
    ctx->path_model = path_model;
  }

  return ctx;
}

struct paraformer_context *paraformer_init_from_file(const char *path_model) {
  return paraformer_init_from_file_with_params(
      path_model, paraformer_context_default_params());
}

// load the model from a ggml file
//
// file format:
//
//   - hparams
//   - pre-computed mel filters
//   - vocab
//   - weights
//
// see the convert-pt-to-ggml.py script for details
//
bool paraformer_model_load(struct paraformer_model_loader *loader,
                           paraformer_context &pctx) {
  PARAFORMER_LOG_INFO("%s: loading model\n", __func__);

  const int64_t t_start_ms = ggml_time_ms();

  pctx.t_start_ms = t_start_ms;

  auto &model = pctx.model;
  auto &vocab = pctx.vocab;

  // verify magic
  {
    uint32_t magic;
    read_safe(loader, magic);
    if (magic != GGML_FILE_MAGIC) {
      PARAFORMER_LOG_INFO("%s: invalid model data (bad magic)\n", __func__);
      return false;
    }
  }

  // load hparams
  {
    auto &hparams = model.hparams;

    read_safe(loader, hparams.n_vocab);
    read_safe(loader, hparams.n_encoder_hidden_state);
    read_safe(loader, hparams.n_encoder_linear_units);
    read_safe(loader, hparams.n_encoder_attention_heads);
    read_safe(loader, hparams.n_encoder_layers);
    read_safe(loader, hparams.n_encoder_0_norm_size);
    read_safe(loader, hparams.n_decoder_hidden_state);
    read_safe(loader, hparams.n_decoder_linear_units);
    read_safe(loader, hparams.n_decoder_attention_heads);
    read_safe(loader, hparams.n_decoder_layers);
    read_safe(loader, hparams.fsmn_kernel_size);
    read_safe(loader, hparams.n_predictor_dim);
    read_safe(loader, hparams.predictor_tail_threshold);

    assert(hparams.n_decoder_hidden_state == hparams.n_encoder_hidden_state);
    model.type = hparams.model_type;

    const int32_t qntvr = hparams.ftype / GGML_QNT_VERSION_FACTOR;

    hparams.ftype %= GGML_QNT_VERSION_FACTOR;

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
    PARAFORMER_LOG_INFO("%s: n_encoder_hidden_state   = %d\n", __func__,
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
    PARAFORMER_LOG_INFO("%s: qntvr         = %d\n", __func__, qntvr);
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
    int n_vocab = 0;
    read_safe(loader, n_vocab);

    if (n_vocab != model.hparams.n_vocab) {
      PARAFORMER_LOG_INFO("%s: invalid model file (bad vocab size %d != %d)\n",
                          __func__, n_vocab, model.hparams.n_vocab);
      return false;
    }

    std::string word;
    std::vector<char> tmp;

    tmp.reserve(128);

    for (int i = 0; i < n_vocab; i++) {
      uint8_t len;
      read_safe(loader, len);

      if (len > 0) {
        tmp.resize(len);
        loader->read(loader->context, &tmp[0],
                     tmp.size());  // read to buffer
        word.assign(&tmp[0], tmp.size());
      } else {
        // seems like we have an empty-string token in multi-language
        // models (i = 50256)
        // PARAFORMER_LOG_INFO("%s: warning: empty-string token in vocab, i =
        // %d\n",
        // __func__, i);
        word = "";
      }

      vocab.token_to_id[word] = i;
      vocab.id_to_token[i] = word;

      // printf("%s: vocab[%d] = '%s'\n", __func__, i, word.c_str());
    }

    vocab.n_vocab = model.hparams.n_vocab;
  }

  size_t ctx_size = 0;

  const ggml_type wtype = pctx.wtype;
  const ggml_type vtype =
      pctx.wtype == GGML_TYPE_F32 ? GGML_TYPE_F32 : GGML_TYPE_F16;  // conv type

  // create the ggml context
  {
    struct ggml_init_params params = {
        /*.mem_size   =*/pctx.model.buf->size(),
        /*.mem_buffer =*/pctx.model.buf->data(),
        /*.no_alloc   =*/false,
    };

    model.ctx = ggml_init(params);
    if (!model.ctx) {
      PARAFORMER_LOG_INFO("%s: ggml_init() failed\n", __func__);
      return false;
    }
  }

  {
    const auto &hparams = model.hparams;

    const int n_vocab = hparams.n_vocab;
    const int n_encoder_layers = hparams.n_encoder_layers;
    const int n_decoder_layers = hparams.n_decoder_layers;

    model.encoder.encoder_layer.resize(n_encoder_layers);
    model.decoder.decoder_layers.resize(n_decoder_layers);

    // create node
    {
      // encoder layers
      for (int i = 0; i < hparams.n_encoder_layers; ++i) {
        auto &layer = model.encoder.encoder_layer[i];
        // encoder_attn.linear_out.weight
        layer.e_attn_ln_out_w =
            ggml_new_tensor_2d(model.ctx, wtype, hparams.n_encoder_hidden_state,
                               hparams.n_encoder_hidden_state);
        layer.e_attn_ln_out_b = ggml_new_tensor_1d(
            model.ctx, wtype, hparams.n_encoder_hidden_state);

        // encoder.self_attn.linear_q_k_v.weight
        layer.e_attn_ln_qkv_w =
            ggml_new_tensor_2d(model.ctx, wtype,
                               i == 0 ? hparams.n_encoder_0_norm_size
                                      : hparams.n_encoder_hidden_state,
                               3 * hparams.n_encoder_hidden_state);
        layer.e_attn_ln_qkv_b = ggml_new_tensor_1d(
            model.ctx, wtype, 3 * hparams.n_encoder_hidden_state);

        // encoder.self_attn.fsmn_block.weight
        layer.e_attn_fsmn_w =
            ggml_new_tensor_3d(model.ctx, wtype, hparams.fsmn_kernel_size,
                               hparams.n_encoder_hidden_state, 1);

        // encoder.feed_forward.w_1.weight
        layer.e_mlp_w1 =
            ggml_new_tensor_2d(model.ctx, wtype, hparams.n_encoder_hidden_state,
                               hparams.n_encoder_linear_units);
        layer.e_mlp_b1 = ggml_new_tensor_1d(model.ctx, wtype,
                                            hparams.n_encoder_linear_units);

        // encoder.feed_forward.w_2.weight
        layer.e_mlp_w2 =
            ggml_new_tensor_2d(model.ctx, wtype, hparams.n_encoder_linear_units,
                               hparams.n_encoder_hidden_state);
        layer.e_mlp_b2 = ggml_new_tensor_1d(model.ctx, wtype,
                                            hparams.n_encoder_hidden_state);

        // encoder.norm1.weight
        layer.e_norm_w1 =
            ggml_new_tensor_1d(model.ctx, wtype,
                               i == 0 ? hparams.n_encoder_0_norm_size
                                      : hparams.n_encoder_hidden_state);
        layer.e_norm_b1 =
            ggml_new_tensor_1d(model.ctx, wtype,
                               i == 0 ? hparams.n_encoder_0_norm_size
                                      : hparams.n_encoder_hidden_state);

        // encoder.norm2.weight
        layer.e_norm_w2 = ggml_new_tensor_1d(model.ctx, wtype,
                                             hparams.n_encoder_hidden_state);
        layer.e_norm_b2 = ggml_new_tensor_1d(model.ctx, wtype,
                                             hparams.n_encoder_hidden_state);

        // map by name
        model.tensors[(i == 0 ? ("encoder.encoders0." + std::to_string(i))
                              : ("encoder.encoders." + std::to_string(i - 1))) +
                      ".self_attn.linear_out.weight"] = layer.e_attn_ln_out_w;
        model.tensors[(i == 0 ? ("encoder.encoders0." + std::to_string(i))
                              : ("encoder.encoders." + std::to_string(i - 1))) +
                      ".self_attn.linear_out.bias"] = layer.e_attn_ln_out_b;

        model.tensors[(i == 0 ? ("encoder.encoders0." + std::to_string(i))
                              : ("encoder.encoders." + std::to_string(i - 1))) +
                      ".self_attn.linear_q_k_v.weight"] = layer.e_attn_ln_qkv_w;
        model.tensors[(i == 0 ? ("encoder.encoders0." + std::to_string(i))
                              : ("encoder.encoders." + std::to_string(i - 1))) +
                      ".self_attn.linear_q_k_v.bias"] = layer.e_attn_ln_qkv_b;

        model.tensors[(i == 0 ? ("encoder.encoders0." + std::to_string(i))
                              : ("encoder.encoders." + std::to_string(i - 1))) +
                      ".self_attn.fsmn_block.weight"] = layer.e_attn_fsmn_w;

        model.tensors[(i == 0 ? ("encoder.encoders0." + std::to_string(i))
                              : ("encoder.encoders." + std::to_string(i - 1))) +
                      ".feed_forward.w_1.weight"] = layer.e_mlp_w1;
        model.tensors[(i == 0 ? ("encoder.encoders0." + std::to_string(i))
                              : ("encoder.encoders.") + std::to_string(i - 1)) +
                      ".feed_forward.w_1.bias"] = layer.e_mlp_b1;

        model.tensors[(i == 0 ? ("encoder.encoders0." + std::to_string(i))
                              : ("encoder.encoders.") + std::to_string(i - 1)) +
                      ".feed_forward.w_2.weight"] = layer.e_mlp_w2;
        model.tensors[(i == 0 ? ("encoder.encoders0." + std::to_string(i))
                              : ("encoder.encoders.") + std::to_string(i - 1)) +
                      ".feed_forward.w_2.bias"] = layer.e_mlp_b2;

        model.tensors[(i == 0 ? ("encoder.encoders0." + std::to_string(i))
                              : ("encoder.encoders.") + std::to_string(i - 1)) +
                      ".norm1.weight"] = layer.e_norm_w1;
        model.tensors[(i == 0 ? ("encoder.encoders0." + std::to_string(i))
                              : ("encoder.encoders.") + std::to_string(i - 1)) +
                      ".norm1.bias"] = layer.e_norm_b1;

        model.tensors[(i == 0 ? ("encoder.encoders0." + std::to_string(i))
                              : ("encoder.encoders.") + std::to_string(i - 1)) +
                      ".norm2.weight"] = layer.e_norm_w2;
        model.tensors[(i == 0 ? ("encoder.encoders0." + std::to_string(i))
                              : ("encoder.encoders.") + std::to_string(i - 1)) +
                      ".norm2.bias"] = layer.e_norm_b2;
      }

      model.encoder.e_after_norm_w =
          ggml_new_tensor_1d(model.ctx, wtype, hparams.n_encoder_hidden_state);
      model.encoder.e_after_norm_b =
          ggml_new_tensor_1d(model.ctx, wtype, hparams.n_encoder_hidden_state);
      model.tensors["encoder.after_norm.weight"] = model.encoder.e_after_norm_w;
      model.tensors["encoder.after_norm.bias"] = model.encoder.e_after_norm_b;

      // decoder layers

      {
        model.decoder.embed = ggml_new_tensor_2d(
            model.ctx, wtype, hparams.n_decoder_hidden_state, hparams.n_vocab);

        model.tensors["decoder.embed.0.weight"] = model.decoder.embed;

        // decoder layers
        for (int i = 0; i < hparams.n_decoder_layers; ++i) {
          auto &layer = model.decoder.decoder_layers[i];

          // decoder(i).self_attn.fsmn_block
          layer.d_attn_fsmn_w =
              ggml_new_tensor_3d(model.ctx, wtype, hparams.fsmn_kernel_size,
                                 hparams.n_decoder_hidden_state, 1);

          // decoder(i).src_attn.linear_q
          layer.d_src_attn_ln_q_w = ggml_new_tensor_2d(
              model.ctx, wtype, hparams.n_decoder_hidden_state,
              hparams.n_decoder_hidden_state);
          layer.d_src_attn_ln_q_b = ggml_new_tensor_1d(
              model.ctx, wtype, hparams.n_decoder_hidden_state);

          // decoder(i).src_attn.linear_k_v
          layer.d_src_attn_ln_kv_w = ggml_new_tensor_2d(
              model.ctx, wtype, hparams.n_decoder_hidden_state,
              2 * hparams.n_decoder_hidden_state);
          layer.d_src_attn_ln_kv_b = ggml_new_tensor_1d(
              model.ctx, wtype, 2 * hparams.n_decoder_hidden_state);

          // decoder(i).src_attn.linear_out
          layer.d_src_attn_ln_o_w = ggml_new_tensor_2d(
              model.ctx, wtype, hparams.n_decoder_hidden_state,
              hparams.n_decoder_hidden_state);
          layer.d_src_attn_ln_o_b = ggml_new_tensor_1d(
              model.ctx, wtype, hparams.n_decoder_hidden_state);

          // decoder(i).feed_forward1
          layer.d_mlp_ln_w1 = ggml_new_tensor_2d(
              model.ctx, wtype, hparams.n_decoder_hidden_state,
              hparams.n_decoder_linear_units);
          layer.d_mlp_ln_b1 = ggml_new_tensor_1d(
              model.ctx, wtype, hparams.n_decoder_linear_units);

          // decoder(i).feed_forward2
          layer.d_mlp_ln_w2 = ggml_new_tensor_2d(
              model.ctx, wtype, hparams.n_decoder_linear_units,
              hparams.n_decoder_hidden_state);

          // decoder(i).feed_forward.norm
          layer.d_mlp_norm_w = ggml_new_tensor_1d(
              model.ctx, wtype, hparams.n_decoder_linear_units);
          layer.d_mlp_norm_b = ggml_new_tensor_1d(
              model.ctx, wtype, hparams.n_decoder_linear_units);

          // decoder.norm1.weight
          layer.d_norm_w1 = ggml_new_tensor_1d(model.ctx, wtype,
                                               hparams.n_decoder_hidden_state);
          layer.d_norm_b1 = ggml_new_tensor_1d(model.ctx, wtype,
                                               hparams.n_decoder_hidden_state);

          // decoder.norm2.weight
          layer.d_norm_w2 = ggml_new_tensor_1d(model.ctx, wtype,
                                               hparams.n_decoder_hidden_state);
          layer.d_norm_b2 = ggml_new_tensor_1d(model.ctx, wtype,
                                               hparams.n_decoder_hidden_state);

          // decoder.norm3.weight
          layer.d_norm_w3 = ggml_new_tensor_1d(model.ctx, wtype,
                                               hparams.n_decoder_hidden_state);
          layer.d_norm_b3 = ggml_new_tensor_1d(model.ctx, wtype,
                                               hparams.n_decoder_hidden_state);

          model.tensors["decoder.decoders." + std::to_string(i) +
                        ".self_attn.fsmn_block.weight"] = layer.d_attn_fsmn_w;

          model.tensors["decoder.decoders." + std::to_string(i) +
                        ".src_attn.linear_q.weight"] = layer.d_src_attn_ln_q_w;
          model.tensors["decoder.decoders." + std::to_string(i) +
                        ".src_attn.linear_q.bias"] = layer.d_src_attn_ln_q_b;

          model.tensors["decoder.decoders." + std::to_string(i) +
                        ".src_attn.linear_k_v.weight"] =
              layer.d_src_attn_ln_kv_w;
          model.tensors["decoder.decoders." + std::to_string(i) +
                        ".src_attn.linear_k_v.bias"] = layer.d_src_attn_ln_kv_b;

          model.tensors["decoder.decoders." + std::to_string(i) +
                        ".src_attn.linear_out.weight"] =
              layer.d_src_attn_ln_o_w;
          model.tensors["decoder.decoders." + std::to_string(i) +
                        ".src_attn.linear_out.bias"] = layer.d_src_attn_ln_o_b;

          model.tensors["decoder.decoders." + std::to_string(i) +
                        ".feed_forward.w_1.weight"] = layer.d_mlp_ln_w1;
          model.tensors["decoder.decoders." + std::to_string(i) +
                        ".feed_forward.w_1.bias"] = layer.d_mlp_ln_b1;

          model.tensors["decoder.decoders." + std::to_string(i) +
                        ".feed_forward.w_2.weight"] = layer.d_mlp_ln_w2;

          model.tensors["decoder.decoders." + std::to_string(i) +
                        ".feed_forward.norm.weight"] = layer.d_mlp_norm_w;
          model.tensors["decoder.decoders." + std::to_string(i) +
                        ".feed_forward.norm.bias"] = layer.d_mlp_norm_b;

          model.tensors["decoder.decoders." + std::to_string(i) +
                        ".norm1.weight"] = layer.d_norm_w1;
          model.tensors["decoder.decoders." + std::to_string(i) +
                        ".norm1.bias"] = layer.d_norm_b1;

          model.tensors["decoder.decoders." + std::to_string(i) +
                        ".norm2.weight"] = layer.d_norm_w2;
          model.tensors["decoder.decoders." + std::to_string(i) +
                        ".norm2.bias"] = layer.d_norm_b2;

          model.tensors["decoder.decoders." + std::to_string(i) +
                        ".norm3.weight"] = layer.d_norm_w3;
          model.tensors["decoder.decoders." + std::to_string(i) +
                        ".norm3.bias"] = layer.d_norm_b3;
        }
        // decoder after norm
        model.decoder.d_after_norm_w = ggml_new_tensor_1d(
            model.ctx, wtype, hparams.n_decoder_hidden_state);
        model.decoder.d_after_norm_b = ggml_new_tensor_1d(
            model.ctx, wtype, hparams.n_decoder_hidden_state);

        model.tensors["decoder.after_norm.weight"] =
            model.decoder.d_after_norm_w;
        model.tensors["decoder.after_norm.bias"] = model.decoder.d_after_norm_b;

        // decoder output layer
        model.decoder.d_output_w = ggml_new_tensor_2d(
            model.ctx, wtype, hparams.n_decoder_hidden_state, hparams.n_vocab);
        model.decoder.d_output_b =
            ggml_new_tensor_1d(model.ctx, wtype, hparams.n_vocab);

        model.tensors["decoder.output_layer.weight"] = model.decoder.d_output_w;
        model.tensors["decoder.output_layer.bias"] = model.decoder.d_output_b;

        // decoders3.0 layers
        {
          //  decoder output layer

          // decoders3.0.feed_forward.w_1
          model.decoder.d3_mlp_ln_w1 = ggml_new_tensor_2d(
              model.ctx, wtype, hparams.n_encoder_hidden_state,
              hparams.n_decoder_linear_units);
          model.decoder.d3_mlp_ln_b1 = ggml_new_tensor_1d(
              model.ctx, wtype, hparams.n_decoder_linear_units);

          model.tensors["decoder.decoders3.0.feed_forward.w_1.weight"] =
              model.decoder.d3_mlp_ln_w1;
          model.tensors["decoder.decoders3.0.feed_forward.w_1.bias"] =
              model.decoder.d3_mlp_ln_b1;

          // decoders3.0.feed_forward.w_2
          model.decoder.d3_mlp_ln_w2 = ggml_new_tensor_2d(
              model.ctx, wtype, hparams.n_decoder_linear_units,
              hparams.n_encoder_hidden_state);
          model.tensors["decoder.decoders3.0.feed_forward.w_2.weight"] =
              model.decoder.d3_mlp_ln_w2;

          // decoders3.0.feed_forward.norm
          model.decoder.d3_mlp_norm_w = ggml_new_tensor_1d(
              model.ctx, wtype, hparams.n_decoder_linear_units);
          model.decoder.d3_mlp_norm_b = ggml_new_tensor_1d(
              model.ctx, wtype, hparams.n_decoder_linear_units);
          model.tensors["decoder.decoders3.0.feed_forward.norm.weight"] =
              model.decoder.d3_mlp_norm_w;
          model.tensors["decoder.decoders3.0.feed_forward.norm.bias"] =
              model.decoder.d3_mlp_norm_b;

          // decoders3.0.norm1
          model.decoder.d3_norm_w1 = ggml_new_tensor_1d(
              model.ctx, wtype, hparams.n_decoder_hidden_state);
          model.decoder.d3_norm_b1 = ggml_new_tensor_1d(
              model.ctx, wtype, hparams.n_decoder_hidden_state);
          model.tensors["decoder.decoders3.0.norm1.weight"] =
              model.decoder.d3_norm_w1;
          model.tensors["decoder.decoders3.0.norm1.bias"] =
              model.decoder.d3_norm_b1;
        }
        // bias_decoder
        {
          // bias_decoder.src_attn.linear_q
          model.decoder.d_bias_src_attn_ln_q_w = ggml_new_tensor_2d(
              model.ctx, wtype, hparams.n_decoder_hidden_state,
              hparams.n_decoder_hidden_state);
          model.decoder.d_bias_src_attn_ln_q_b = ggml_new_tensor_1d(
              model.ctx, wtype, hparams.n_decoder_hidden_state);
          model.tensors["decoder.bias_decoder.src_attn.linear_q.weight"] =
              model.decoder.d_bias_src_attn_ln_q_w;
          model.tensors["decoder.bias_decoder.src_attn.linear_q.bias"] =
              model.decoder.d_bias_src_attn_ln_q_b;

          // bias_decoder.src_attn.linear_k_v
          model.decoder.d_bias_src_attn_ln_kv_w = ggml_new_tensor_2d(
              model.ctx, wtype, hparams.n_decoder_hidden_state,
              hparams.n_decoder_hidden_state * 2);
          model.decoder.d_bias_src_attn_ln_kv_b = ggml_new_tensor_1d(
              model.ctx, wtype, hparams.n_decoder_hidden_state * 2);
          model.tensors["decoder.bias_decoder.src_attn.linear_k_v.weight"] =
              model.decoder.d_bias_src_attn_ln_kv_w;
          model.tensors["decoder.bias_decoder.src_attn.linear_k_v.bias"] =
              model.decoder.d_bias_src_attn_ln_kv_b;

          // bias_decoder.src_attn.linear_out
          model.decoder.d_bias_src_attn_ln_o_w = ggml_new_tensor_2d(
              model.ctx, wtype, hparams.n_decoder_hidden_state,
              hparams.n_decoder_hidden_state);
          model.decoder.d_bias_src_attn_ln_o_b = ggml_new_tensor_1d(
              model.ctx, wtype, hparams.n_decoder_hidden_state);
          model.tensors["decoder.bias_decoder.src_attn.linear_out.weight"] =
              model.decoder.d_bias_src_attn_ln_o_w;
          model.tensors["decoder.bias_decoder.src_attn.linear_out.bias"] =
              model.decoder.d_bias_src_attn_ln_o_b;

          // bias_decoder.norm3
          model.decoder.d_bias_norm_w3 = ggml_new_tensor_1d(
              model.ctx, wtype, hparams.n_decoder_hidden_state);
          model.decoder.d_bias_norm_b3 = ggml_new_tensor_1d(
              model.ctx, wtype, hparams.n_decoder_hidden_state);
          model.tensors["decoder.bias_decoder.norm3.weight"] =
              model.decoder.d_bias_norm_w3;
          model.tensors["decoder.bias_decoder.norm3.bias"] =
              model.decoder.d_bias_norm_b3;

          // bias_decoder.output
          model.decoder.d_bias_ln_o_w = ggml_new_tensor_3d(
              model.ctx, wtype, hparams.n_decoder_hidden_state * 2,
              hparams.n_decoder_hidden_state, 1);

          model.tensors["decoder.bias_output.weight"] =
              model.decoder.d_bias_ln_o_w;
        }
        // decoder.last_decoder
        {
          // decoder.last_decoder.self_attn.fsmn_block
          model.decoder.d_last_attn_fsmn_w =
              ggml_new_tensor_3d(model.ctx, wtype, hparams.fsmn_kernel_size,
                                 hparams.n_decoder_hidden_state, 1);

          // decoder.last_decoder.self_attn.linear_q
          model.decoder.d_last_src_attn_ln_q_w = ggml_new_tensor_2d(
              model.ctx, wtype, hparams.n_decoder_hidden_state,
              hparams.n_decoder_hidden_state);
          model.decoder.d_last_src_attn_ln_q_b = ggml_new_tensor_1d(
              model.ctx, wtype, hparams.n_decoder_hidden_state);

          // decoder.last_decoder.src_attn.linear_k_v
          model.decoder.d_last_src_attn_ln_kv_w = ggml_new_tensor_2d(
              model.ctx, wtype, hparams.n_decoder_hidden_state,
              2 * hparams.n_decoder_hidden_state);
          model.decoder.d_last_src_attn_ln_kv_b = ggml_new_tensor_1d(
              model.ctx, wtype, 2 * hparams.n_decoder_hidden_state);

          // decoder.last_decoder.src_attn.linear_out
          model.decoder.d_last_src_attn_ln_o_w = ggml_new_tensor_2d(
              model.ctx, wtype, hparams.n_decoder_hidden_state,
              hparams.n_decoder_hidden_state);
          model.decoder.d_last_src_attn_ln_o_b = ggml_new_tensor_1d(
              model.ctx, wtype, hparams.n_decoder_hidden_state);

          // decoder.last_decoder.feed_forward1
          model.decoder.d_last_mlp_ln_w1 = ggml_new_tensor_2d(
              model.ctx, wtype, hparams.n_decoder_hidden_state,
              hparams.n_decoder_linear_units);
          model.decoder.d_last_mlp_ln_b1 = ggml_new_tensor_1d(
              model.ctx, wtype, hparams.n_decoder_linear_units);

          // decoder.last_decoder.feed_forward2
          model.decoder.d_last_mlp_ln_w2 = ggml_new_tensor_2d(
              model.ctx, wtype, hparams.n_decoder_linear_units,
              hparams.n_decoder_hidden_state);

          // decoder.last_decoder.feed_forward.norm
          model.decoder.d_last_mlp_norm_w = ggml_new_tensor_1d(
              model.ctx, wtype, hparams.n_decoder_linear_units);
          model.decoder.d_last_mlp_norm_b = ggml_new_tensor_1d(
              model.ctx, wtype, hparams.n_decoder_linear_units);

          // decoder.last_decoder.norm1.weight
          model.decoder.d_last_norm_w1 = ggml_new_tensor_1d(
              model.ctx, wtype, hparams.n_decoder_hidden_state);
          model.decoder.d_last_norm_b1 = ggml_new_tensor_1d(
              model.ctx, wtype, hparams.n_decoder_hidden_state);

          // decoder.last_decoder.norm2.weight
          model.decoder.d_last_norm_w2 = ggml_new_tensor_1d(
              model.ctx, wtype, hparams.n_decoder_hidden_state);
          model.decoder.d_last_norm_b2 = ggml_new_tensor_1d(
              model.ctx, wtype, hparams.n_decoder_hidden_state);

          // decoder.last_decoder.norm3.weight
          model.decoder.d_last_norm_w3 = ggml_new_tensor_1d(
              model.ctx, wtype, hparams.n_decoder_hidden_state);
          model.decoder.d_last_norm_b3 = ggml_new_tensor_1d(
              model.ctx, wtype, hparams.n_decoder_hidden_state);

          model.tensors["decoder.last_decoder.self_attn.fsmn_block.weight"] =
              model.decoder.d_last_attn_fsmn_w;

          model.tensors["decoder.last_decoder.src_attn.linear_q.weight"] =
              model.decoder.d_last_src_attn_ln_q_w;
          model.tensors["decoder.last_decoder.src_attn.linear_q.bias"] =
              model.decoder.d_last_src_attn_ln_q_b;

          model.tensors["decoder.last_decoder.src_attn.linear_k_v.weight"] =
              model.decoder.d_last_src_attn_ln_kv_w;
          model.tensors["decoder.last_decoder.src_attn.linear_k_v.bias"] =
              model.decoder.d_last_src_attn_ln_kv_b;

          model.tensors["decoder.last_decoder.src_attn.linear_out.weight"] =
              model.decoder.d_last_src_attn_ln_o_w;
          model.tensors["decoder.last_decoder.src_attn.linear_out.bias"] =
              model.decoder.d_last_src_attn_ln_o_b;

          model.tensors["decoder.last_decoder.feed_forward.w_1.weight"] =
              model.decoder.d_last_mlp_ln_w1;
          model.tensors["decoder.last_decoder.feed_forward.w_1.bias"] =
              model.decoder.d_last_mlp_ln_b1;

          model.tensors["decoder.last_decoder.feed_forward.w_2.weight"] =
              model.decoder.d_last_mlp_ln_w2;

          model.tensors["decoder.last_decoder.feed_forward.norm.weight"] =
              model.decoder.d_last_mlp_norm_w;
          model.tensors["decoder.last_decoder.feed_forward.norm.bias"] =
              model.decoder.d_last_mlp_norm_b;

          model.tensors["decoder.last_decoder.norm1.weight"] =
              model.decoder.d_last_norm_w1;
          model.tensors["decoder.last_decoder.norm1.bias"] =
              model.decoder.d_last_norm_b1;

          model.tensors["decoder.last_decoder.norm2.weight"] =
              model.decoder.d_last_norm_w2;
          model.tensors["decoder.last_decoder.norm2.bias"] =
              model.decoder.d_last_norm_b2;

          model.tensors["decoder.last_decoder.norm3.weight"] =
              model.decoder.d_last_norm_w3;
          model.tensors["decoder.last_decoder.norm3.bias"] =
              model.decoder.d_last_norm_b3;
        }
      }
      // bias_encoder
      {
        // bias_encoder.ih_l0
        model.bias_encoder.be_ih_l_w =
            ggml_new_tensor_2d(model.ctx, wtype, hparams.n_decoder_hidden_state,
                               hparams.n_decoder_linear_units);
        model.bias_encoder.be_ih_l_b = ggml_new_tensor_1d(
            model.ctx, wtype, hparams.n_decoder_linear_units);

        model.tensors["bias_encoder.weight_ih_l0"] =
            model.bias_encoder.be_ih_l_w;
        model.tensors["bias_encoder.bias_ih_l0"] = model.bias_encoder.be_ih_l_b;

        // bias_encoder.hh_l0
        model.bias_encoder.be_hh_l_w =
            ggml_new_tensor_2d(model.ctx, wtype, hparams.n_decoder_hidden_state,
                               hparams.n_decoder_linear_units);
        model.bias_encoder.be_hh_l_b = ggml_new_tensor_1d(
            model.ctx, wtype, hparams.n_decoder_linear_units);

        model.tensors["bias_encoder.weight_hh_l0"] =
            model.bias_encoder.be_hh_l_w;
        model.tensors["bias_encoder.bias_hh_l0"] = model.bias_encoder.be_hh_l_b;

        // bias_embed.weight
        model.bias_encoder.bias_embed = ggml_new_tensor_2d(
            model.ctx, wtype, hparams.n_decoder_hidden_state, hparams.n_vocab);
        model.tensors["bias_embed.weight"] = model.bias_encoder.bias_embed;
      }

      // predict layers
      {
        // predictor.cif_conv1d
        model.predictor.cif_conv1d_w = ggml_new_tensor_3d(
            model.ctx, wtype, 3, hparams.n_encoder_hidden_state,
            hparams.n_encoder_hidden_state);
        model.predictor.cif_conv1d_b = ggml_new_tensor_1d(
            model.ctx, wtype, hparams.n_encoder_hidden_state);

        model.predictor.cif_ln_out_w = ggml_new_tensor_2d(
            model.ctx, wtype, hparams.n_encoder_hidden_state, 1);
        model.predictor.cif_ln_out_b = ggml_new_tensor_1d(model.ctx, wtype, 1);

        model.tensors["predictor.cif_conv1d.weight"] =
            model.predictor.cif_conv1d_w;
        model.tensors["predictor.cif_conv1d.bias"] =
            model.predictor.cif_conv1d_b;
        model.tensors["predictor.cif_output.weight"] =
            model.predictor.cif_ln_out_w;
        model.tensors["predictor.cif_output.bias"] =
            model.predictor.cif_ln_out_b;
      }
    }

    // load weights
    {
      size_t total_size = 0;

      model.n_loaded = 0;

      while (true) {
        int32_t n_dims;
        int32_t length;
        int32_t ttype;

        read_safe(loader, n_dims);
        read_safe(loader, length);
        read_safe(loader, ttype);

        if (loader->eof(loader->context)) {
          break;
        }

        int32_t nelements = 1;
        int32_t ne[4] = {1, 1, 1, 1};
        for (int i = 0; i < n_dims; ++i) {
          read_safe(loader, ne[i]);
          nelements *= ne[i];
        }

        std::string name;
        std::vector<char> tmp(length);  // create a buffer
        loader->read(loader->context, &tmp[0],
                     tmp.size());  // read to buffer
        name.assign(&tmp[0], tmp.size());

        if (model.tensors.find(name) == model.tensors.end()) {
          PARAFORMER_LOG_INFO("%s: unknown tensor '%s' in model file\n",
                              __func__, name.data());
          return false;
        }

        auto tensor = model.tensors[name.data()];
        if (tensor == nullptr) {
          PARAFORMER_LOG_INFO("%s: tensor '%s' has not init\n", __func__,
                              name.data());
          return false;
        }
        if (ggml_nelements(tensor) != nelements) {
          PARAFORMER_LOG_INFO("%s: tensor '%s' has wrong size in model file\n",
                              __func__, name.data());
          PARAFORMER_LOG_INFO(
              "%s: shape: [%d, %d, %d], expected: [%d, %d, %d]\n", __func__,
              ne[0], ne[1], ne[2], (int)tensor->ne[0], (int)tensor->ne[1],
              (int)tensor->ne[2]);
          return false;
        }

        if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1] ||
            tensor->ne[2] != ne[2]) {
          PARAFORMER_LOG_INFO(
              "%s: tensor '%s' has wrong shape in model file: got "
              "[%d, "
              "%d, %d], expected [%d, %d, %d]\n",
              __func__, name.data(), (int)tensor->ne[0], (int)tensor->ne[1],
              (int)tensor->ne[2], ne[0], ne[1], ne[2]);
          return false;
        }

        const size_t bpe = ggml_type_size(ggml_type(ttype));

        if ((nelements * bpe) / ggml_blck_size(tensor->type) !=
            ggml_nbytes(tensor)) {
          PARAFORMER_LOG_INFO(
              "%s: tensor '%s' has wrong size in model file: got "
              "%zu, "
              "expected %zu\n",
              __func__, name.data(), ggml_nbytes(tensor), nelements * bpe);
          return false;
        }

        loader->read(loader->context, tensor->data, ggml_nbytes(tensor));
        BYTESWAP_TENSOR(tensor);

        // printf("%48s - [%5d, %5d, %5d], type = %6s, %6.2f MB\n",
        // name.data(), ne[0], ne[1], ne[2], ggml_type_name((ggml_type)
        // ttype), ggml_nbytes(tensor)/1024.0/1024.0);
        total_size += ggml_nbytes(tensor);
        model.n_loaded++;
      }

      PARAFORMER_LOG_INFO("%s: model size    = %7.2f MB\n", __func__,
                          total_size / 1024.0 / 1024.0);

      if (model.n_loaded == 0) {
        PARAFORMER_LOG_INFO(
            "%s: WARN no tensors loaded from model file - assuming "
            "empty "
            "model for testing\n",
            __func__);
      } else if (model.n_loaded != (int)model.tensors.size()) {
        PARAFORMER_LOG_INFO(
            "%s: ERROR not all tensors loaded from model file - "
            "expected "
            "%zu, got %d\n",
            __func__, model.tensors.size(), model.n_loaded);
        return false;
      }
    }
  }
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
//         cur = ggml_add(ctx0, ggml_repeat(ctx0, model.e_conv_1_b, cur), cur);
//
//         cur = ggml_gelu(ctx0, cur);
//
//         cur = ggml_conv_1d_ph(ctx0, model.e_conv_2_w, cur, 2, 1);
//         cur = ggml_add(ctx0, ggml_repeat(ctx0, model.e_conv_2_b, cur), cur);
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
  struct paraformer_state *state = new paraformer_state;

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

  // todo bias allocator

  //    {
  //        paraformer_allocr_graph_init(state->alloc_bias_encoder, [&]() {
  //        return paraformer_build_graph_bias(*ctx, *state, 0); });
  //
  //        PARAFORMER_LOG_INFO("%s: compute buffer (conv)   = %7.2f MB\n",
  //        __func__,
  //            PARAFORMER_allocr_size(state->alloc_bias_encoder) / 1024.0 /
  //            1024.0);
  //    }

  // encoder allocator
  {
    state->embd_enc =
        ggml_new_tensor_2d(ctx->model.ctx, GGML_TYPE_F16, 132, 512);
    state->exp_n_audio_ctx = 132;
    paraformer_allocr_graph_init(state->alloc_encode, ctx->backend, [&]() {
      return paraformer_build_graph_encoder(*ctx, *state);
    });

    PARAFORMER_LOG_INFO(
        "%s: compute buffer (encode) = %7.2f MB\n", __func__,
        paraformer_allocr_size(state->alloc_encode) / 1024.0 / 1024.0);
  }

  // todo predict allocator
  {
    //        paraformer_allocr_graph_init(state->alloc_predict,
    //                                     [&]() { return
    //                                     paraformer_build_graph_predict(*ctx,
    //                                     *state); });

    PARAFORMER_LOG_INFO(
        "%s: compute buffer (cross)  = %7.2f MB\n", __func__,
        paraformer_allocr_size(state->alloc_predict) / 1024.0 / 1024.0);
  }

  // todo decoder allocator
  {
    PARAFORMER_LOG_INFO(
        "%s: compute buffer (decode) = %7.2f MB\n", __func__,
        paraformer_allocr_size(state->alloc_decode) / 1024.0 / 1024.0);
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

  // todo   change the parameter 4096
  ggml_cgraph *gf = ggml_new_graph_custom(ctx0, 4096, false);

  struct ggml_tensor *cur = ggml_view_tensor(ctx0, pstate.embd_enc);

  const auto dim = cur->ne[1];
  struct ggml_tensor *position =
      ggml_new_tensor_2d(ctx0, pctx.wtype, n_ctx, dim);

  // SinusoidalPositionEncoder
  // position = concat(sin(10000^(-2i/(dim -2))), cos(10000^(-2i/(dim -2)))
  // ref:
  // https://github.com/alibaba-damo-academy/FunASR/blob/main/funasr/modules/embedding.py#L420-L426

  for (int j = 0; j < n_ctx; j++) {
    for (int k = 0; k < dim; k++) {
      if (k < (dim / 2)) {
        ((float **)position->data)[j][k] =
            sin((j + 1) * pow(10000, (-2 * k) / (dim - 2)));
      } else {
        ((float **)position->data)[j][k] =
            cos((j + 1) * pow(10000, (-2 * (k - 128)) / (dim - 2)));
      }
    }
  }

  cur = ggml_add(ctx0, cur, position);

  for (int il = 0; il < n_layer; ++il) {
    const auto &layer = model.encoder.encoder_layer[il];

    // norm1
    {
      cur = ggml_norm(ctx0, cur, hparams.eps);

      // cur = norm1.weight * cur + norm1 * bias
      cur =
          ggml_add(ctx0, ggml_mul(ctx0, cur, layer.e_norm_w1), layer.e_norm_b1);
    }

    // norm2
    {
      cur = ggml_norm(ctx0, cur, hparams.eps);

      // cur = norm2.weight * cur + norm2 * bias
      cur =
          ggml_add(ctx0, ggml_mul(ctx0, cur, layer.e_norm_w2), layer.e_norm_b2);
    }

    // self-attention
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

    if (!ggml_gallocr_alloc_graph(*alloc, gf)) {
      // should never happen as we pre-allocate the memory
      return false;
    }
  }
}

int paraformer_full_with_state(struct paraformer_context *ctx,
                               struct paraformer_state *state,
                               struct paraformer_hparams params,
                               const paraformer_feature &feature,
                               int n_threads) {
  if (!paraformer_encode_internal(*ctx, *state, n_threads)) {
    fprintf(stderr, "%s: failed to encode\n", __func__);
    return -6;
  }
}
