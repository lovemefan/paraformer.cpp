//
// Created by lovemefan on 2024/3/23.
//

#ifndef PARAFORMER_CPP_PARAFORMER_OFFLINE_H
#define PARAFORMER_CPP_PARAFORMER_OFFLINE_H

#include <ggml.h>

#include "paraformer-frontend.h"
#ifdef __GNUC__
#define PARAFORMER_DEPRECATED(func, hint) func __attribute__((deprecated(hint)))
#elif defined(_MSC_VER)
#define PARAFORMER_DEPRECATED(func, hint) __declspec(deprecated(hint)) func
#else
#define PARAFORMER_DEPRECATED(func, hint) func
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

#define PARAFORMER_MAX_NODES 4096

#ifdef __cplusplus
extern "C" {
#endif
//
// C interface
//
// The following interface is thread-safe as long as the sample
// paraformer_context is not used by multiple threads concurrently.
//
// Basic usage:
//
//     #include "paraformer.h"
//
//     ...
//
//     paraformer_context_params cparams = paraformer_context_default_params();
//
//     struct paraformer_context * ctx =
//     paraformer_init_from_file_with_params("/path/to/ggml-base.en.bin",
//     cparams);
//
//     if (paraformer_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
//         fprintf(stderr, "failed to process audio\n");
//         return 7;
//     }
//
//     const int n_segments = paraformer_full_n_segments(ctx);
//     for (int i = 0; i < n_segments; ++i) {
//         const char * text = paraformer_full_get_segment_text(ctx, i);
//         printf("%s", text);
//     }
//
//     paraformer_free(ctx);
//
//     ...
//
// This is a demonstration of the most straightforward usage of the library.
// "pcmf32" contains the RAW audio data in 32-bit floating point format.
//
// The interface also allows for more fine-grained control over the computation,
// but it requires a deeper understanding of how the model works.
//

struct paraformer_context;
struct paraformer_state;
struct paraformer_full_params;

// available paraformer models
enum e_model {
  MODEL_ONLINE,
  MODEL_OFFLINE,
  MODEL_CONTEXTUAL_OFFLINE,
  MODEL_SEACO_OFFLINE
};

struct paraformer_context_params {
  bool use_gpu;
  int gpu_device;  // CUDA device
};

// Available sampling strategies
enum paraformer_decoding_strategy {
  PARAFORMER_SAMPLING_GREEDY,       // similar to OpenAI's GreedyDecoder
  PARAFORMER_SAMPLING_BEAM_SEARCH,  // similar to OpenAI's BeamSearchDecoder
};

// Progress callback
typedef void (*paraformer_progress_callback)(struct paraformer_context *ctx,
                                             struct paraformer_state *state,
                                             int progress, void *user_data);

// Parameters for the paraformer_full() function
struct paraformer_full_params {
  enum paraformer_decoding_strategy strategy;
  int n_threads;
  int n_max_text_ctx;  // max tokens to use from past text as prompt for the
                       // decoder
  int offset_ms;       // start offset in ms
  int duration_ms;     // audio duration to process in ms

  bool no_timestamps;     // do not generate timestamps
  bool single_segment;    // force single segment output (useful for streaming)
  bool print_progress;    // print progress information
  bool print_timestamps;  // print timestamps for each text segment when
                          // printing realtime

  bool debug_mode;  // enable debug_mode provides extra info (eg. Dump log_mel)

  struct {
    int best_of;
  } greedy;

  struct {
    int beam_size;
  } beam_search;

  // called on each progress update
  paraformer_progress_callback progress_callback;
  void *progress_callback_user_data;
};

struct paraformer_model_loader {
  void *context;
  size_t (*read)(void *ctx, void *output, size_t read_size);
  bool (*eof)(void *ctx);
  void (*close)(void *ctx);
};

bool paraformer_model_load(const char *path_model, paraformer_context &wctx);

// Various functions for loading a ggml paraformer model.
// Allocate (almost) all memory needed for the model.
// Return NULL on failure

PARAFORMER_API struct paraformer_context_params
paraformer_context_default_params();
PARAFORMER_API struct paraformer_context *paraformer_init_from_file_with_params(
    const char *path_model, struct paraformer_context_params params);
PARAFORMER_API struct paraformer_context *paraformer_init_from_file(
    const char *path_model);
PARAFORMER_API struct paraformer_context *paraformer_init_from_buffer(
    void *buffer, size_t buffer_size);
PARAFORMER_API struct paraformer_context *paraformer_init(
    struct gguf_context *loader);
PARAFORMER_API struct paraformer_state *paraformer_init_state(
    paraformer_context *ctx);
PARAFORMER_API struct ggml_cgraph *paraformer_build_graph_bias_encoder(
    paraformer_context &wctx, paraformer_state &wstate);
PARAFORMER_API struct ggml_cgraph *paraformer_build_graph_encoder(
    paraformer_context &wctx, paraformer_state &wstate);
PARAFORMER_API struct ggml_cgraph *paraformer_build_graph_predict(
    paraformer_context &wctx, paraformer_state &wstate);
PARAFORMER_API struct ggml_cgraph *paraformer_build_graph_decoder(
    paraformer_context &wctx, paraformer_state &wstate);

// Frees all allocated memory
PARAFORMER_API void paraformer_free(struct paraformer_context *ctx);
PARAFORMER_API void paraformer_free_params(
    struct paraformer_full_params *params);

#ifdef __cplusplus
}
#endif

#endif  // PARAFORMER_CPP_PARAFORMER_OFFLINE_H
