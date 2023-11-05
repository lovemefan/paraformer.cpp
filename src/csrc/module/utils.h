//
// Created by lovemefan on 2023/10/4.
//

#ifndef PARAFORMER_CPP_UTILS_H
#define PARAFORMER_CPP_UTILS_H

#include <vector>
#include "ggml.h"
#include "ggml-alloc.h"
#include "hparams.h"

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

#if defined(GGML_BIG_ENDIAN)
#include <bit>

template<typename T>
static T byteswap(T value) {
    return std::byteswap(value);
}

template<>
float byteswap(float value) {
    return std::bit_cast<float>(byteswap(std::bit_cast<std::uint32_t>(value)));
}

template<typename T>
static void byteswap_tensor_data(ggml_tensor * tensor) {
    T * datum = reinterpret_cast<T *>(tensor->data);
    for (int i = 0; i < ggml_nelements(tensor); i++) {
        datum[i] = byteswap(datum[i]);
    }
}

static void byteswap_tensor(ggml_tensor * tensor) {
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
        default: { // GML_TYPE_I8
            break;
        }
    }
}

#define BYTESWAP_VALUE(d) d = byteswap(d)
#define BYTESWAP_FILTERS(f)            \
    do {                              \
        for (auto & datum : f.data) { \
            datum = byteswap(datum);  \
        }                             \
    } while (0)
#define BYTESWAP_TENSOR(t)       \
    do {                         \
        byteswap_tensor(t); \
    } while (0)
#else
#define BYTESWAP_VALUE(d) do {} while (0)
#define BYTESWAP_FILTERS(f) do {} while (0)
#define BYTESWAP_TENSOR(t) do {} while (0)
#endif


PARAFORMER_API void paraformer_set_log_callback(paraformer_log_callback callback);

////////////////////////////////////////////////////////////////////////////

// Temporary helpers needed for exposing ggml interface

PARAFORMER_API int          paraformer_bench_memcpy          (int n_threads);
PARAFORMER_API int          paraformer_bench_ggml_mul_mat    (int n_threads);
PARAFORMER_API const char * paraformer_bench_memcpy_str      (int n_threads);
PARAFORMER_API const char * paraformer_bench_ggml_mul_mat_str(int n_threads);


typedef struct paraformer_model_loader {
    void * context;

    size_t (*read)(void * ctx, void * output, size_t read_size);
    bool    (*eof)(void * ctx);
    void  (*close)(void * ctx);
} paraformer_model_loader;


template<typename T>
static void read_safe(paraformer_model_loader * loader, T & dest) {
    loader->read(loader->context, &dest, sizeof(T));
    BYTESWAP_VALUE(dest);
}

// Convert RAW PCM audio to log mel spectrogram.
// The resulting spectrogram is stored inside the default state of the provided paraformer context.
// Returns 0 on success
PARAFORMER_API int paraformer_pcm_to_mel(
        struct paraformer_context * ctx,
        const float * samples,
        int   n_samples,
        int   n_threads);

PARAFORMER_API int paraformer_pcm_to_mel_with_state(
        struct paraformer_context * ctx,
        struct paraformer_state * state,
        const float * samples,
        int   n_samples,
        int   n_threads);

// Convert RAW PCM audio to log mel spectrogram but applies a Phase Vocoder to speed up the audio x2.
// The resulting spectrogram is stored inside the default state of the provided paraformer context.
// Returns 0 on success
PARAFORMER_API int paraformer_pcm_to_mel_phase_vocoder(
        struct paraformer_context * ctx,
        const float * samples,
        int   n_samples,
        int   n_threads);

PARAFORMER_API int paraformer_pcm_to_mel_phase_vocoder_with_state(
        struct paraformer_context * ctx,
        struct paraformer_state * state,
        const float * samples,
        int   n_samples,
        int   n_threads);

// This can be used to set a custom log mel spectrogram inside the default state of the provided paraformer context.
// Use this instead of paraformer_pcm_to_mel() if you want to provide your own log mel spectrogram.
// n_mel must be 80
// Returns 0 on success
PARAFORMER_API int paraformer_set_mel(
        struct paraformer_context * ctx,
        const float * data,
        int   n_len,
        int   n_mel);

PARAFORMER_API int paraformer_set_mel_with_state(
        struct paraformer_context * ctx,
        struct paraformer_state * state,
        const float * data,
        int   n_len,
        int   n_mel);

static void ggml_graph_compute_helper(std::vector<uint8_t> & buf, ggml_cgraph * graph, int n_threads);
static void paraformer_allocr_graph_init(struct paraformer_allocr & allocr, std::function<struct ggml_cgraph *()> && get_graph);

#endif //PARAFORMER_CPP_UTILS_H
