//
// Created by lovemefan on 2023/10/4.
//

#include "hparams.h"

#define PARAFORMER_ASSERT(x) \
    do { \
        if (!(x)) { \
            log("PARAFORMER_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x); \
            abort(); \
        } \
    } while (0)
static bool kv_cache_init(
        const struct paraformer_hparams & hparams,
        struct paraformer_kv_cache & cache,
        ggml_type   wtype,
        int   n_ctx) {
    const int16_t n_decoder_hidden_state = hparams.n_decoder_hidden_state;
    const int8_t n_decoder_layers = hparams.n_decoder_layers;

    const int32_t n_mem      = n_decoder_layers*n_ctx;
    const int64_t n_elements = n_decoder_hidden_state*n_mem;

    const size_t mem_bytes = 2*(ggml_type_size(wtype)*n_elements + ggml_tensor_overhead());

    cache.buf.resize(mem_bytes);

    struct ggml_init_params params = {
            /*.mem_size   =*/ cache.buf.size(),
            /*.mem_buffer =*/ cache.buf.data(),
            /*.no_alloc   =*/ false,
    };

    cache.ctx = ggml_init(params);

    if (!cache.ctx) {
        log("%s: failed to allocate memory for kv cache\n", __func__);
        return false;
    }

    cache.k = ggml_new_tensor_1d(cache.ctx, wtype, n_elements);
    cache.v = ggml_new_tensor_1d(cache.ctx, wtype, n_elements);

    return true;
}

static bool kv_cache_reinit(struct paraformer_kv_cache & cache) {
    PARAFORMER_ASSERT(cache.ctx);

    const int n_elements = ggml_nelements(cache.k);
    PARAFORMER_ASSERT(n_elements == ggml_nelements(cache.v));

    const ggml_type wtype = cache.k->type;
    PARAFORMER_ASSERT(wtype == cache.v->type);

    PARAFORMER_ASSERT(cache.buf.size() >= 2*n_elements*ggml_type_sizef(wtype));

    struct ggml_init_params params = {
            /*.mem_size   =*/ cache.buf.size(),
            /*.mem_buffer =*/ cache.buf.data(),
            /*.no_alloc   =*/ false,
    };

    cache.ctx = ggml_init(params);

    if (!cache.ctx) {
        log("%s: failed to allocate memory for kv cache\n", __func__);
        return false;
    }

    cache.k = ggml_new_tensor_1d(cache.ctx, wtype, n_elements);
    cache.v = ggml_new_tensor_1d(cache.ctx, wtype, n_elements);

    return true;
}

static void kv_cache_free(struct paraformer_kv_cache & cache) {
    if (cache.ctx) {
        ggml_free(cache.ctx);
        cache.ctx = nullptr;
    }
}