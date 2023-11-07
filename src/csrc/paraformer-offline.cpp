//
// Created by lovemefan on 2023/11/4.
//

#include "paraformer-offline.h"

#include <fstream>
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
#define BYTESWAP_FILTERS(f)          \
    do {                             \
        for (auto &datum : f.data) { \
            datum = byteswap(datum); \
        }                            \
    } while (0)
#define BYTESWAP_TENSOR(t)  \
    do {                    \
        byteswap_tensor(t); \
    } while (0)
#else
#define BYTESWAP_VALUE(d) \
    do {                  \
    } while (0)
#define BYTESWAP_FILTERS(f) \
    do {                    \
    } while (0)
#define BYTESWAP_TENSOR(t) \
    do {                   \
    } while (0)
#endif

#define PARAFORMER_ASSERT(x)                                            \
    do {                                                                \
        if (!(x)) {                                                     \
            log("WHISPER_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x); \
            abort();                                                    \
        }                                                               \
    } while (0)

template <typename T>
static void read_safe(paraformer_model_loader *loader, T &dest) {
    loader->read(loader->context, &dest, sizeof(T));
    BYTESWAP_VALUE(dest);
}

static void log(const char *fmt, ...) {
    char buf[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    fprintf(stderr, "%s", buf);
}

static const size_t MB = 1ull * 1024 * 1024;

// TODO: avoid using GGUF
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

struct paraformer_context *paraformer_init(struct paraformer_model_loader *loader) {
    ggml_time_init();

    paraformer_context *ctx = new paraformer_context;

    if (!paraformer_model_load(loader, *ctx)) {
        loader->close(loader->context);
        log("%s: failed to load model\n", __func__);
        delete ctx;
        return nullptr;
    }

    loader->close(loader->context);

    return ctx;
}

struct paraformer_context *paraformer_init_from_file(const char *path_model) {
    log("%s: loading model from '%s'\n", __func__, path_model);

    auto fin = std::ifstream(path_model, std::ios::binary);
    if (!fin) {
        log("%s: failed to open '%s'\n", __func__, path_model);
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

    if (ctx) {
        ctx->path_model = path_model;
    }

    return ctx;
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
bool paraformer_model_load(struct paraformer_model_loader *loader, paraformer_context &wctx) {
    log("%s: loading model\n", __func__);

    const int64_t t_start_us = ggml_time_us();

    wctx.t_start_us = t_start_us;

    auto &model = wctx.model;
    auto &vocab = wctx.vocab;

    // verify magic
    {
        uint32_t magic;
        read_safe(loader, magic);
        if (magic != GGML_FILE_MAGIC) {
            log("%s: invalid model data (bad magic)\n", __func__);
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
        wctx.wtype = ggml_ftype_to_ggml_type((ggml_ftype)(model.hparams.ftype));
        if (wctx.wtype == GGML_TYPE_COUNT) {
            log("%s: invalid model (bad ftype value %d)\n", __func__, model.hparams.ftype);
            return false;
        }

        const size_t scale = model.hparams.ftype ? 1 : 2;

        log("%s: n_vocab       = %d\n", __func__, hparams.n_vocab);
        log("%s: n_encoder_hidden_state   = %d\n", __func__, hparams.n_encoder_hidden_state);
        log("%s: n_encoder_linear_units = %d\n", __func__, hparams.n_encoder_linear_units);
        log("%s: n_encoder_attention_heads  = %d\n", __func__, hparams.n_encoder_attention_heads);
        log("%s: n_encoder_layers = %d\n", __func__, hparams.n_encoder_layers);
        log("%s: n_decoder_hidden_state    = %d\n", __func__, hparams.n_decoder_hidden_state);
        log("%s: n_decoder_linear_units  = %d\n", __func__, hparams.n_decoder_linear_units);
        log("%s: n_decoder_attention_heads   = %d\n", __func__, hparams.n_decoder_attention_heads);
        log("%s: n_decoder_layers  = %d\n", __func__, hparams.n_decoder_layers);
        log("%s: n_predictor_dim  = %d\n", __func__, hparams.n_predictor_dim);
        log("%s: predictor_tail_threshold  = %f\n", __func__, hparams.predictor_tail_threshold);
        log("%s: n_mels        = %d\n", __func__, hparams.n_mels);
        log("%s: ftype         = %d\n", __func__, model.hparams.ftype);
        log("%s: qntvr         = %d\n", __func__, qntvr);
        log("%s: type          = %d\n", __func__, model.type);

        // print memory requirements
        {
            // TODO
            // log("%s: mem required  = %7.2f MB (+ %7.2f MB per decoder)\n",
            // __func__,
            //        mem_required / 1024.0 / 1024.0, mem_required_decoder /
            //        1024.0 / 1024.0);
        }

        // initialize all memory buffers
        // always have at least one decoder

        wctx.model.buf = new std::vector<uint8_t>();
        wctx.model.buf->resize(scale * MEM_REQ_MODEL.at(wctx.wtype).at(model.type));

        // we skip initialization of the state until it is needed
        // because it might be that state will always be provided externally.
    }

    // load vocab
    {
        int n_vocab = 0;
        read_safe(loader, n_vocab);

        if (n_vocab != model.hparams.n_vocab) {
            log("%s: invalid model file (bad vocab size %d != %d)\n", __func__, n_vocab, model.hparams.n_vocab);
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
                // log("%s: warning: empty-string token in vocab, i = %d\n",
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

    const ggml_type wtype = wctx.wtype;
    const ggml_type vtype = wctx.wtype == GGML_TYPE_F32 ? GGML_TYPE_F32 : GGML_TYPE_F16;  // conv type

    // create the ggml context
    {
        struct ggml_init_params params = {
            /*.mem_size   =*/wctx.model.buf->size(),
            /*.mem_buffer =*/wctx.model.buf->data(),
            /*.no_alloc   =*/false,
        };

        model.ctx = ggml_init(params);
        if (!model.ctx) {
            log("%s: ggml_init() failed\n", __func__);
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
                layer.e_attn_ln_out_w = ggml_new_tensor_2d(model.ctx, wtype, hparams.n_encoder_hidden_state,
                                                           hparams.n_encoder_hidden_state);
                layer.e_attn_ln_out_b = ggml_new_tensor_1d(model.ctx, wtype, hparams.n_encoder_hidden_state);

                // encoder.self_attn.linear_q_k_v.weight
                layer.e_attn_ln_qkv_w = ggml_new_tensor_2d(
                    model.ctx, wtype, i == 0 ? hparams.n_encoder_0_norm_size : hparams.n_encoder_hidden_state,
                    3 * hparams.n_encoder_hidden_state);
                layer.e_attn_ln_qkv_b = ggml_new_tensor_1d(model.ctx, wtype, 3 * hparams.n_encoder_hidden_state);

                // encoder.self_attn.fsmn_block.weight
                layer.e_attn_fsmn_w =
                    ggml_new_tensor_3d(model.ctx, wtype, hparams.fsmn_kernel_size, hparams.n_encoder_hidden_state, 1);

                // encoder.feed_forward.w_1.weight
                layer.e_mlp_w1 = ggml_new_tensor_2d(model.ctx, wtype, hparams.n_encoder_hidden_state,
                                                    hparams.n_encoder_linear_units);
                layer.e_mlp_b1 = ggml_new_tensor_1d(model.ctx, wtype, hparams.n_encoder_linear_units);

                // encoder.feed_forward.w_2.weight
                layer.e_mlp_w2 = ggml_new_tensor_2d(model.ctx, wtype, hparams.n_encoder_linear_units,
                                                    hparams.n_encoder_hidden_state);
                layer.e_mlp_b2 = ggml_new_tensor_1d(model.ctx, wtype, hparams.n_encoder_hidden_state);

                // encoder.norm1.weight
                layer.e_norm_w1 = ggml_new_tensor_1d(
                    model.ctx, wtype, i == 0 ? hparams.n_encoder_0_norm_size : hparams.n_encoder_hidden_state);
                layer.e_norm_b1 = ggml_new_tensor_1d(
                    model.ctx, wtype, i == 0 ? hparams.n_encoder_0_norm_size : hparams.n_encoder_hidden_state);

                // encoder.norm2.weight
                layer.e_norm_w2 = ggml_new_tensor_1d(model.ctx, wtype, hparams.n_encoder_hidden_state);
                layer.e_norm_b2 = ggml_new_tensor_1d(model.ctx, wtype, hparams.n_encoder_hidden_state);

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

            model.encoder.e_after_norm_w = ggml_new_tensor_1d(model.ctx, wtype, hparams.n_encoder_hidden_state);
            model.encoder.e_after_norm_b = ggml_new_tensor_1d(model.ctx, wtype, hparams.n_encoder_hidden_state);
            model.tensors["encoder.after_norm.weight"] = model.encoder.e_after_norm_w;
            model.tensors["encoder.after_norm.bias"] = model.encoder.e_after_norm_b;

            // decoder layers

            {
                model.decoder.embed =
                    ggml_new_tensor_2d(model.ctx, wtype, hparams.n_decoder_hidden_state, hparams.n_vocab);

                model.tensors["decoder.embed.0.weight"] = model.decoder.embed;

                // decoder layers
                for (int i = 0; i < hparams.n_decoder_layers; ++i) {
                    auto &layer = model.decoder.decoder_layers[i];

                    // decoder(i).self_attn.fsmn_block
                    layer.d_attn_fsmn_w = ggml_new_tensor_3d(model.ctx, wtype, hparams.fsmn_kernel_size,
                                                             hparams.n_decoder_hidden_state, 1);

                    // decoder(i).src_attn.linear_q
                    layer.d_src_attn_ln_q_w = ggml_new_tensor_2d(model.ctx, wtype, hparams.n_decoder_hidden_state,
                                                                 hparams.n_decoder_hidden_state);
                    layer.d_src_attn_ln_q_b = ggml_new_tensor_1d(model.ctx, wtype, hparams.n_decoder_hidden_state);

                    // decoder(i).src_attn.linear_k_v
                    layer.d_src_attn_ln_kv_w = ggml_new_tensor_2d(model.ctx, wtype, hparams.n_decoder_hidden_state,
                                                                  2 * hparams.n_decoder_hidden_state);
                    layer.d_src_attn_ln_kv_b = ggml_new_tensor_1d(model.ctx, wtype, 2 * hparams.n_decoder_hidden_state);

                    // decoder(i).src_attn.linear_out
                    layer.d_src_attn_ln_o_w = ggml_new_tensor_2d(model.ctx, wtype, hparams.n_decoder_hidden_state,
                                                                 hparams.n_decoder_hidden_state);
                    layer.d_src_attn_ln_o_b = ggml_new_tensor_1d(model.ctx, wtype, hparams.n_decoder_hidden_state);

                    // decoder(i).feed_forward1
                    layer.d_mlp_ln_w1 = ggml_new_tensor_2d(model.ctx, wtype, hparams.n_decoder_hidden_state,
                                                           hparams.n_decoder_linear_units);
                    layer.d_mlp_ln_b1 = ggml_new_tensor_1d(model.ctx, wtype, hparams.n_decoder_linear_units);

                    // decoder(i).feed_forward2
                    layer.d_mlp_ln_w2 = ggml_new_tensor_2d(model.ctx, wtype, hparams.n_decoder_linear_units,
                                                           hparams.n_decoder_hidden_state);

                    // decoder(i).feed_forward.norm
                    layer.d_mlp_norm_w = ggml_new_tensor_1d(model.ctx, wtype, hparams.n_decoder_linear_units);
                    layer.d_mlp_norm_b = ggml_new_tensor_1d(model.ctx, wtype, hparams.n_decoder_linear_units);

                    // decoder.norm1.weight
                    layer.d_norm_w1 = ggml_new_tensor_1d(model.ctx, wtype, hparams.n_decoder_hidden_state);
                    layer.d_norm_b1 = ggml_new_tensor_1d(model.ctx, wtype, hparams.n_decoder_hidden_state);

                    // decoder.norm2.weight
                    layer.d_norm_w2 = ggml_new_tensor_1d(model.ctx, wtype, hparams.n_decoder_hidden_state);
                    layer.d_norm_b2 = ggml_new_tensor_1d(model.ctx, wtype, hparams.n_decoder_hidden_state);

                    // decoder.norm3.weight
                    layer.d_norm_w3 = ggml_new_tensor_1d(model.ctx, wtype, hparams.n_decoder_hidden_state);
                    layer.d_norm_b3 = ggml_new_tensor_1d(model.ctx, wtype, hparams.n_decoder_hidden_state);

                    model.tensors["decoder.decoders." + std::to_string(i) + ".self_attn.fsmn_block.weight"] =
                        layer.d_attn_fsmn_w;

                    model.tensors["decoder.decoders." + std::to_string(i) + ".src_attn.linear_q.weight"] =
                        layer.d_src_attn_ln_q_w;
                    model.tensors["decoder.decoders." + std::to_string(i) + ".src_attn.linear_q.bias"] =
                        layer.d_src_attn_ln_q_b;

                    model.tensors["decoder.decoders." + std::to_string(i) + ".src_attn.linear_k_v.weight"] =
                        layer.d_src_attn_ln_kv_w;
                    model.tensors["decoder.decoders." + std::to_string(i) + ".src_attn.linear_k_v.bias"] =
                        layer.d_src_attn_ln_kv_b;

                    model.tensors["decoder.decoders." + std::to_string(i) + ".src_attn.linear_out.weight"] =
                        layer.d_src_attn_ln_o_w;
                    model.tensors["decoder.decoders." + std::to_string(i) + ".src_attn.linear_out.bias"] =
                        layer.d_src_attn_ln_o_b;

                    model.tensors["decoder.decoders." + std::to_string(i) + ".feed_forward.w_1.weight"] =
                        layer.d_mlp_ln_w1;
                    model.tensors["decoder.decoders." + std::to_string(i) + ".feed_forward.w_1.bias"] =
                        layer.d_mlp_ln_b1;

                    model.tensors["decoder.decoders." + std::to_string(i) + ".feed_forward.w_2.weight"] =
                        layer.d_mlp_ln_w2;

                    model.tensors["decoder.decoders." + std::to_string(i) + ".feed_forward.norm.weight"] =
                        layer.d_mlp_norm_w;
                    model.tensors["decoder.decoders." + std::to_string(i) + ".feed_forward.norm.bias"] =
                        layer.d_mlp_norm_b;

                    model.tensors["decoder.decoders." + std::to_string(i) + ".norm1.weight"] = layer.d_norm_w1;
                    model.tensors["decoder.decoders." + std::to_string(i) + ".norm1.bias"] = layer.d_norm_b1;

                    model.tensors["decoder.decoders." + std::to_string(i) + ".norm2.weight"] = layer.d_norm_w2;
                    model.tensors["decoder.decoders." + std::to_string(i) + ".norm2.bias"] = layer.d_norm_b2;

                    model.tensors["decoder.decoders." + std::to_string(i) + ".norm3.weight"] = layer.d_norm_w3;
                    model.tensors["decoder.decoders." + std::to_string(i) + ".norm3.bias"] = layer.d_norm_b3;
                }
                // decoder after norm
                model.decoder.d_after_norm_w = ggml_new_tensor_1d(model.ctx, wtype, hparams.n_encoder_hidden_state);
                model.decoder.d_after_norm_b = ggml_new_tensor_1d(model.ctx, wtype, hparams.n_encoder_hidden_state);

                model.tensors["decoder.after_norm.weight"] = model.decoder.d_after_norm_w;
                model.tensors["decoder.after_norm.bias"] = model.decoder.d_after_norm_b;

                // decofer output layer
                model.decoder.d_output_w =
                    ggml_new_tensor_2d(model.ctx, wtype, hparams.n_encoder_hidden_state, hparams.n_vocab);
                model.decoder.d_output_b = ggml_new_tensor_1d(model.ctx, wtype, hparams.n_vocab);

                model.tensors["decoder.output_layer.weight"] = model.decoder.d_output_w;
                model.tensors["decoder.output_layer.bias"] = model.decoder.d_output_b;
            }

            // predict layers
            {}

            //        log("%s: model ctx     = %7.2f MB\n", __func__,
            //            ctx_size / (1024.0 * 1024.0));
        }

        // prepare memory for the weights
        {
            auto &ctx = model.ctx;

            const auto &hparams = model.hparams;

            const int n_vocab = hparams.n_vocab;

            const int n_mels = hparams.n_mels;

            // encoder
            {

            }

            // decoder
            {}
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
                    log("%s: unknown tensor '%s' in model file\n", __func__, name.data());
                    return false;
                }

                auto tensor = model.tensors[name.data()];
                if (tensor == nullptr) {
                    log("%s: tensor '%s' has not init\n", __func__, name.data());
                    return false;
                }
                if (ggml_nelements(tensor) != nelements) {
                    log("%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
                    log("%s: shape: [%d, %d, %d], expected: [%d, %d, %d]\n", __func__, ne[0], ne[1], ne[2],
                        (int)tensor->ne[0], (int)tensor->ne[1], (int)tensor->ne[2]);
                    return false;
                }

                if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1] || tensor->ne[2] != ne[2]) {
                    log("%s: tensor '%s' has wrong shape in model file: got "
                        "[%d, "
                        "%d, %d], expected [%d, %d, %d]\n",
                        __func__, name.data(), (int)tensor->ne[0], (int)tensor->ne[1], (int)tensor->ne[2], ne[0], ne[1],
                        ne[2]);
                    return false;
                }

                const size_t bpe = ggml_type_size(ggml_type(ttype));

                if ((nelements * bpe) / ggml_blck_size(tensor->type) != ggml_nbytes(tensor)) {
                    log("%s: tensor '%s' has wrong size in model file: got "
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

            log("%s: model size    = %7.2f MB\n", __func__, total_size / 1024.0 / 1024.0);

            if (model.n_loaded == 0) {
                log("%s: WARN no tensors loaded from model file - assuming "
                    "empty "
                    "model for testing\n",
                    __func__);
            } else if (model.n_loaded != (int)model.tensors.size()) {
                log("%s: ERROR not all tensors loaded from model file - "
                    "expected "
                    "%zu, got %d\n",
                    __func__, model.tensors.size(), model.n_loaded);
                return false;
            }
        }

        wctx.t_load_us = ggml_time_us() - t_start_us;

        return true;
    }
}