//
// Created by lovemefan on 2023/9/20.
//
#include "paraformer.h"
#ifdef PARAFORMER_USE_COREML
#include "../coreml/paraformer-encoder.h"
#endif

#ifdef GGML_USE_METAL
#  include "ggml-metal.h"
#endif


#include <algorithm>
#include <cassert>
#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdio>
#include <cstdarg>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <vector>
#include <regex>
#include <random>
#include <functional>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif


// define this to enable verbose trace logging - useful for debugging purposes
//#define PARAFORMER_DEBUG

#if defined(PARAFORMER_DEBUG)
#define PARAFORMER_PRINT_DEBUG(...) \
    do { \
        fprintf(stderr, __VA_ARGS__); \
    } while (0)
#else
#define PARAFORMER_PRINT_DEBUG(...)
#endif




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
static bool paraformer_model_load(struct paraformer_model_loader * loader, paraformer_context & wctx) {
    log("%s: loading model\n", __func__);

    const int64_t t_start_us = ggml_time_us();

    wctx.t_start_us = t_start_us;

    auto & model = wctx.model;
    auto & vocab = wctx.vocab;

    // verify magic
    {
        uint32_t magic;
        read_safe(loader, magic);
        if (magic != GGML_FILE_MAGIC) {
            log("%s: invalid model data (bad magic)\n", __func__);
            return false;
        }
    }

    //load hparams
    {
        auto & hparams = model.hparams;

        read_safe(loader, hparams.n_vocab);
        read_safe(loader, hparams.n_encoder_hidden_state);
        read_safe(loader, hparams.n_encoder_linear_units);
        read_safe(loader, hparams.n_encoder_attention_heads);
        read_safe(loader, hparams.n_encoder_layers);
        read_safe(loader, hparams.n_decoder_hidden_state);
        read_safe(loader, hparams.n_decoder_linear_units);
        read_safe(loader, hparams.n_decoder_attention_heads);
        read_safe(loader, hparams.n_decoder_layers);
        read_safe(loader, hparams.n_predictor_dim);
        read_safe(loader, hparams.predictor_tail_threshold);
        read_safe(loader, hparams.n_mels);
        read_safe(loader, hparams.ftype);

        assert(hparams.n_decoder_hidden_state == hparams.n_encoder_hidden_state);
        model.type = hparams.model_type;

        const int32_t qntvr = hparams.ftype / GGML_QNT_VERSION_FACTOR;

        hparams.ftype %= GGML_QNT_VERSION_FACTOR;

        // for the big tensors, we have the option to store the data in 16-bit floats or quantized
        // in order to save memory and also to speed up the computation
        wctx.wtype = ggml_ftype_to_ggml_type((ggml_ftype) (model.hparams.ftype));
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
            //log("%s: mem required  = %7.2f MB (+ %7.2f MB per decoder)\n", __func__,
            //        mem_required / 1024.0 / 1024.0, mem_required_decoder / 1024.0 / 1024.0);
        }

        // initialize all memory buffers
        // always have at least one decoder

        wctx.model.buf = new std::vector<uint8_t>();
        wctx.model.buf->resize(scale*MEM_REQ_MODEL.at(wctx.wtype).at(model.type));

        // we skip initialization of the state until it is needed
        // because it might be that state will always be provided externally.
    }


    // load vocab
    {
        int16_t n_vocab = 0;
        read_safe(loader, n_vocab);

        if (n_vocab != model.hparams.n_vocab) {
            log("%s: invalid model file (bad vocab size %d != %d)\n",
                    __func__, n_vocab, model.hparams.n_vocab);
            return false;
        }

        std::string word;
        std::vector<char> tmp;

        tmp.reserve(128);

        for (int16_t i = 0; i < n_vocab; i++) {
            uint8_t len;
            read_safe(loader, len);

            tmp.resize(len);
            loader->read(loader->context, &tmp[0], tmp.size()); // read to buffer
            word.assign(&tmp[0], tmp.size());

            vocab.token_to_id[word] = i;
            vocab.id_to_token[i] = word;

            log("%s: vocab[%d] = '%s'\n", __func__, i, word.c_str());
        }

        vocab.n_vocab = model.hparams.n_vocab;
    }

    size_t ctx_size = 0;

    const ggml_type wtype = wctx.wtype;
    const ggml_type vtype = wctx.wtype == GGML_TYPE_F32 ? GGML_TYPE_F32 : GGML_TYPE_F16; // conv type

//    {
//        const auto & hparams = model.hparams;
//
//        const int n_vocab = hparams.n_vocab;
//
//        const int8_t n_mels = hparams.n_mels;
//
//        // encoder
//        {
//            ctx_size += n_audio_ctx*n_audio_state*ggml_type_sizef(GGML_TYPE_F32); // e_pe;
//
//            ctx_size += 3*n_mels*n_audio_state*ggml_type_sizef(vtype);         // e_conv_1_w
//            ctx_size +=          n_audio_state*ggml_type_sizef(GGML_TYPE_F32); // e_conv_1_b
//
//            ctx_size += 3*n_audio_state*n_audio_state*ggml_type_sizef(vtype);         // e_conv_2_w
//            ctx_size +=                 n_audio_state*ggml_type_sizef(GGML_TYPE_F32); // e_conv_2_b
//
//            ctx_size += n_audio_state*ggml_type_sizef(GGML_TYPE_F32); // e_ln_w;
//            ctx_size += n_audio_state*ggml_type_sizef(GGML_TYPE_F32); // e_ln_b;
//        }
//
//        // decoder
//        {
//            ctx_size += n_text_ctx*n_text_state*ggml_type_sizef(GGML_TYPE_F32); // d_pe;
//
//            ctx_size += n_vocab*n_text_state*ggml_type_sizef(wtype); // d_te;
//
//            ctx_size += n_text_state*ggml_type_sizef(GGML_TYPE_F32); // d_ln_w;
//            ctx_size += n_text_state*ggml_type_sizef(GGML_TYPE_F32); // d_ln_b;
//        }
//
//        // encoder layers
//        {
//            ctx_size += n_audio_layer*(n_audio_state*ggml_type_sizef(GGML_TYPE_F32)); // mlp_ln_w
//            ctx_size += n_audio_layer*(n_audio_state*ggml_type_sizef(GGML_TYPE_F32)); // mlp_ln_b
//
//            ctx_size += n_audio_layer*(4*n_audio_state*n_audio_state*ggml_type_sizef(wtype));         // mlp_0_w
//            ctx_size += n_audio_layer*(              4*n_audio_state*ggml_type_sizef(GGML_TYPE_F32)); // mlp_0_b
//
//            ctx_size += n_audio_layer*(4*n_audio_state*n_audio_state*ggml_type_sizef(wtype));         // mlp_1_w
//            ctx_size += n_audio_layer*(                n_audio_state*ggml_type_sizef(GGML_TYPE_F32)); // mlp_1_b
//
//            ctx_size += n_audio_layer*(n_audio_state*ggml_type_sizef(GGML_TYPE_F32)); // attn_ln_0_w
//            ctx_size += n_audio_layer*(n_audio_state*ggml_type_sizef(GGML_TYPE_F32)); // attn_ln_0_b
//
//            ctx_size += n_audio_layer*(n_audio_state*n_audio_state*ggml_type_sizef(wtype));         // attn_q_w
//            ctx_size += n_audio_layer*(              n_audio_state*ggml_type_sizef(GGML_TYPE_F32)); // attn_q_b
//
//            ctx_size += n_audio_layer*(n_audio_state*n_audio_state*ggml_type_sizef(wtype)); // attn_k_w
//
//            ctx_size += n_audio_layer*(n_audio_state*n_audio_state*ggml_type_sizef(wtype));         // attn_v_w
//            ctx_size += n_audio_layer*(              n_audio_state*ggml_type_sizef(GGML_TYPE_F32)); // attn_v_b
//
//            ctx_size += n_audio_layer*(n_audio_state*n_audio_state*ggml_type_sizef(wtype));         // attn_ln_1_w
//            ctx_size += n_audio_layer*(              n_audio_state*ggml_type_sizef(GGML_TYPE_F32)); // attn_ln_1_b
//        }
//
//        // decoder layers
//        {
//            ctx_size += n_text_layer*(n_text_state*ggml_type_sizef(GGML_TYPE_F32)); // mlp_ln_w
//            ctx_size += n_text_layer*(n_text_state*ggml_type_sizef(GGML_TYPE_F32)); // mlp_ln_b
//
//            ctx_size += n_text_layer*(4*n_text_state*n_text_state*ggml_type_sizef(wtype));         // mlp_0_w
//            ctx_size += n_text_layer*(             4*n_text_state*ggml_type_sizef(GGML_TYPE_F32)); // mlp_0_b
//
//            ctx_size += n_text_layer*(4*n_text_state*n_text_state*ggml_type_sizef(wtype));         // mlp_1_w
//            ctx_size += n_text_layer*(               n_text_state*ggml_type_sizef(GGML_TYPE_F32)); // mlp_1_b
//
//            ctx_size += n_text_layer*(n_text_state*ggml_type_sizef(GGML_TYPE_F32)); // attn_ln_0_w
//            ctx_size += n_text_layer*(n_text_state*ggml_type_sizef(GGML_TYPE_F32)); // attn_ln_0_b
//
//            ctx_size += n_text_layer*(n_text_state*n_text_state*ggml_type_sizef(wtype));         // attn_q_w
//            ctx_size += n_text_layer*(             n_text_state*ggml_type_sizef(GGML_TYPE_F32)); // attn_q_b
//
//            ctx_size += n_text_layer*(n_text_state*n_text_state*ggml_type_sizef(wtype)); // attn_k_w
//
//            ctx_size += n_text_layer*(n_text_state*n_text_state*ggml_type_sizef(wtype));         // attn_v_w
//            ctx_size += n_text_layer*(             n_text_state*ggml_type_sizef(GGML_TYPE_F32)); // attn_v_b
//
//            ctx_size += n_text_layer*(n_text_state*n_text_state*ggml_type_sizef(wtype));         // attn_ln_1_w
//            ctx_size += n_text_layer*(             n_text_state*ggml_type_sizef(GGML_TYPE_F32)); // attn_ln_1_b
//            //
//            ctx_size += n_text_layer*(n_text_state*ggml_type_sizef(GGML_TYPE_F32)); // cross_attn_ln_0_w
//            ctx_size += n_text_layer*(n_text_state*ggml_type_sizef(GGML_TYPE_F32)); // cross_attn_ln_0_b
//
//            ctx_size += n_text_layer*(n_text_state*n_text_state*ggml_type_sizef(wtype));         // cross_attn_q_w
//            ctx_size += n_text_layer*(             n_text_state*ggml_type_sizef(GGML_TYPE_F32)); // cross_attn_q_b
//
//            ctx_size += n_text_layer*(n_text_state*n_text_state*ggml_type_sizef(wtype)); // cross_attn_k_w
//
//            ctx_size += n_text_layer*(n_text_state*n_text_state*ggml_type_sizef(wtype));         // cross_attn_v_w
//            ctx_size += n_text_layer*(             n_text_state*ggml_type_sizef(GGML_TYPE_F32)); // cross_attn_v_b
//
//            ctx_size += n_text_layer*(n_text_state*n_text_state*ggml_type_sizef(wtype));         // cross_attn_ln_1_w
//            ctx_size += n_text_layer*(             n_text_state*ggml_type_sizef(GGML_TYPE_F32)); // cross_attn_ln_1_b
//        }
//
//        ctx_size += (15 + 15*n_audio_layer + 24*n_text_layer)*512; // object overhead
//
//        log("%s: model ctx     = %7.2f MB\n", __func__, ctx_size/(1024.0*1024.0));
//    }

    // create the ggml context
    {
        struct ggml_init_params params = {
                /*.mem_size   =*/ wctx.model.buf->size(),
                /*.mem_buffer =*/ wctx.model.buf->data(),
                /*.no_alloc   =*/ false,
        };

        model.ctx = ggml_init(params);
        if (!model.ctx) {
            log("%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

    // prepare memory for the weights
    {
        auto & ctx = model.ctx;

        const auto & hparams = model.hparams;

        const int n_vocab = hparams.n_vocab;

        const int n_audio_ctx   = hparams.n_audio_ctx;
        const int n_audio_state = hparams.n_audio_state;
        const int n_audio_layer = hparams.n_audio_layer;

        const int n_text_ctx   = hparams.n_text_ctx;
        const int n_text_state = hparams.n_text_state;
        const int n_text_layer = hparams.n_text_layer;

        const int n_mels = hparams.n_mels;



        // decoder
        {
            model.d_pe   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_text_state, n_text_ctx);

            model.d_te   = ggml_new_tensor_2d(ctx, wtype,         n_text_state, n_vocab);

            model.d_ln_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state);
            model.d_ln_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state);

            // map by name
            model.tensors["decoder.positional_embedding"]   = model.d_pe;

            model.tensors["decoder.token_embedding.weight"] = model.d_te;

            model.tensors["decoder.ln.weight"]              = model.d_ln_w;
            model.tensors["decoder.ln.bias"]                = model.d_ln_b;

            for (int i = 0; i < n_text_layer; ++i) {
                auto & layer = model.layers_decoder[i];

                layer.mlp_ln_w          = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_text_state);
                layer.mlp_ln_b          = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_text_state);

                layer.mlp_0_w           = ggml_new_tensor_2d(ctx, wtype,           n_text_state, 4*n_text_state);
                layer.mlp_0_b           = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*n_text_state);

                layer.mlp_1_w           = ggml_new_tensor_2d(ctx, wtype,         4*n_text_state, n_text_state);
                layer.mlp_1_b           = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_text_state);

                layer.attn_ln_0_w       = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_text_state);
                layer.attn_ln_0_b       = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_text_state);

                layer.attn_q_w          = ggml_new_tensor_2d(ctx, wtype,           n_text_state, n_text_state);
                layer.attn_q_b          = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_text_state);

                layer.attn_k_w          = ggml_new_tensor_2d(ctx, wtype,           n_text_state, n_text_state);

                layer.attn_v_w          = ggml_new_tensor_2d(ctx, wtype,           n_text_state, n_text_state);
                layer.attn_v_b          = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_text_state);

                layer.attn_ln_1_w       = ggml_new_tensor_2d(ctx, wtype,           n_text_state, n_text_state);
                layer.attn_ln_1_b       = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_text_state);

                layer.cross_attn_ln_0_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_text_state);
                layer.cross_attn_ln_0_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_text_state);

                layer.cross_attn_q_w    = ggml_new_tensor_2d(ctx, wtype,           n_text_state, n_text_state);
                layer.cross_attn_q_b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_text_state);

                layer.cross_attn_k_w    = ggml_new_tensor_2d(ctx, wtype,           n_text_state, n_text_state);

                layer.cross_attn_v_w    = ggml_new_tensor_2d(ctx, wtype,           n_text_state, n_text_state);
                layer.cross_attn_v_b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_text_state);

                layer.cross_attn_ln_1_w = ggml_new_tensor_2d(ctx, wtype,           n_text_state, n_text_state);
                layer.cross_attn_ln_1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_text_state);

                // map by name
                model.tensors["decoder.blocks." + std::to_string(i) + ".mlp_ln.weight"]           = layer.mlp_ln_w;
                model.tensors["decoder.blocks." + std::to_string(i) + ".mlp_ln.bias"]             = layer.mlp_ln_b;

                model.tensors["decoder.blocks." + std::to_string(i) + ".mlp.0.weight"]            = layer.mlp_0_w;
                model.tensors["decoder.blocks." + std::to_string(i) + ".mlp.0.bias"]              = layer.mlp_0_b;

                model.tensors["decoder.blocks." + std::to_string(i) + ".mlp.2.weight"]            = layer.mlp_1_w;
                model.tensors["decoder.blocks." + std::to_string(i) + ".mlp.2.bias"]              = layer.mlp_1_b;

                model.tensors["decoder.blocks." + std::to_string(i) + ".attn_ln.weight"]          = layer.attn_ln_0_w;
                model.tensors["decoder.blocks." + std::to_string(i) + ".attn_ln.bias"]            = layer.attn_ln_0_b;

                model.tensors["decoder.blocks." + std::to_string(i) + ".attn.query.weight"]       = layer.attn_q_w;
                model.tensors["decoder.blocks." + std::to_string(i) + ".attn.query.bias"]         = layer.attn_q_b;

                model.tensors["decoder.blocks." + std::to_string(i) + ".attn.key.weight"]         = layer.attn_k_w;

                model.tensors["decoder.blocks." + std::to_string(i) + ".attn.value.weight"]       = layer.attn_v_w;
                model.tensors["decoder.blocks." + std::to_string(i) + ".attn.value.bias"]         = layer.attn_v_b;

                model.tensors["decoder.blocks." + std::to_string(i) + ".attn.out.weight"]         = layer.attn_ln_1_w;
                model.tensors["decoder.blocks." + std::to_string(i) + ".attn.out.bias"]           = layer.attn_ln_1_b;

                model.tensors["decoder.blocks." + std::to_string(i) + ".cross_attn_ln.weight"]    = layer.cross_attn_ln_0_w;
                model.tensors["decoder.blocks." + std::to_string(i) + ".cross_attn_ln.bias"]      = layer.cross_attn_ln_0_b;

                model.tensors["decoder.blocks." + std::to_string(i) + ".cross_attn.query.weight"] = layer.cross_attn_q_w;
                model.tensors["decoder.blocks." + std::to_string(i) + ".cross_attn.query.bias"]   = layer.cross_attn_q_b;

                model.tensors["decoder.blocks." + std::to_string(i) + ".cross_attn.key.weight"]   = layer.cross_attn_k_w;

                model.tensors["decoder.blocks." + std::to_string(i) + ".cross_attn.value.weight"] = layer.cross_attn_v_w;
                model.tensors["decoder.blocks." + std::to_string(i) + ".cross_attn.value.bias"]   = layer.cross_attn_v_b;

                model.tensors["decoder.blocks." + std::to_string(i) + ".cross_attn.out.weight"]   = layer.cross_attn_ln_1_w;
                model.tensors["decoder.blocks." + std::to_string(i) + ".cross_attn.out.bias"]     = layer.cross_attn_ln_1_b;
            }
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
            int32_t ne[4] = { 1, 1, 1, 1 };
            for (int i = 0; i < n_dims; ++i) {
                read_safe(loader, ne[i]);
                nelements *= ne[i];
            }

            std::string name;
            std::vector<char> tmp(length); // create a buffer
            loader->read(loader->context, &tmp[0], tmp.size()); // read to buffer
            name.assign(&tmp[0], tmp.size());

            if (model.tensors.find(name) == model.tensors.end()) {
                log("%s: unknown tensor '%s' in model file\n", __func__, name.data());
                return false;
            }

            auto tensor = model.tensors[name.data()];
            if (ggml_nelements(tensor) != nelements) {
                log("%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
                log("%s: shape: [%d, %d, %d], expected: [%d, %d, %d]\n",
                    __func__, ne[0], ne[1], ne[2], (int) tensor->ne[0], (int) tensor->ne[1], (int) tensor->ne[2]);
                return false;
            }

            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1] || tensor->ne[2] != ne[2]) {
                log("%s: tensor '%s' has wrong shape in model file: got [%d, %d, %d], expected [%d, %d, %d]\n",
                    __func__, name.data(), (int) tensor->ne[0], (int) tensor->ne[1], (int) tensor->ne[2], ne[0], ne[1], ne[2]);
                return false;
            }

            const size_t bpe = ggml_type_size(ggml_type(ttype));

            if ((nelements*bpe)/ggml_blck_size(tensor->type) != ggml_nbytes(tensor)) {
                log("%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                    __func__, name.data(), ggml_nbytes(tensor), nelements*bpe);
                return false;
            }

            loader->read(loader->context, tensor->data, ggml_nbytes(tensor));
            BYTESWAP_TENSOR(tensor);

            //printf("%48s - [%5d, %5d, %5d], type = %6s, %6.2f MB\n", name.data(), ne[0], ne[1], ne[2], ggml_type_name((ggml_type) ttype), ggml_nbytes(tensor)/1024.0/1024.0);
            total_size += ggml_nbytes(tensor);
            model.n_loaded++;
        }

        log("%s: model size    = %7.2f MB\n", __func__, total_size/1024.0/1024.0);

        if (model.n_loaded == 0) {
            log("%s: WARN no tensors loaded from model file - assuming empty model for testing\n", __func__);
        } else if (model.n_loaded != (int) model.tensors.size()) {
            log("%s: ERROR not all tensors loaded from model file - expected %zu, got %d\n", __func__, model.tensors.size(), model.n_loaded);
            return false;
        }
    }

    wctx.t_load_us = ggml_time_us() - t_start_us;

    return true;
}

static bool paraformer_encode_external(const paraformer_state & wstate) {
    GGML_UNUSED(wstate);

#ifndef PARAFORMER_USE_COREML
    const bool use_coreml = false;
#else
    const bool use_coreml = wstate.ctx_coreml != nullptr;
#endif

#ifndef PARAFORMER_USE_OPENVINO
    const bool use_openvino = false;
#else
    const bool use_openvino = wstate.ctx_openvino != nullptr;
#endif

    return use_coreml || use_openvino;
}



// evaluate the encoder with the given state
//
// given audio recording (more specifically, its log mel spectrogram), runs forward pass of the encoder
// part of the transformer model and returns the encoded features
//
//   - wctx:      the model
//   - wstate:     the state of the encoder
//   - n_threads:  number of threads to use
//   - mel_offset: offset in the mel spectrogram (i.e. audio offset)
//
static bool paraformer_encode_internal(
        paraformer_context & wctx,
        paraformer_state & wstate,
        const int   mel_offset,
        const int   n_threads) {
    const int64_t t_start_us = ggml_time_us();


    // encoder
    if (!paraformer_encode_external(wstate)) {
        auto & alloc = wstate.alloc_encode.alloc;

        ggml_allocr_reset(alloc);

        ggml_cgraph * gf = paraformer_build_graph_encoder(wctx, wstate);

        ggml_allocr_alloc_graph(alloc, gf);

#ifdef GGML_USE_METAL
        if (wstate.ctx_metal) {
            ggml_metal_set_n_cb     (wstate.ctx_metal, n_threads);
            ggml_metal_graph_compute(wstate.ctx_metal, gf);
        } else {
            ggml_graph_compute_helper(wstate.work_buffer, gf, n_threads);
        }
#else
        ggml_graph_compute_helper(wstate.work_buffer, gf, n_threads);
#endif
    }

    // cross
    {
        auto & alloc = wstate.alloc_cross.alloc;

        ggml_allocr_reset(alloc);

        ggml_cgraph * gf = paraformer_build_graph_cross(wctx, wstate);

        ggml_allocr_alloc_graph(alloc, gf);

#ifdef GGML_USE_METAL
        if (wstate.ctx_metal) {
            ggml_metal_set_n_cb     (wstate.ctx_metal, n_threads);
            ggml_metal_graph_compute(wstate.ctx_metal, gf);
        } else {
            ggml_graph_compute_helper(wstate.work_buffer, gf, n_threads);
        }
#else
        ggml_graph_compute_helper(wstate.work_buffer, gf, n_threads);
#endif
    }

    // ggml_graph_compute_with_ctx(ctx0, &gf, n_threads);

    wstate.t_encode_us += ggml_time_us() - t_start_us;
    wstate.n_encode++;

    return true;
}



// evaluate the decoder
//
// given text prompt + audio features -> computes the logits for the next token
//
//   - model:      the model
//   - n_threads:  number of threads to use
//   - tokens:     text prompt
//   - n_tokens:   number of tokens in the prompt
//   - n_past:     number of past tokens to prefix the prompt with
//
static bool paraformer_decode_internal(
        paraformer_context & wctx,
        paraformer_state & wstate,
        paraformer_decoder & decoder,
        const paraformer_token * tokens,
        const int   n_tokens,
        const int   n_past,
        const int   n_threads) {
    const int64_t t_start_us = ggml_time_us();

    const auto & model   = wctx.model;
    const auto & hparams = model.hparams;

    const int n_vocab = hparams.n_vocab;

    auto & logits_out = wstate.logits;

    struct ggml_tensor * logits;

    // decoder
    {
        auto & alloc = wstate.alloc_decode.alloc;

        ggml_allocr_reset(alloc);

        ggml_cgraph * gf = paraformer_build_graph_decoder(wctx, wstate, decoder, tokens, n_tokens, n_past);

        ggml_allocr_alloc_graph(alloc, gf);

        logits = gf->nodes[gf->n_nodes - 1];

#ifdef GGML_USE_METAL
        if (wstate.ctx_metal) {
            ggml_metal_set_n_cb     (wstate.ctx_metal, n_threads);
            ggml_metal_graph_compute(wstate.ctx_metal, gf);
        } else {
            ggml_graph_compute_helper(wstate.work_buffer, gf, n_threads);
        }
#else
        ggml_graph_compute_helper(wstate.work_buffer, gf, n_threads);
#endif
    }

    // extract logits for all N tokens
    //logits_out.resize(n_tokens*n_vocab);
    //memcpy(logits_out.data(), ggml_get_data(logits), sizeof(float)*n_tokens*n_vocab);

    // extract logits only for the last token
    logits_out.resize(n_vocab);
    memcpy(logits_out.data(), ggml_get_data(logits), sizeof(float)*n_vocab);

    if (n_tokens > 1) {
        //printf("%s: used_mem = %f MB, %f MB, %f MB %f MB %f MB\n", __func__,
        //        ggml_used_mem(ctx0)/1024.0/1024.0,
        //        wstate.get_buf_max_mem(0)/1024.0/1024.0,
        //        wstate.get_buf_max_mem(1)/1024.0/1024.0,
        //        wstate.get_buf_max_mem(2)/1024.0/1024.0,
        //        wstate.get_buf_max_mem(3)/1024.0/1024.0);
    }

    if (n_tokens == 1) {
        wstate.t_decode_us += ggml_time_us() - t_start_us;
        wstate.n_decode++;
    } else {
        wstate.t_prompt_us += ggml_time_us() - t_start_us;
        wstate.n_prompt++;
    }

    return true;
}


//  500 -> 00:05.000
// 6000 -> 01:00.000
static std::string to_timestamp(int64_t t, bool comma = false) {
    int64_t msec = t * 10;
    int64_t hr = msec / (1000 * 60 * 60);
    msec = msec - hr * (1000 * 60 * 60);
    int64_t min = msec / (1000 * 60);
    msec = msec - min * (1000 * 60);
    int64_t sec = msec / 1000;
    msec = msec - sec * 1000;

    char buf[32];
    snprintf(buf, sizeof(buf), "%02d:%02d:%02d%s%03d", (int) hr, (int) min, (int) sec, comma ? "," : ".", (int) msec);

    return std::string(buf);
}


// split text into tokens
//
// ref: https://github.com/openai/gpt-2/blob/a74da5d99abaaba920de8131d64da2862a8f213b/src/encoder.py#L53
//
// Regex (Python):
// r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
//
// Regex (C++):
// R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)"
//
static std::vector<paraformer_vocab::id> tokenize(const paraformer_vocab & vocab, const std::string & text) {
    std::vector<std::string> words;

    // first split the text into words
    {
        std::string str = text;
        std::string pat = R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)";

        std::regex re(pat);
        std::smatch m;

        while (std::regex_search(str, m, re)) {
            for (auto x : m) {
                words.push_back(x);
            }
            str = m.suffix();
        }
    }

    // find the longest tokens that form the words:
    std::vector<paraformer_vocab::id> tokens;
    for (const auto & word : words) {
        if (word.empty()) continue;

        int i = 0;
        int n = word.size();
        while (i < n) {
            int j = n;
            bool found = false;
            while (j > i) {
                auto sub = word.substr(i, j-i);
                auto it = vocab.token_to_id.find(sub);
                if (it != vocab.token_to_id.end()) {
                    tokens.push_back(it->second);
                    i = j;
                    found = true;
                    break;
                }
                --j;
            }
            if (!found) {
                log("unknown token\n");
                ++i;
            }
        }
    }

    return tokens;
}

//
// interface implementation
//

#ifdef PARAFORMER_USE_COREML
// replace .bin with -encoder.mlmodelc
static std::string paraformer_get_coreml_path_encoder(std::string path_bin) {
    auto pos = path_bin.rfind('.');
    if (pos != std::string::npos) {
        path_bin = path_bin.substr(0, pos);
    }

    // match "-qx_x"
    pos = path_bin.rfind('-');
    if (pos != std::string::npos) {
        auto sub = path_bin.substr(pos);
        if (sub.size() == 5 && sub[1] == 'q' && sub[3] == '_') {
            path_bin = path_bin.substr(0, pos);
        }
    }

    path_bin += "-encoder.mlmodelc";

    return path_bin;
}
#endif

#ifdef PARAFORMER_USE_OPENVINO
// replace .bin with-encoder-openvino.xml
static std::string paraformer_openvino_get_path_encoder(std::string path_bin) {
    auto pos = path_bin.rfind('.');
    if (pos != std::string::npos) {
        path_bin = path_bin.substr(0, pos);
    }

    path_bin += "-encoder-openvino.xml";

    return path_bin;
}

static std::string paraformer_openvino_get_path_cache(std::string path_bin) {
    auto pos = path_bin.rfind('.');
    if (pos != std::string::npos) {
        path_bin = path_bin.substr(0, pos);
    }

    path_bin += "-encoder-openvino-cache";

    return path_bin;
}
#endif

struct paraformer_state * paraformer_init_state(paraformer_context * ctx) {
    fill_sin_cos_table();
    paraformer_state * state = new paraformer_state;

    if (!kv_cache_init(ctx->model.hparams, state->decoders[0].kv_self, ctx->itype, ctx->model.hparams.n_text_ctx)) {
        log("%s: kv_cache_init() failed for self-attention cache\n", __func__);
        delete state;
        return nullptr;
    }

    {
        const size_t memory_size = ggml_nbytes(state->decoders[0].kv_self.k) + ggml_nbytes(state->decoders[0].kv_self.v);
        log("%s: kv self size  = %7.2f MB\n", __func__, memory_size / 1024.0 / 1024.0);
    }

    if (!kv_cache_init(ctx->model.hparams, state->kv_cross, ctx->itype, ctx->model.hparams.n_audio_ctx)) {
        log("%s: kv_cache_init() failed for cross-attention cache\n", __func__);
        delete state;
        return nullptr;
    }

    {
        const size_t memory_size = ggml_nbytes(state->kv_cross.k) + ggml_nbytes(state->kv_cross.v);
        log("%s: kv cross size = %7.2f MB\n", __func__, memory_size / 1024.0 / 1024.0);
    }

#ifdef PARAFORMER_USE_COREML
    const auto path_coreml = paraformer_get_coreml_path_encoder(ctx->path_model);

    log("%s: loading Core ML model from '%s'\n", __func__, path_coreml.c_str());
    log("%s: first run on a device may take a while ...\n", __func__);

    state->ctx_coreml = paraformer_coreml_init(path_coreml.c_str());
    if (!state->ctx_coreml) {
        log("%s: failed to load Core ML model from '%s'\n", __func__, path_coreml.c_str());
#ifndef PARAFORMER_COREML_ALLOW_FALLBACK
        delete state;
        return nullptr;
#endif
    } else {
        log("%s: Core ML model loaded\n", __func__);
    }
#endif

    state->logits.reserve(ctx->vocab.n_vocab * ctx->model.hparams.n_text_ctx);

    state->logits_id.reserve(ctx->model.hparams.n_vocab);

    // TAGS: PARAFORMER_DECODER_INIT
    state->decoders[0].sequence.tokens.reserve(ctx->model.hparams.n_text_ctx);

    state->decoders[0].probs.reserve   (ctx->vocab.n_vocab);
    state->decoders[0].logits.reserve  (ctx->vocab.n_vocab);
    state->decoders[0].logprobs.reserve(ctx->vocab.n_vocab);


    // encoder allocator
    if (!paraformer_encode_external(*state)) {
        paraformer_allocr_graph_init(state->alloc_encode,
                                  [&]() {
                                      return paraformer_build_graph_encoder(*ctx, *state);
                                  });

        log("%s: compute buffer (encode) = %7.2f MB\n", __func__, paraformer_allocr_size(state->alloc_encode) / 1024.0 / 1024.0);
    }

    // cross allocator
    {
        paraformer_allocr_graph_init(state->alloc_cross,
                                  [&]() {
                                      return paraformer_build_graph_cross(*ctx, *state);
                                  });

        log("%s: compute buffer (cross)  = %7.2f MB\n", __func__, paraformer_allocr_size(state->alloc_cross) / 1024.0 / 1024.0);
    }

    // decoder allocator
    {
        paraformer_allocr_graph_init(state->alloc_decode,
                                  [&]() {
                                      const auto & hparams = ctx->model.hparams;

                                      // TODO: make sure this is the worst-case scenario
                                      const int n_tokens = hparams.n_text_ctx;
                                      const int n_past   = 0;

                                      return paraformer_build_graph_decoder(*ctx, *state, state->decoders[0], nullptr, n_tokens, n_past);
                                  });

        log("%s: compute buffer (decode) = %7.2f MB\n", __func__, paraformer_allocr_size(state->alloc_decode) / 1024.0 / 1024.0);
    }

#ifdef GGML_USE_METAL
    state->ctx_metal = ggml_metal_init(1);
    if (!state->ctx_metal) {
        log("%s: ggml_metal_init() failed\n", __func__);
        delete state;
        return nullptr;
    }

    log("%s: Metal context initialized\n", __func__);

    // this allocates all Metal resources and memory buffers

    void * data_ptr  = NULL;
    size_t data_size = 0;

    // TODO: add mmap support
    //if (params.use_mmap) {
    //    data_ptr  = ctx->model.mapping->addr;
    //    data_size = ctx->model.mapping->size;
    //} else {
    //    data_ptr  = ggml_get_mem_buffer(ctx->model.ctx);
    //    data_size = ggml_get_mem_size  (ctx->model.ctx);
    //}

    data_ptr  = ggml_get_mem_buffer(ctx->model.ctx);
    data_size = ggml_get_mem_size  (ctx->model.ctx);

    const size_t max_size = ggml_get_max_tensor_size(ctx->model.ctx);

    log("%s: max tensor size = %8.2f MB\n", __func__, max_size/1024.0/1024.0);

#define PARAFORMER_METAL_CHECK_BUF(result)              \
    if (!(result)) {                                 \
        log("%s: failed to add metal buffer\n", __func__); \
        delete state;                                \
        return nullptr;                              \
    }

    PARAFORMER_METAL_CHECK_BUF(ggml_metal_add_buffer(state->ctx_metal, "data", data_ptr, data_size, max_size));

    PARAFORMER_METAL_CHECK_BUF(ggml_metal_add_buffer(state->ctx_metal, "meta_conv",   state->alloc_conv.meta.data(),   state->alloc_conv.meta.size(),   0));
    PARAFORMER_METAL_CHECK_BUF(ggml_metal_add_buffer(state->ctx_metal, "meta_encode", state->alloc_encode.meta.data(), state->alloc_encode.meta.size(), 0));
    PARAFORMER_METAL_CHECK_BUF(ggml_metal_add_buffer(state->ctx_metal, "meta_cross",  state->alloc_cross.meta.data(),  state->alloc_cross.meta.size(),  0));
    PARAFORMER_METAL_CHECK_BUF(ggml_metal_add_buffer(state->ctx_metal, "meta_decode", state->alloc_decode.meta.data(), state->alloc_decode.meta.size(), 0));

    PARAFORMER_METAL_CHECK_BUF(ggml_metal_add_buffer(state->ctx_metal, "data_conv",   state->alloc_conv.data.data(),   state->alloc_conv.data.size(),   0));
    PARAFORMER_METAL_CHECK_BUF(ggml_metal_add_buffer(state->ctx_metal, "data_encode", state->alloc_encode.data.data(), state->alloc_encode.data.size(), 0));
    PARAFORMER_METAL_CHECK_BUF(ggml_metal_add_buffer(state->ctx_metal, "data_cross",  state->alloc_cross.data.data(),  state->alloc_cross.data.size(),  0));
    PARAFORMER_METAL_CHECK_BUF(ggml_metal_add_buffer(state->ctx_metal, "data_decode", state->alloc_decode.data.data(), state->alloc_decode.data.size(), 0));

    PARAFORMER_METAL_CHECK_BUF(ggml_metal_add_buffer(state->ctx_metal, "kv_cross",  state->kv_cross.buf.data(), state->kv_cross.buf.size(), 0));

    PARAFORMER_METAL_CHECK_BUF(ggml_metal_add_buffer(state->ctx_metal, "kv_self_0", state->decoders[0].kv_self.buf.data(), state->decoders[0].kv_self.buf.size(), 0));
#undef PARAFORMER_METAL_CHECK_BUF
#endif

    state->rng = std::mt19937(0);

    return state;
}

int paraformer_ctx_init_openvino_encoder(
        struct paraformer_context * ctx,
        const char * model_path,
        const char * device,
        const char * cache_dir) {
#ifndef PARAFORMER_USE_OPENVINO
    (void)(ctx);
    (void)(model_path);
    (void)(device);
    (void)(cache_dir);

    return 1;
#else
    if (!model_path && ctx->path_model.empty()) {
        log("%s: model_path is nullptr, and ctx has no model_path set.\n", __func__);
        return 1;
    }

    std::string path_encoder;
    if (!model_path) {
        //if model_path is not set, attempt to find it in the same directory as ggml-<model>.bin model
        path_encoder = paraformer_openvino_get_path_encoder(ctx->path_model);
    } else {
        path_encoder = model_path;
    }

    std::string path_cache;
    if (!cache_dir) {
        //if cache_dir is not set, set it as a dir residing next to ggml-<model>.bin
        path_cache = paraformer_openvino_get_path_cache(ctx->path_model);
    } else {
        path_cache = cache_dir;
    }

    log("%s: loading OpenVINO model from '%s'\n", __func__, path_encoder.c_str());
    log("%s: first run on a device may take a while ...\n", __func__);

    ctx->state->ctx_openvino = paraformer_openvino_init(path_encoder.c_str(), device, path_cache.c_str());
    if (!ctx->state->ctx_openvino) {
        log("%s: failed to init OpenVINO encoder from '%s'\n", __func__, path_encoder.c_str());
        return 1;
    } else {
        log("%s: OpenVINO model loaded\n", __func__);
    }

    return 0;
#endif
}

struct paraformer_context * paraformer_init_from_file_no_state(const char * path_model) {
    log("%s: loading model from '%s'\n", __func__, path_model);

    auto fin = std::ifstream(path_model, std::ios::binary);
    if (!fin) {
        log("%s: failed to open '%s'\n", __func__, path_model);
        return nullptr;
    }

    paraformer_model_loader loader = {};

    loader.context = &fin;

    loader.read = [](void * ctx, void * output, size_t read_size) {
        std::ifstream * fin = (std::ifstream*)ctx;
        fin->read((char *)output, read_size);
        return read_size;
    };

    loader.eof = [](void * ctx) {
        std::ifstream * fin = (std::ifstream*)ctx;
        return fin->eof();
    };

    loader.close = [](void * ctx) {
        std::ifstream * fin = (std::ifstream*)ctx;
        fin->close();
    };

    auto ctx = paraformer_init_no_state(&loader);

    if (ctx) {
        ctx->path_model = path_model;
    }

    return ctx;
}

struct paraformer_context * paraformer_init_from_buffer_no_state(void * buffer, size_t buffer_size) {
    struct buf_context {
        uint8_t* buffer;
        size_t size;
        size_t current_offset;
    };

    buf_context ctx = { reinterpret_cast<uint8_t*>(buffer), buffer_size, 0 };

    log("%s: loading model from buffer\n", __func__);

    paraformer_model_loader loader = {};

    loader.context = &ctx;

    loader.read = [](void * ctx, void * output, size_t read_size) {
        buf_context * buf = reinterpret_cast<buf_context *>(ctx);

        size_t size_to_copy = buf->current_offset + read_size < buf->size ? read_size : buf->size - buf->current_offset;

        memcpy(output, buf->buffer + buf->current_offset, size_to_copy);
        buf->current_offset += size_to_copy;

        return size_to_copy;
    };

    loader.eof = [](void * ctx) {
        buf_context * buf = reinterpret_cast<buf_context *>(ctx);

        return buf->current_offset >= buf->size;
    };

    loader.close = [](void * /*ctx*/) { };

    return paraformer_init_no_state(&loader);
}

struct paraformer_context * paraformer_init_no_state(struct paraformer_model_loader * loader) {
    ggml_time_init();

    paraformer_context * ctx = new paraformer_context;

    if (!paraformer_model_load(loader, *ctx)) {
        loader->close(loader->context);
        log("%s: failed to load model\n", __func__);
        delete ctx;
        return nullptr;
    }

    loader->close(loader->context);

    return ctx;
}

struct paraformer_context * paraformer_init_from_file(const char * path_model) {
    paraformer_context * ctx = paraformer_init_from_file_no_state(path_model);
    if (!ctx) {
        return nullptr;
    }

    ctx->state = paraformer_init_state(ctx);
    if (!ctx->state) {
        paraformer_free(ctx);
        return nullptr;
    }

    return ctx;
}

struct paraformer_context * paraformer_init_from_buffer(void * buffer, size_t buffer_size) {
    paraformer_context * ctx = paraformer_init_from_buffer_no_state(buffer, buffer_size);
    if (!ctx) {
        return nullptr;
    }

    ctx->state = paraformer_init_state(ctx);
    if (!ctx->state) {
        paraformer_free(ctx);
        return nullptr;
    }

    return ctx;
}

struct paraformer_context * paraformer_init(struct paraformer_model_loader * loader) {
    paraformer_context * ctx = paraformer_init_no_state(loader);
    if (!ctx) {
        return nullptr;
    }

    ctx->state = paraformer_init_state(ctx);
    if (!ctx->state) {
        paraformer_free(ctx);
        return nullptr;
    }

    return ctx;
}

void paraformer_free_state(struct paraformer_state * state)
{
    if (state) {
        kv_cache_free(state->kv_cross);

        for (int i = 0; i < PARAFORMER_MAX_DECODERS; ++i) {
            kv_cache_free(state->decoders[i].kv_self);
        }

#ifdef PARAFORMER_USE_COREML
        if (state->ctx_coreml != nullptr) {
            paraformer_coreml_free(state->ctx_coreml);
            state->ctx_coreml = nullptr;
        }
#endif

#ifdef GGML_USE_METAL
        if (state->ctx_metal) {
            ggml_metal_free(state->ctx_metal);
            state->ctx_metal = nullptr;
        }
#endif

#ifdef PARAFORMER_USE_OPENVINO
        if (state->ctx_openvino != nullptr) {
            paraformer_openvino_free(state->ctx_openvino);
            state->ctx_openvino = nullptr;
        }
#endif

        paraformer_allocr_free(state->alloc_conv);
        paraformer_allocr_free(state->alloc_decode);
        paraformer_allocr_free(state->alloc_cross);
        paraformer_allocr_free(state->alloc_encode);

        delete state;
    }
}

void paraformer_free(struct paraformer_context * ctx) {
    if (ctx) {
        if (ctx->model.ctx) {
            ggml_free(ctx->model.ctx);
        }
        if (ctx->model.buf) {
            delete ctx->model.buf;
        }

        paraformer_free_state(ctx->state);

        delete ctx;
    }
}

void paraformer_free_params(struct paraformer_full_params * params) {
    if (params) {
        delete params;
    }
}



int paraformer_encode_with_state(struct paraformer_context * ctx, struct paraformer_state * state, int offset, int n_threads) {
    if (!paraformer_encode_internal(*ctx, *state, offset, n_threads)) {
        log("%s: failed to eval\n", __func__);
        return -1;
    }

    return 0;
}

int paraformer_encode(struct paraformer_context * ctx, int offset, int n_threads) {
    if (!paraformer_encode_internal(*ctx, *ctx->state, offset, n_threads)) {
        log("%s: failed to eval\n", __func__);
        return -1;
    }

    return 0;
}

int paraformer_decode_with_state(struct paraformer_context * ctx, struct paraformer_state * state, const paraformer_token * tokens, int n_tokens, int n_past, int n_threads) {
    const int selected_decoder_id = 0;

    if (!paraformer_decode_internal(*ctx, *state, state->decoders[selected_decoder_id], tokens, n_tokens, n_past, n_threads)) {
        log("%s: failed to eval\n", __func__);
        return 1;
    }

    return 0;
}

int paraformer_decode(struct paraformer_context * ctx, const paraformer_token * tokens, int n_tokens, int n_past, int n_threads) {
    // TODO: add selected_decoder_id to state
    const int selected_decoder_id = 0;

    if (ctx->state == nullptr) {
        log("%s: ERROR state was not loaded.\n", __func__);
        return false;
    }

    if (!paraformer_decode_internal(*ctx, *ctx->state, ctx->state->decoders[selected_decoder_id], tokens, n_tokens, n_past, n_threads)) {
        log("%s: failed to eval\n", __func__);
        return 1;
    }

    return 0;
}

int paraformer_tokenize(struct paraformer_context * ctx, const char * text, paraformer_token * tokens, int n_max_tokens) {
    const auto res = tokenize(ctx->vocab, text);

    if (n_max_tokens < (int) res.size()) {
        log("%s: too many resulting tokens: %d (max %d)\n", __func__, (int) res.size(), n_max_tokens);
        return -1;
    }

    for (int i = 0; i < (int) res.size(); i++) {
        tokens[i] = res[i];
    }

    return res.size();
}

int paraformer_lang_max_id() {
    auto max_id = 0;
    for (const auto & kv : g_lang) {
        max_id = std::max(max_id, kv.second.first);
    }

    return max_id;
}

int paraformer_lang_id(const char * lang) {
    if (!g_lang.count(lang)) {
        for (const auto & kv : g_lang) {
            if (kv.second.second == lang) {
                return kv.second.first;
            }
        }

        log("%s: unknown language '%s'\n", __func__, lang);
        return -1;
    }
    return g_lang.at(lang).first;
}

const char * paraformer_lang_str(int id) {
    for (const auto & kv : g_lang) {
        if (kv.second.first == id) {
            return kv.first.c_str();
        }
    }

    log("%s: unknown language id %d\n", __func__, id);
    return nullptr;
}

int paraformer_lang_auto_detect_with_state(
        struct paraformer_context * ctx,
        struct paraformer_state * state,
        int   offset_ms,
        int   n_threads,
        float * lang_probs) {
    const int seek = offset_ms/10;

    if (seek < 0) {
        log("%s: offset %dms is before the start of the audio\n", __func__, offset_ms);
        return -1;
    }

    if (seek >= state->mel.n_len_org) {
        log("%s: offset %dms is past the end of the audio (%dms)\n", __func__, offset_ms, state->mel.n_len_org*10);
        return -2;
    }

    // run the encoder
    if (paraformer_encode_with_state(ctx, state, seek, n_threads) != 0) {
        log("%s: failed to encode\n", __func__);
        return -6;
    }

    const std::vector<paraformer_token> prompt = { paraformer_token_sot(ctx) };

    if (paraformer_decode_with_state(ctx, state, prompt.data(), prompt.size(), 0, n_threads) != 0) {
        log("%s: failed to decode\n", __func__);
        return -7;
    }

    auto & logits_id = state->logits_id;
    logits_id.clear();

    for (const auto & kv : g_lang) {
        const auto token_lang = paraformer_token_lang(ctx, kv.second.first);
        logits_id.emplace_back(state->logits[token_lang], kv.second.first);
    }

    // sort descending
    {
        using pair_type = std::remove_reference<decltype(logits_id)>::type::value_type;
        std::sort(logits_id.begin(), logits_id.end(), [](const pair_type & a, const pair_type & b) {
            return a.first > b.first;
        });
    }

    // softmax
    {
        const auto max = logits_id[0].first;

        double sum = 0.0f;
        for (auto & kv : logits_id) {
            kv.first = exp(kv.first - max);
            sum += kv.first;
        }

        for (auto & kv : logits_id) {
            kv.first /= sum;
        }
    }

    {
        for (const auto & prob : logits_id) {
            if (lang_probs) {
                lang_probs[prob.second] = prob.first;
            }

            //printf("%s: lang %2d (%3s): %f\n", __func__, prob.second, paraformer_lang_str(prob.second), prob.first);
        }
    }

    return logits_id[0].second;
}

int paraformer_lang_auto_detect(
        struct paraformer_context * ctx,
        int   offset_ms,
        int   n_threads,
        float * lang_probs) {
    return paraformer_lang_auto_detect_with_state(ctx, ctx->state, offset_ms, n_threads, lang_probs);
}

int paraformer_model_n_vocab(struct paraformer_context * ctx) {
    return ctx->model.hparams.n_vocab;
}

int paraformer_model_n_audio_ctx(struct paraformer_context * ctx) {
    return ctx->model.hparams.n_audio_ctx;
}

int paraformer_model_n_audio_state(struct paraformer_context * ctx) {
    return ctx->model.hparams.n_audio_state;
}

int paraformer_model_n_audio_head(struct paraformer_context * ctx) {
    return ctx->model.hparams.n_audio_head;
}

int paraformer_model_n_audio_layer(struct paraformer_context * ctx) {
    return ctx->model.hparams.n_audio_layer;
}

int paraformer_model_n_text_ctx(struct paraformer_context * ctx) {
    return ctx->model.hparams.n_text_ctx;
}

int paraformer_model_n_text_state(struct paraformer_context * ctx) {
    return ctx->model.hparams.n_text_state;
}

int paraformer_model_n_text_head(struct paraformer_context * ctx) {
    return ctx->model.hparams.n_text_head;
}

int paraformer_model_n_text_layer(struct paraformer_context * ctx) {
    return ctx->model.hparams.n_text_layer;
}

int paraformer_model_n_mels(struct paraformer_context * ctx) {
    return ctx->model.hparams.n_mels;
}

int paraformer_model_ftype(struct paraformer_context * ctx) {
    return ctx->model.hparams.ftype;
}

int paraformer_model_type(struct paraformer_context * ctx) {
    return ctx->model.type;
}

const char *paraformer_model_type_readable(struct paraformer_context * ctx) {
    switch (ctx->model.type) {
        case e_model::MODEL_TINY:
            return "tiny";
        case e_model::MODEL_BASE:
            return "base";
        case e_model::MODEL_SMALL:
            return "small";
        case e_model::MODEL_MEDIUM:
            return "medium";
        case e_model::MODEL_LARGE:
            return "large";
        default:
            return "unknown";
    }
}

int paraformer_n_len_from_state(struct paraformer_state * state) {
    return state->mel.n_len_org;
}

int paraformer_n_len(struct paraformer_context * ctx) {
    return ctx->state->mel.n_len_org;
}

int paraformer_n_vocab(struct paraformer_context * ctx) {
    return ctx->vocab.n_vocab;
}

int paraformer_n_text_ctx(struct paraformer_context * ctx) {
    return ctx->model.hparams.n_text_ctx;
}

int paraformer_n_audio_ctx(struct paraformer_context * ctx) {
    return ctx->model.hparams.n_audio_ctx;
}

int paraformer_is_multilingual(struct paraformer_context * ctx) {
    return ctx->vocab.is_multilingual() ? 1 : 0;
}

float * paraformer_get_logits(struct paraformer_context * ctx) {
    return ctx->state->logits.data();
}

float * paraformer_get_logits_from_state(struct paraformer_state * state) {
    return state->logits.data();
}

const char * paraformer_token_to_str(struct paraformer_context * ctx, paraformer_token token) {
    return ctx->vocab.id_to_token.at(token).c_str();
}

paraformer_token paraformer_token_eot(struct paraformer_context * ctx) {
    return ctx->vocab.token_eot;
}

paraformer_token paraformer_token_sot(struct paraformer_context * ctx) {
    return ctx->vocab.token_sot;
}


paraformer_token paraformer_token_lang(struct paraformer_context * ctx, int lang_id) {
    return paraformer_token_sot(ctx) + 1 + lang_id;
}

paraformer_token paraformer_token_translate(struct paraformer_context * ctx) {
    return ctx->vocab.token_translate;
}

paraformer_token paraformer_token_transcribe(struct paraformer_context * ctx) {
    return ctx->vocab.token_transcribe;
}

void paraformer_print_timings(struct paraformer_context * ctx) {
    const int64_t t_end_us = ggml_time_us();

    log("\n");
    log("%s:     load time = %8.2f ms\n", __func__, ctx->t_load_us / 1000.0f);
    if (ctx->state != nullptr) {

        const int32_t n_sample = std::max(1, ctx->state->n_sample);
        const int32_t n_encode = std::max(1, ctx->state->n_encode);
        const int32_t n_decode = std::max(1, ctx->state->n_decode);
        const int32_t n_prompt = std::max(1, ctx->state->n_prompt);

        log("%s:     fallbacks = %3d p / %3d h\n", __func__, ctx->state->n_fail_p, ctx->state->n_fail_h);
        log("%s:      mel time = %8.2f ms\n", __func__, ctx->state->t_mel_us / 1000.0f);
        log("%s:   sample time = %8.2f ms / %5d runs (%8.2f ms per run)\n", __func__, 1e-3f * ctx->state->t_sample_us, n_sample, 1e-3f * ctx->state->t_sample_us / n_sample);
        log("%s:   encode time = %8.2f ms / %5d runs (%8.2f ms per run)\n", __func__, 1e-3f * ctx->state->t_encode_us, n_encode, 1e-3f * ctx->state->t_encode_us / n_encode);
        log("%s:   decode time = %8.2f ms / %5d runs (%8.2f ms per run)\n", __func__, 1e-3f * ctx->state->t_decode_us, n_decode, 1e-3f * ctx->state->t_decode_us / n_decode);
        log("%s:   prompt time = %8.2f ms / %5d runs (%8.2f ms per run)\n", __func__, 1e-3f * ctx->state->t_prompt_us, n_prompt, 1e-3f * ctx->state->t_prompt_us / n_prompt);
    }
    log("%s:    total time = %8.2f ms\n", __func__, (t_end_us - ctx->t_start_us)/1000.0f);
}

void paraformer_reset_timings(struct paraformer_context * ctx) {
    if (ctx->state != nullptr) {
        ctx->state->t_sample_us = 0;
        ctx->state->t_encode_us = 0;
        ctx->state->t_decode_us = 0;
        ctx->state->t_prompt_us = 0;
        ctx->state->n_sample = 0;
        ctx->state->n_encode = 0;
        ctx->state->n_decode = 0;
        ctx->state->n_prompt = 0;
    }
}

static int paraformer_has_coreml(void) {
#ifdef PARAFORMER_USE_COREML
    return 1;
#else
    return 0;
#endif
}

static int paraformer_has_openvino(void) {
#ifdef PARAFORMER_USE_OPENVINO
    return 1;
#else
    return 0;
#endif
}

const char * paraformer_print_system_info(void) {
    static std::string s;

    s  = "";
    s += "AVX = "       + std::to_string(ggml_cpu_has_avx())       + " | ";
    s += "AVX2 = "      + std::to_string(ggml_cpu_has_avx2())      + " | ";
    s += "AVX512 = "    + std::to_string(ggml_cpu_has_avx512())    + " | ";
    s += "FMA = "       + std::to_string(ggml_cpu_has_fma())       + " | ";
    s += "NEON = "      + std::to_string(ggml_cpu_has_neon())      + " | ";
    s += "ARM_FMA = "   + std::to_string(ggml_cpu_has_arm_fma())   + " | ";
    s += "METAL = "     + std::to_string(ggml_cpu_has_metal())     + " | ";
    s += "F16C = "      + std::to_string(ggml_cpu_has_f16c())      + " | ";
    s += "FP16_VA = "   + std::to_string(ggml_cpu_has_fp16_va())   + " | ";
    s += "WASM_SIMD = " + std::to_string(ggml_cpu_has_wasm_simd()) + " | ";
    s += "BLAS = "      + std::to_string(ggml_cpu_has_blas())      + " | ";
    s += "SSE3 = "      + std::to_string(ggml_cpu_has_sse3())      + " | ";
    s += "SSSE3 = "     + std::to_string(ggml_cpu_has_ssse3())     + " | ";
    s += "VSX = "       + std::to_string(ggml_cpu_has_vsx())       + " | ";
    s += "COREML = "    + std::to_string(paraformer_has_coreml())     + " | ";
    s += "OPENVINO = "  + std::to_string(paraformer_has_openvino())   + " | ";

    return s.c_str();
}

////////////////////////////////////////////////////////////////////////////

struct paraformer_full_params * paraformer_full_default_params_by_ref(enum paraformer_sampling_strategy strategy) {
    struct paraformer_full_params params = paraformer_full_default_params(strategy);

    struct paraformer_full_params* result = new paraformer_full_params();
    *result = params;
    return result;
}

struct paraformer_full_params paraformer_full_default_params(enum paraformer_sampling_strategy strategy) {
    struct paraformer_full_params result = {
            /*.strategy          =*/ strategy,

            /*.n_threads         =*/ std::min(4, (int32_t) std::thread::hardware_concurrency()),
            /*.n_max_text_ctx    =*/ 16384,
            /*.offset_ms         =*/ 0,
            /*.duration_ms       =*/ 0,

            /*.translate         =*/ false,
            /*.no_context        =*/ true,
            /*.single_segment    =*/ false,
            /*.print_special     =*/ false,
            /*.print_progress    =*/ true,
            /*.print_realtime    =*/ false,
            /*.print_timestamps  =*/ true,

            /*.token_timestamps  =*/ false,
            /*.thold_pt          =*/ 0.01f,
            /*.thold_ptsum       =*/ 0.01f,
            /*.max_len           =*/ 0,
            /*.split_on_word     =*/ false,
            /*.max_tokens        =*/ 0,

            /*.speed_up          =*/ false,
            /*.debug_mode        =*/ false,
            /*.audio_ctx         =*/ 0,

            /*.tdrz_enable       =*/ false,

            /*.initial_prompt    =*/ nullptr,
            /*.prompt_tokens     =*/ nullptr,
            /*.prompt_n_tokens   =*/ 0,

            /*.language          =*/ "en",
            /*.detect_language   =*/ false,

            /*.suppress_blank    =*/ true,
            /*.suppress_non_speech_tokens =*/ false,

            /*.temperature       =*/  0.0f,
            /*.max_initial_ts    =*/  1.0f,
            /*.length_penalty    =*/ -1.0f,

            /*.temperature_inc   =*/  0.4f,
            /*.entropy_thold     =*/  2.4f,
            /*.logprob_thold     =*/ -1.0f,
            /*.no_speech_thold   =*/  0.6f,

            /*.greedy            =*/ {
                                             /*.best_of   =*/ -1,
                                     },

            /*.beam_search      =*/ {
                                             /*.beam_size =*/ -1,

                                             /*.patience  =*/ -1.0f,
                                     },

            /*.new_segment_callback           =*/ nullptr,
            /*.new_segment_callback_user_data =*/ nullptr,

            /*.progress_callback           =*/ nullptr,
            /*.progress_callback_user_data =*/ nullptr,

            /*.encoder_begin_callback           =*/ nullptr,
            /*.encoder_begin_callback_user_data =*/ nullptr,

            /*.logits_filter_callback           =*/ nullptr,
            /*.logits_filter_callback_user_data =*/ nullptr,
    };

    switch (strategy) {
        case PARAFORMER_SAMPLING_GREEDY:
        {
            result.greedy = {
                    /*.best_of   =*/ 2, // TODO: increase to 5 when we speed-up batch decoding
            };
        } break;
        case PARAFORMER_SAMPLING_BEAM_SEARCH:
        {
            result.beam_search = {
                    /*.beam_size =*/ 2, // TODO: increase to 5 when we speed-up batch decoding

                    /*.patience  =*/ -1.0f,
            };
        } break;
    }

    return result;
}

// forward declarations
static std::vector<float> get_signal_energy(const float * signal, int n_samples, int n_samples_per_half_window);
static void paraformer_exp_compute_token_level_timestamps(
        struct paraformer_context & ctx,
        struct paraformer_state & state,
        int   i_segment,
        float   thold_pt,
        float   thold_ptsum);

static inline bool should_split_on_word(const char * txt, bool split_on_word) {
    if (!split_on_word) return true;

    return txt[0] == ' ';
}

// wrap the last segment to max_len characters
// returns the number of new segments
static int paraformer_wrap_segment(struct paraformer_context & ctx, struct paraformer_state & state, int max_len, bool split_on_word) {
    auto segment = state.result_all.back();

    int res = 1;
    int acc = 0;

    std::string text;

    for (int i = 0; i < (int) segment.tokens.size(); i++) {
        const auto & token = segment.tokens[i];
        if (token.id >= paraformer_token_eot(&ctx)) {
            continue;
        }

        const auto txt = paraformer_token_to_str(&ctx, token.id);
        const int cur = strlen(txt);

        if (acc + cur > max_len && i > 0 && should_split_on_word(txt, split_on_word)) {
            state.result_all.back().text = std::move(text);
            state.result_all.back().t1 = token.t0;
            state.result_all.back().tokens.resize(i);
            state.result_all.back().speaker_turn_next = false;

            state.result_all.push_back({});
            state.result_all.back().t0 = token.t0;
            state.result_all.back().t1 = segment.t1;

            // add tokens [i, end] to the new segment
            state.result_all.back().tokens.insert(
                    state.result_all.back().tokens.end(),
                    segment.tokens.begin() + i,
                    segment.tokens.end());

            state.result_all.back().speaker_turn_next = segment.speaker_turn_next;

            acc = 0;
            text = "";

            segment = state.result_all.back();
            i = -1;

            res++;
        } else {
            acc += cur;
            text += txt;
        }
    }

    state.result_all.back().text = std::move(text);

    return res;
}


// process the logits for the selected decoder
// - applies logit filters
// - computes logprobs and probs
static void paraformer_process_logits(
        struct paraformer_context & ctx,
        struct paraformer_state  & state,
        const struct paraformer_full_params   params,
        struct paraformer_decoder & decoder,
        float   temperature) {
    const auto & vocab      = ctx.vocab;
    const auto & tokens_cur = decoder.sequence.tokens;

    const bool is_initial = tokens_cur.size() == 0;
    const int  n_logits   = vocab.id_to_token.size();

    PARAFORMER_ASSERT(n_logits == ctx.vocab.n_vocab);

    // extract the logits for the last token
    // we will be mutating, and therefore we don't want to use the ctx.logits buffer directly
    auto & probs    = decoder.probs;
    auto & logits   = decoder.logits;
    auto & logprobs = decoder.logprobs;
    {
        logits.resize(n_logits);
        memcpy(logits.data(), state.logits.data() + (state.logits.size() - n_logits), n_logits*sizeof(float));

        if (temperature > 0.0f) {
            for (int i = 0; i < n_logits; i++) {
                logits[i] /= temperature;
            }
        }

        // will be populated a bit later
        probs.resize(n_logits);
        logprobs.resize(n_logits);
    }

    // apply logit filters here
    // ref: https://github.com/openai/paraformer/blob/0b1ba3d46ebf7fe6f953acfd8cad62a4f851b49f/paraformer/decoding.py#L480-L493
    {
        // suppress blank
        // https://github.com/openai/paraformer/blob/0b1ba3d46ebf7fe6f953acfd8cad62a4f851b49f/paraformer/decoding.py#L388-L390
        if (params.suppress_blank) {
            if (is_initial) {
                logits[vocab.token_eot]           = -INFINITY;
                logits[vocab.token_to_id.at(" ")] = -INFINITY;
            }
        }

        // suppress <|notimestamps|> token
        // ref: https://github.com/openai/paraformer/blob/0b1ba3d46ebf7fe6f953acfd8cad62a4f851b49f/paraformer/decoding.py#L410-L412
        logits[vocab.token_not] = -INFINITY;

        // suppress sot and nosp tokens
        logits[vocab.token_sot]  = -INFINITY;
        logits[vocab.token_nosp] = -INFINITY; // TODO: ignore this token for now

        // [TDRZ] when tinydiarize is disabled, suppress solm token
        if (params.tdrz_enable == false) {
            logits[vocab.token_solm] = -INFINITY;
        }

        // suppress task tokens
        logits[vocab.token_translate]  = -INFINITY;
        logits[vocab.token_transcribe] = -INFINITY;

        if (params.logits_filter_callback) {
            params.logits_filter_callback(&ctx, &state, tokens_cur.data(), tokens_cur.size(), logits.data(), params.logits_filter_callback_user_data);
        }

        // suppress non-speech tokens
        // ref: https://github.com/openai/paraformer/blob/7858aa9c08d98f75575035ecd6481f462d66ca27/paraformer/tokenizer.py#L224-L253
        if (params.suppress_non_speech_tokens) {
            for (const std::string & token : non_speech_tokens) {
                const std::string suppress_tokens[] = {token, " " + token};
                for (const std::string & suppress_token : suppress_tokens) {
                    if (vocab.token_to_id.find(suppress_token) != vocab.token_to_id.end()) {
                        logits[vocab.token_to_id.at(suppress_token)] = -INFINITY;
                    }
                }
            }

            // allow hyphens "-" and single quotes "'" between words, but not at the beginning of a word
            if (vocab.token_to_id.find(" -") != vocab.token_to_id.end()) {
                logits[vocab.token_to_id.at(" -")] = -INFINITY;
            }
            if (vocab.token_to_id.find(" '") != vocab.token_to_id.end()) {
                logits[vocab.token_to_id.at(" '")] = -INFINITY;
            }
        }

        // timestamps have to appear in pairs, except directly before EOT; mask logits accordingly
        // https://github.com/openai/paraformer/blob/0b1ba3d46ebf7fe6f953acfd8cad62a4f851b49f/paraformer/decoding.py#L414-L424
        {
            const bool last_was_timestamp        = tokens_cur.size() > 0 && tokens_cur.back().id >= vocab.token_beg;
            const bool penultimate_was_timestamp = tokens_cur.size() < 2 || tokens_cur[tokens_cur.size() - 2].id >= vocab.token_beg;

            //log("last_was_timestamp=%d penultimate_was_timestamp=%d\n", last_was_timestamp, penultimate_was_timestamp);

            if (last_was_timestamp) {
                if (penultimate_was_timestamp) {
                    for (int i = vocab.token_beg; i < n_logits; ++i) {
                        logits[i] = -INFINITY;
                    }
                } else {
                    for (int i = 0; i < vocab.token_eot; ++i) {
                        logits[i] = -INFINITY;
                    }
                }
            }
        }

        // the initial timestamp cannot be larger than max_initial_ts
        // ref: https://github.com/openai/paraformer/blob/0b1ba3d46ebf7fe6f953acfd8cad62a4f851b49f/paraformer/decoding.py#L426-L429
        if (is_initial && params.max_initial_ts > 0.0f) {
            const float precision = float(PARAFORMER_CHUNK_SIZE)/ctx.model.hparams.n_audio_ctx;
            const int   tid0      = std::round(params.max_initial_ts/precision);

            for (int i = vocab.token_beg + tid0 + 1; i < n_logits; ++i) {
                logits[i] = -INFINITY;
            }
        }

        // condition timestamp tokens to be increasing
        // ref: https://github.com/openai/paraformer/pull/831#issuecomment-1385910556
        if (decoder.has_ts) {
            const int tid0 = decoder.seek_delta/2;

            for (int i = vocab.token_beg; i < vocab.token_beg + tid0; ++i) {
                logits[i] = -INFINITY;
            }
        }

        // populate the logprobs array (log_softmax)
        {
            const float logit_max = *std::max_element(logits.begin(), logits.end());
            float logsumexp = 0.0f;
            for (int i = 0; i < n_logits; ++i) {
                if (logits[i] > -INFINITY) {
                    logsumexp += expf(logits[i] - logit_max);
                }
            }
            logsumexp = logf(logsumexp) + logit_max;

            for (int i = 0; i < n_logits; ++i) {
                if (logits[i] > -INFINITY) {
                    logprobs[i] = logits[i] - logsumexp;
                } else {
                    logprobs[i] = -INFINITY;
                }
            }
        }

        // if sum of probability over timestamps is above any other token, sample timestamp
        // ref: https://github.com/openai/paraformer/blob/0b1ba3d46ebf7fe6f953acfd8cad62a4f851b49f/paraformer/decoding.py#L431-L437
        {
            // logsumexp over timestamps
            float timestamp_logprob = -INFINITY;
            {
                float logsumexp = 0.0f;
                const float logprob_max = *std::max_element(logprobs.begin() + vocab.token_beg, logprobs.end());
                for (int i = vocab.token_beg; i < n_logits; ++i) {
                    if (logprobs[i] > -INFINITY) {
                        logsumexp += expf(logprobs[i] - logprob_max);
                    }
                }
                if (logsumexp > 0.0f) {
                    timestamp_logprob = logf(logsumexp) + logprob_max;
                }
            }

            const float max_text_token_logprob = *std::max_element(logprobs.begin(), logprobs.begin() + vocab.token_beg);

            //log("timestamp_logprob=%f max_text_token_logprob=%f\n", timestamp_logprob, max_text_token_logprob);

            if (timestamp_logprob > max_text_token_logprob) {
                for (int i = 0; i < vocab.token_beg; ++i) {
                    logits[i]   = -INFINITY;
                    logprobs[i] = -INFINITY;
                }
            }
        }
    }

    // compute probs
    {
        for (int i = 0; i < n_logits; ++i) {
            if (logits[i] == -INFINITY) {
                probs[i] = 0.0f;
            } else {
                probs[i] = expf(logprobs[i]);
            }
        }
    }

#if 0
    // print first 100 logits - token string : logit
    for (int i = 0; i < 100; i++) {
        const auto token   = vocab.id_to_token.at(i);
        const auto prob    = probs[i];
        const auto logit   = logits[i];
        const auto logprob = logprobs[i];
        printf("%s : prob=%9.5f logit=%9.5f logprob=%9.5f\n", token.c_str(), prob, logit, logprob);
    }

    // "And", "and", " And", " and"
    printf("logits[\"and\"]  = %f\n", logits[vocab.token_to_id.at("and")]);
    printf("logits[\"And\"]  = %f\n", logits[vocab.token_to_id.at("And")]);
    printf("logits[\" and\"] = %f\n", logits[vocab.token_to_id.at(" and")]);
    printf("logits[\" And\"] = %f\n", logits[vocab.token_to_id.at(" And")]);
    printf("logits[\" so\"]  = %f\n", logits[vocab.token_to_id.at(" so")]);

    printf("logprobs[\"and\"]  = %f\n", logprobs[vocab.token_to_id.at("and")]);
    printf("logprobs[\"And\"]  = %f\n", logprobs[vocab.token_to_id.at("And")]);
    printf("logprobs[\" and\"] = %f\n", logprobs[vocab.token_to_id.at(" and")]);
    printf("logprobs[\" And\"] = %f\n", logprobs[vocab.token_to_id.at(" And")]);
    printf("logprobs[\" so\"]  = %f\n", logprobs[vocab.token_to_id.at(" so")]);

    printf("probs[\"and\"]  = %f\n", probs[vocab.token_to_id.at("and")]);
    printf("probs[\"And\"]  = %f\n", probs[vocab.token_to_id.at("And")]);
    printf("probs[\" and\"] = %f\n", probs[vocab.token_to_id.at(" and")]);
    printf("probs[\" And\"] = %f\n", probs[vocab.token_to_id.at(" And")]);
    printf("probs[\" so\"]  = %f\n", probs[vocab.token_to_id.at(" so")]);
#endif
}

static paraformer_token_data paraformer_sample_token(
        paraformer_context & ctx,
        paraformer_state & state,
        const paraformer_decoder & decoder,
        bool   best) {
    paraformer_token_data result = {
            0, 0, 0.0f, 0.0f, 0.0f, 0.0f, -1, -1, 0.0f,
    };

    const auto & vocab = ctx.vocab;

    const auto & probs    = decoder.probs;
    const auto & logprobs = decoder.logprobs;

    const int n_logits = vocab.n_vocab;

    {
        double sum_ts = 0.0;
        double max_ts = 0.0;

        for (int i = vocab.token_beg; i < n_logits; i++) {
            if (probs[i] == -INFINITY) {
                continue;
            }

            sum_ts += probs[i];
            if (max_ts < probs[i]) {
                max_ts = probs[i];
                result.tid = i;
            }
        }

        result.pt    = max_ts/(sum_ts + 1e-10);
        result.ptsum = sum_ts;
    }

    if (best) {
        for (int i = 0; i < n_logits; ++i) {
            if (result.p < probs[i]) {
                result.id   = i;
                result.p    = probs[i];
                result.plog = logprobs[i];
            }
        }
    } else {
        std::discrete_distribution<> dist(probs.begin(), probs.end());

        result.id   = dist(state.rng);
        result.p    = probs[result.id];
        result.plog = logprobs[result.id];
    }

    if (result.id >= vocab.token_beg) {
        result.tid = result.id;
        result.pt  = result.p;
    }

    state.n_sample++;

    return result;
}

static std::vector<paraformer_token_data> paraformer_sample_token_topk(
        paraformer_context & ctx,
        paraformer_state & state,
        const paraformer_decoder & decoder,
        int   k) {
    const auto & vocab = ctx.vocab;

    const auto & probs    = decoder.probs;
    const auto & logits   = decoder.logits;
    const auto & logprobs = decoder.logprobs;

    const int n_logits = vocab.n_vocab;

    auto & logits_id = state.logits_id;

    logits_id.resize(n_logits);
    for (int i = 0; i < n_logits; ++i) {
        logits_id[i].first = logits[i];
        logits_id[i].second = i;
    }

    {
        using pair_type = std::remove_reference<decltype(logits_id)>::type::value_type;
        std::partial_sort(
                logits_id.begin(),
                logits_id.begin() + k, logits_id.end(),
                [](const pair_type & a, const pair_type & b) {
                    return a.first > b.first;
                });
    }

    std::vector<paraformer_token_data> result;
    result.reserve(k);

    paraformer_token tid = vocab.token_beg;

    float pt    = 0.0;
    float ptsum = 0.0;

    {
        double sum_ts = 0.0;
        double max_ts = 0.0;

        for (int i = vocab.token_beg; i < n_logits; i++) {
            if (probs[i] == -INFINITY) {
                continue;
            }

            sum_ts += probs[i];
            if (max_ts < probs[i]) {
                max_ts = probs[i];
                tid = i;
            }
        }

        pt    = max_ts/(sum_ts + 1e-10);
        ptsum = sum_ts;
    }

    for (int i = 0; i < k; ++i) {
        const auto id = logits_id[i].second;

        result.push_back({ id, tid, probs[id], logprobs[id], pt, ptsum, -1, -1, 0.0f, });

        if (result[i].id >= vocab.token_beg) {
            result[i].tid = result[i].id;
            result[i].pt  = result[i].p;
        }
    }

    state.n_sample++;

    return result;
}

// ref: https://github.com/openai/paraformer/blob/0b1ba3d46ebf7fe6f953acfd8cad62a4f851b49f/paraformer/decoding.py#L178-L192
static void paraformer_sequence_score(
        const struct paraformer_full_params & params,
        paraformer_sequence & sequence) {
    if (sequence.result_len == 0) {
        return;
    }

    double result = 0.0f;

    for (int i = 0; i < sequence.result_len; ++i) {
        result += sequence.tokens[i].plog;
    }

    sequence.sum_logprobs = result;
    sequence.avg_logprobs = result/sequence.result_len;

    double penalty = sequence.result_len;

    if (params.length_penalty > 0.0f) {
        penalty = pow((5.0 + penalty)/6.0, params.length_penalty);
    }

    sequence.score = result/penalty;

    // compute the entropy of the sequence of the last 32 tokens
    {
        const int n = 32;

        int cnt = 0;
        double entropy = 0.0f;

        std::map<paraformer_token, int> token_counts;
        for (int i = std::max(0, sequence.result_len - n); i < sequence.result_len; ++i) {
            token_counts[sequence.tokens[i].id]++;
            cnt++;
        }

        for (const auto & kv : token_counts) {
            const auto p = kv.second/(double)cnt;
            entropy -= p*log(p);

            //PARAFORMER_PRINT_DEBUG("entropy: %d %f %f, count %d\n", kv.first, p, log(p), kv.second);
        }

        sequence.entropy = entropy;
    }
}

static bool paraformer_kv_swap_fast(
        std::vector<int> & view,
        paraformer_decoder   src[],
        std::vector<kv_buf> & kv_swap_bufs,
        const int & n_decoders) {
    PARAFORMER_PRINT_DEBUG("%s: n_decoders %d\n", __func__, n_decoders);

    // (decoder->buffer->decoder or decoder->buffer + decoder->decoder)
    std::set<int> two_copy; // decoder indices require two copies to safely modify KV caches

    // (buffer->decoder or decoder->decoder)
    std::set<int> one_copy; // decoder indices require one copy to safely modify KV caches

    // (decoder<->decoder)
    std::set<int> p_swap_set; // decoder indices able to swap KV-cache pointers
    std::vector<paraformer_pair<int, int>> p_swap_vec;
    p_swap_vec.reserve(n_decoders);

    // see https://github.com/ggerganov/paraformer.cpp/wiki
    for (int i = 0; i < n_decoders; i++) {
        // zero-copy (no modification)
        if (i == view[i] || view[i] < 0) {
            continue;
        }

        bool is_one_copy = true;
        // since we modify data sequentially, we only consider decoder indices after current index
        for (int j = i + 1; j < n_decoders; j++) {
            if (i == view[j]) {
                // detect symmetric diagram
                if (j == view[i]) {
                    p_swap_set.insert(i);
                    p_swap_set.insert(j);
                    p_swap_vec.emplace_back(i, j);
                } else {
                    two_copy.insert(i);
                    is_one_copy = false;
                }
                break;
            }
        }
        if (is_one_copy) {
            one_copy.insert(i);
        }
    }

    kv_swap_bufs.resize(n_decoders);

    for (int i = 0; i < n_decoders; i++) {
        kv_swap_bufs[i].k.resize(ggml_nbytes(src[i].kv_self.k));
        kv_swap_bufs[i].v.resize(ggml_nbytes(src[i].kv_self.v));
    }

    for (auto & i : two_copy) {
        // make a copy of KV caches
        PARAFORMER_PRINT_DEBUG("%s: store KV cache into swap: idx %d\n", __func__, i);
        memcpy(kv_swap_bufs[i].k.data(), src[i].kv_self.k->data, kv_swap_bufs[i].k.size());
        memcpy(kv_swap_bufs[i].v.data(), src[i].kv_self.v->data, kv_swap_bufs[i].v.size());
    }

    // since two-copy decoder KV caches are protected by kv_swap_bufs, modify them first
    for (auto & i : two_copy) {
        // skip the decoder indices that require pointer swapping
        if (p_swap_set.find(i) != p_swap_set.end()) {
            continue;
        }

        if (two_copy.find(view[i]) != two_copy.end()) {
            // modify KV caches of decoder using data from kv_swap_bufs
            PARAFORMER_PRINT_DEBUG("%s: two-copy decoder using   swap buffers: swap[%d] -> %d\n", __func__, view[i], i);
            memcpy(src[i].kv_self.k->data, kv_swap_bufs[view[i]].k.data(), kv_swap_bufs[view[i]].k.size());
            memcpy(src[i].kv_self.v->data, kv_swap_bufs[view[i]].v.data(), kv_swap_bufs[view[i]].v.size());
        } else {
            // modify KV caches of decoder using data from correspond decoder KV caches directly
            PARAFORMER_PRINT_DEBUG("%s: two-copy decoder without swap buffers:      %d  -> %d\n", __func__, view[i], i);
            memcpy(src[i].kv_self.k->data, src[view[i]].kv_self.k->data, ggml_nbytes(src[view[i]].kv_self.k));
            memcpy(src[i].kv_self.v->data, src[view[i]].kv_self.v->data, ggml_nbytes(src[view[i]].kv_self.v));
        }
    }

    // then modify one-copy decoder KV caches
    for (auto & i : one_copy) {
        // skip the decoder indices that require pointer swapping
        if (p_swap_set.find(i) != p_swap_set.end()) {
            continue;
        }

        if (two_copy.find(view[i]) != two_copy.end()) {
            // modify KV caches of decoder using data from kv_swap_bufs
            PARAFORMER_PRINT_DEBUG("%s: one-copy decoder using   swap buffers: swap[%d] -> %d\n", __func__, view[i], i);
            memcpy(src[i].kv_self.k->data, kv_swap_bufs[view[i]].k.data(), kv_swap_bufs[view[i]].k.size());
            memcpy(src[i].kv_self.v->data, kv_swap_bufs[view[i]].v.data(), kv_swap_bufs[view[i]].v.size());
        } else {
            // modify KV caches of decoder using data from correspond decoder KV caches directly
            PARAFORMER_PRINT_DEBUG("%s: one-copy decoder without swap buffers:      %d  -> %d\n", __func__, view[i], i);
            memcpy(src[i].kv_self.k->data, src[view[i]].kv_self.k->data, ggml_nbytes(src[view[i]].kv_self.k));
            memcpy(src[i].kv_self.v->data, src[view[i]].kv_self.v->data, ggml_nbytes(src[view[i]].kv_self.v));
        }
    }

    // swap the pointers
    for (auto & i : p_swap_vec) {
        PARAFORMER_PRINT_DEBUG("%s: swap pointers: %d <-> %d\n", __func__, i.first, i.second);
        std::swap(src[i.first].kv_self, src[i.second].kv_self);
    }

    return true;
}

int paraformer_full_with_state(
        struct paraformer_context * ctx,
        struct paraformer_state * state,
        struct paraformer_full_params   params,
        const float * samples,
        int   n_samples) {
    // clear old results
    auto & result_all = state->result_all;

    result_all.clear();

    if (n_samples > 0) {
        // compute log mel spectrogram
        if (params.speed_up) {
            // TODO: Replace PV with more advanced algorithm
            log("%s: failed to compute log mel spectrogram\n", __func__);
            return -1;
        } else {
            if (paraformer_pcm_to_mel_with_state(ctx, state, samples, n_samples, params.n_threads) != 0) {
                log("%s: failed to compute log mel spectrogram\n", __func__);
                return -2;
            }
        }
    }

    // auto-detect language if not specified
    if (params.language == nullptr || strlen(params.language) == 0 || strcmp(params.language, "auto") == 0 || params.detect_language) {
        std::vector<float> probs(paraformer_lang_max_id() + 1, 0.0f);

        const auto lang_id = paraformer_lang_auto_detect_with_state(ctx, state, 0, params.n_threads, probs.data());
        if (lang_id < 0) {
            log("%s: failed to auto-detect language\n", __func__);
            return -3;
        }
        state->lang_id = lang_id;
        params.language = paraformer_lang_str(lang_id);

        log("%s: auto-detected language: %s (p = %f)\n", __func__, params.language, probs[paraformer_lang_id(params.language)]);
        if (params.detect_language) {
            return 0;
        }
    }

    if (params.token_timestamps) {
        state->t_beg    = 0;
        state->t_last   = 0;
        state->tid_last = 0;
        if (n_samples > 0) {
            state->energy = get_signal_energy(samples, n_samples, 32);
        }
    }

    const int seek_start = params.offset_ms/10;
    const int seek_end = params.duration_ms == 0 ? paraformer_n_len_from_state(state) : seek_start + params.duration_ms/10;

    // if length of spectrogram is less than 1.0s (100 frames), then return
    // basically don't process anything that is less than 1.0s
    // see issue #39: https://github.com/ggerganov/paraformer.cpp/issues/39
    if (seek_end < seek_start + (params.speed_up ? 50 : 100)) {
        return 0;
    }

    // a set of temperatures to use
    // [ t0, t0 + delta, t0 + 2*delta, ..., < 1.0f + 1e-6f ]
    std::vector<float> temperatures;
    if (params.temperature_inc > 0.0f) {
        for (float t = params.temperature; t < 1.0f + 1e-6f; t += params.temperature_inc) {
            temperatures.push_back(t);
        }
    } else {
        temperatures.push_back(params.temperature);
    }

    // initialize the decoders
    int n_decoders = 1;

    switch (params.strategy) {
        case PARAFORMER_SAMPLING_GREEDY:
        {
            n_decoders = params.greedy.best_of;
        } break;
        case PARAFORMER_SAMPLING_BEAM_SEARCH:
        {
            n_decoders = std::max(params.greedy.best_of, params.beam_search.beam_size);
        } break;
    };

    n_decoders = std::max(1, n_decoders);

    // TAGS: PARAFORMER_DECODER_INIT
    for (int j = 1; j < n_decoders; j++) {
        auto & decoder = state->decoders[j];

        if (decoder.kv_self.ctx == nullptr) {
            decoder.kv_self = state->decoders[0].kv_self;
            if (!kv_cache_reinit(decoder.kv_self)) {
                log("%s: kv_cache_reinit() failed for self-attention, decoder %d\n", __func__, j);
                return -4;
            }

            PARAFORMER_PRINT_DEBUG("%s: initialized self-attention kv cache, decoder %d\n", __func__, j);

            decoder.sequence.tokens.reserve(state->decoders[0].sequence.tokens.capacity());

            decoder.probs.resize   (ctx->vocab.n_vocab);
            decoder.logits.resize  (ctx->vocab.n_vocab);
            decoder.logprobs.resize(ctx->vocab.n_vocab);

            // TODO: not very clean - look for a better way and potentially merging with the init of decoder 0
#ifdef GGML_USE_METAL
            #define PARAFORMER_METAL_CHECK_BUF(result)              \
            if (!(result)) {                                 \
                log("%s: failed to add metal buffer\n", __func__); \
                return 0;                              \
            }

            const std::string kv_name = "kv_self_" + std::to_string(j);
            auto & kv_self = decoder.kv_self;

            PARAFORMER_METAL_CHECK_BUF(ggml_metal_add_buffer(state->ctx_metal, kv_name.c_str(), kv_self.buf.data(), kv_self.buf.size(), 0));
#undef PARAFORMER_METAL_CHECK_BUF
#endif
        }
    }

    // the accumulated text context so far
    auto & prompt_past = state->prompt_past;
    if (params.no_context) {
        prompt_past.clear();
    }

    // prepare prompt
    {
        std::vector<paraformer_token> prompt_tokens;

        // initial prompt
        if (!params.prompt_tokens && params.initial_prompt) {
            prompt_tokens.resize(2048);
            prompt_tokens.resize(paraformer_tokenize(ctx, params.initial_prompt, prompt_tokens.data(), prompt_tokens.size()));
            params.prompt_tokens   = prompt_tokens.data();
            params.prompt_n_tokens = prompt_tokens.size();
        }

        // prepend the prompt tokens to the prompt_past
        if (params.prompt_tokens && params.prompt_n_tokens > 0) {
            // parse tokens from the pointer
            for (int i = 0; i < params.prompt_n_tokens; i++) {
                prompt_past.push_back(params.prompt_tokens[i]);
            }
            std::rotate(prompt_past.begin(), prompt_past.end() - params.prompt_n_tokens, prompt_past.end());
        }
    }

    // overwrite audio_ctx, max allowed is hparams.n_audio_ctx
    if (params.audio_ctx > paraformer_n_audio_ctx(ctx)) {
        log("%s: audio_ctx is larger than the maximum allowed (%d > %d)\n", __func__, params.audio_ctx, paraformer_n_audio_ctx(ctx));
        return -5;
    }
    state->exp_n_audio_ctx = params.audio_ctx;

    // these tokens determine the task that will be performed
    std::vector<paraformer_token> prompt_init = { paraformer_token_sot(ctx) };
    if (paraformer_is_multilingual(ctx)) {
        const int lang_id = paraformer_lang_id(params.language);
        state->lang_id = lang_id;
        prompt_init.push_back(paraformer_token_lang(ctx, lang_id));
        if (params.translate) {
            prompt_init.push_back(paraformer_token_translate(ctx));
        } else {
            prompt_init.push_back(paraformer_token_transcribe(ctx));
        }
    }

    int seek = seek_start;

    std::vector<paraformer_token> prompt;
    prompt.reserve(paraformer_n_text_ctx(ctx));

    struct beam_candidate {
        int decoder_idx;
        int seek_delta;

        bool has_ts;

        paraformer_sequence sequence;
    };

    std::vector<beam_candidate> beam_candidates;

    // main loop
    while (true) {
        if (params.progress_callback) {
            const int progress_cur = (100*(seek - seek_start))/(seek_end - seek_start);

            params.progress_callback(
                    ctx, ctx->state, progress_cur, params.progress_callback_user_data);
        }

        // of only 1 second left, then stop
        if (seek + 100 >= seek_end) {
            break;
        }

        if (params.encoder_begin_callback) {
            if (params.encoder_begin_callback(ctx, state, params.encoder_begin_callback_user_data) == false) {
                log("%s: encoder_begin_callback returned false - aborting\n", __func__);
                break;
            }
        }

        // encode audio features starting at offset seek
        if (!paraformer_encode_internal(*ctx, *state, seek, params.n_threads)) {
            log("%s: failed to encode\n", __func__);
            return -6;
        }

        // if there is a very short audio segment left to process, we remove any past prompt since it tends
        // to confuse the decoder and often make it repeat or hallucinate stuff
        if (seek > seek_start && seek + 500 >= seek_end) {
            prompt_past.clear();
        }

        int best_decoder_id = 0;

        for (int it = 0; it < (int) temperatures.size(); ++it) {
            const float t_cur = temperatures[it];

            int n_decoders_cur = 1;

            switch (params.strategy) {
                case paraformer_sampling_strategy::PARAFORMER_SAMPLING_GREEDY:
                {
                    if (t_cur > 0.0f) {
                        n_decoders_cur = params.greedy.best_of;
                    }
                } break;
                case paraformer_sampling_strategy::PARAFORMER_SAMPLING_BEAM_SEARCH:
                {
                    if (t_cur > 0.0f) {
                        n_decoders_cur = params.greedy.best_of;
                    } else {
                        n_decoders_cur = params.beam_search.beam_size;
                    }
                } break;
            };

            n_decoders_cur = std::max(1, n_decoders_cur);

            PARAFORMER_PRINT_DEBUG("\n%s: decoding with %d decoders, temperature = %.2f\n", __func__, n_decoders_cur, t_cur);

            // TAGS: PARAFORMER_DECODER_INIT
            for (int j = 0; j < n_decoders_cur; ++j) {
                auto & decoder = state->decoders[j];

                decoder.kv_self.n = 0;

                decoder.sequence.tokens.clear();
                decoder.sequence.result_len       = 0;
                decoder.sequence.sum_logprobs_all = 0.0;
                decoder.sequence.sum_logprobs     = -INFINITY;
                decoder.sequence.avg_logprobs     = -INFINITY;
                decoder.sequence.entropy          = 0.0;
                decoder.sequence.score            = -INFINITY;

                decoder.seek_delta = 100*PARAFORMER_CHUNK_SIZE;

                decoder.failed    = false;
                decoder.completed = false;
                decoder.has_ts    = false;
            }

            // init prompt and kv cache for the current iteration
            // run paraformer_decoder() only for decoder 0 and copy the results for the other decoders
            {
                prompt.clear();

                // if we have already generated some text, use it as a prompt to condition the next generation
                if (!prompt_past.empty() && t_cur < 0.5f && params.n_max_text_ctx > 0) {
                    int n_take = std::min(std::min(params.n_max_text_ctx, paraformer_n_text_ctx(ctx)/2), int(prompt_past.size()));

                    prompt = { paraformer_token_prev(ctx) };
                    prompt.insert(prompt.begin() + 1, prompt_past.end() - n_take, prompt_past.end());
                }

                // init new transcription with sot, language (opt) and task tokens
                prompt.insert(prompt.end(), prompt_init.begin(), prompt_init.end());

                // print the prompt
                PARAFORMER_PRINT_DEBUG("\n\n");
                for (int i = 0; i < (int) prompt.size(); i++) {
                    PARAFORMER_PRINT_DEBUG("%s: prompt[%d] = %s\n", __func__, i, ctx->vocab.id_to_token.at(prompt[i]).c_str());
                }
                PARAFORMER_PRINT_DEBUG("\n\n");

                if (!paraformer_decode_internal(*ctx, *state, state->decoders[0], prompt.data(), prompt.size(), 0, params.n_threads)) {
                    log("%s: failed to decode\n", __func__);
                    return -7;
                }

                {
                    const int64_t t_start_sample_us = ggml_time_us();

                    paraformer_process_logits(*ctx, *state, params, state->decoders[0], t_cur);

                    state->decoders[0].kv_self.n += prompt.size();

                    for (int j = 1; j < n_decoders_cur; ++j) {
                        auto & decoder = state->decoders[j];

                        memcpy(decoder.kv_self.k->data, state->decoders[0].kv_self.k->data, ggml_nbytes(decoder.kv_self.k));
                        memcpy(decoder.kv_self.v->data, state->decoders[0].kv_self.v->data, ggml_nbytes(decoder.kv_self.v));

                        decoder.kv_self.n += prompt.size();

                        memcpy(decoder.probs.data(),    state->decoders[0].probs.data(),    decoder.probs.size()*sizeof(decoder.probs[0]));
                        memcpy(decoder.logits.data(),   state->decoders[0].logits.data(),   decoder.logits.size()*sizeof(decoder.logits[0]));
                        memcpy(decoder.logprobs.data(), state->decoders[0].logprobs.data(), decoder.logprobs.size()*sizeof(decoder.logprobs[0]));
                    }

                    state->t_sample_us += ggml_time_us() - t_start_sample_us;
                }
            }

            for (int i = 0, n_max = paraformer_n_text_ctx(ctx)/2 - 4; i < n_max; ++i) {
                const int64_t t_start_sample_us = ggml_time_us();

                if (params.strategy == paraformer_sampling_strategy::PARAFORMER_SAMPLING_BEAM_SEARCH) {
                    beam_candidates.clear();
                }

                // generate new sequence candidates for each decoder
                for (int j = 0; j < n_decoders_cur; ++j) {
                    auto & decoder = state->decoders[j];

                    if (decoder.completed || decoder.failed) {
                        continue;
                    }

                    switch (params.strategy) {
                        case paraformer_sampling_strategy::PARAFORMER_SAMPLING_GREEDY:
                        {
                            if (t_cur < 1e-6f) {
                                decoder.sequence.tokens.push_back(paraformer_sample_token(*ctx, *state, decoder, true));
                            } else {
                                decoder.sequence.tokens.push_back(paraformer_sample_token(*ctx, *state, decoder, false));
                            }

                            decoder.sequence.sum_logprobs_all += decoder.sequence.tokens.back().plog;
                        } break;
                        case paraformer_sampling_strategy::PARAFORMER_SAMPLING_BEAM_SEARCH:
                        {
                            const auto tokens_new = paraformer_sample_token_topk(*ctx, *state, decoder, params.beam_search.beam_size);

                            for (const auto & token : tokens_new) {
                                beam_candidates.push_back({ j, decoder.seek_delta, decoder.has_ts, decoder.sequence });
                                beam_candidates.back().sequence.tokens.push_back(token);
                                beam_candidates.back().sequence.sum_logprobs_all += token.plog;

                                //PARAFORMER_PRINT_DEBUG("%s: beam candidate: %s (%f, %f)\n", __func__, ctx->vocab.id_to_token.at(token.id).c_str(), token.plog, beam_candidates.back().sequence.sum_logprobs_all);
                            }
                        } break;
                    };
                }

                // for beam-search, choose the top candidates and update the KV caches
                if (params.strategy == paraformer_sampling_strategy::PARAFORMER_SAMPLING_BEAM_SEARCH) {
                    std::sort(
                            beam_candidates.begin(),
                            beam_candidates.end(),
                            [](const beam_candidate & a, const beam_candidate & b) {
                                return a.sequence.sum_logprobs_all > b.sequence.sum_logprobs_all;
                            });

                    uint32_t cur_c = 0;
                    std::vector<int> decoder_idx(n_decoders_cur, -1);

                    for (int j = 0; j < n_decoders_cur; ++j) {
                        auto & decoder = state->decoders[j];

                        if (decoder.completed || decoder.failed) {
                            continue;
                        }

                        auto & cur = beam_candidates[cur_c++];

                        while (beam_candidates.size() > cur_c && beam_candidates[cur_c].sequence.sum_logprobs_all == cur.sequence.sum_logprobs_all && i > 0) {
                            ++cur_c;
                        }

                        decoder.sequence   = cur.sequence;
                        decoder.seek_delta = cur.seek_delta;
                        decoder.has_ts     = cur.has_ts;

                        decoder_idx[j] = cur.decoder_idx;
                        PARAFORMER_PRINT_DEBUG("%s: beam search: decoder %d: from decoder %d: token = %10s, plog = %8.5f, sum_logprobs = %8.5f\n",
                                            __func__, j, cur.decoder_idx, ctx->vocab.id_to_token.at(decoder.sequence.tokens.back().id).c_str(), decoder.sequence.tokens.back().plog, decoder.sequence.sum_logprobs_all);
                    }

                    // update KV caches
                    paraformer_kv_swap_fast(decoder_idx, state->decoders, state->kv_swap_bufs, n_decoders_cur);
                }

                // update the decoder state
                // - check if the sequence is completed
                // - check if the sequence is failed
                // - update sliding window based on timestamp tokens
                for (int j = 0; j < n_decoders_cur; ++j) {
                    auto & decoder = state->decoders[j];

                    if (decoder.completed || decoder.failed) {
                        continue;
                    }

                    auto & has_ts     = decoder.has_ts;
                    auto & failed     = decoder.failed;
                    auto & completed  = decoder.completed;
                    auto & seek_delta = decoder.seek_delta;
                    auto & result_len = decoder.sequence.result_len;

                    {
                        const auto & token = decoder.sequence.tokens.back();

                        // timestamp token - update sliding window
                        if (token.id > paraformer_token_beg(ctx)) {
                            const int seek_delta_new = 2*(token.id - paraformer_token_beg(ctx));

                            // do not allow to go back in time
                            if (has_ts && seek_delta > seek_delta_new && result_len < i) {
                                failed = true; // TODO: maybe this is not a failure ?
                                continue;
                            }

                            seek_delta = seek_delta_new;
                            result_len = i + 1;
                            has_ts = true;
                        }

#ifdef PARAFORMER_DEBUG
                        {
                            const auto tt = token.pt > 0.10 ? ctx->vocab.id_to_token.at(token.tid) : "[?]";
                            PARAFORMER_PRINT_DEBUG("%s: id = %3d, decoder = %d, token = %6d, p = %6.3f, ts = %10s, %6.3f, result_len = %4d '%s'\n",
                                    __func__, i, j, token.id, token.p, tt.c_str(), token.pt, result_len, ctx->vocab.id_to_token.at(token.id).c_str());
                        }
#endif

                        // end of segment
                        if (token.id == paraformer_token_eot(ctx) ||               // end of text token
                            (params.max_tokens > 0 && i >= params.max_tokens) || // max tokens per segment reached
                            (has_ts && seek + seek_delta + 100 >= seek_end)      // end of audio reached
                                ) {
                            if (result_len == 0) {
                                if (seek + seek_delta + 100 >= seek_end) {
                                    result_len = i + 1;
                                } else {
                                    failed = true;
                                    continue;
                                }
                            }

                            if (params.single_segment) {
                                result_len = i + 1;
                                seek_delta = 100*PARAFORMER_CHUNK_SIZE;
                            }

                            completed = true;
                            continue;
                        }

                        // TESTS: if no tensors are loaded, it means we are running tests
                        if (ctx->model.n_loaded == 0) {
                            seek_delta = 100*PARAFORMER_CHUNK_SIZE;
                            completed = true;
                            continue;
                        }
                    }

                    // sometimes, the decoding can get stuck in a repetition loop
                    // this is an attempt to mitigate such cases - we flag the decoding as failed and use a fallback strategy
                    if (i == n_max - 1 && (result_len == 0 || seek_delta < 100*PARAFORMER_CHUNK_SIZE/2)) {
                        failed = true;
                        continue;
                    }
                }

                // check if all decoders have finished (i.e. completed or failed)
                {
                    bool completed_all = true;

                    for (int j = 0; j < n_decoders_cur; ++j) {
                        auto & decoder = state->decoders[j];

                        if (decoder.completed || decoder.failed) {
                            continue;
                        }

                        completed_all = false;
                    }

                    if (completed_all) {
                        break;
                    }
                }

                state->t_sample_us += ggml_time_us() - t_start_sample_us;

                // obtain logits for the next token
                for (int j = 0; j < n_decoders_cur; ++j) {
                    auto & decoder = state->decoders[j];

                    if (decoder.failed || decoder.completed) {
                        continue;
                    }

                    decoder.tokens_tmp.resize(1);
                    decoder.tokens_tmp[0] = decoder.sequence.tokens.back().id;

                    //PARAFORMER_PRINT_DEBUG("%s: decoder %d: token %d, kv_self.n %d, seek_delta %d\n", __func__, j, decoder.tokens_tmp[0], decoder.kv_self.n, decoder.seek_delta);

                    if (!paraformer_decode_internal(*ctx, *state, decoder, decoder.tokens_tmp.data(), decoder.tokens_tmp.size(), decoder.kv_self.n, params.n_threads)) {
                        log("%s: failed to decode\n", __func__);
                        return -8;
                    }

                    {
                        const int64_t t_start_sample_us = ggml_time_us();

                        paraformer_process_logits(*ctx, *state, params, decoder, t_cur);

                        ++decoder.kv_self.n;

                        state->t_sample_us += ggml_time_us() - t_start_sample_us;
                    }
                }
            }

            // rank the resulting sequences and select the best one
            {
                double best_score = -INFINITY;

                for (int j = 0; j < n_decoders_cur; ++j) {
                    auto & decoder = state->decoders[j];

                    if (decoder.failed) {
                        continue;
                    }

                    decoder.sequence.tokens.resize(decoder.sequence.result_len);
                    paraformer_sequence_score(params, decoder.sequence);

                    PARAFORMER_PRINT_DEBUG("%s: decoder %2d: score = %8.5f, result_len = %3d, avg_logprobs = %8.5f, entropy = %8.5f\n",
                                        __func__, j, decoder.sequence.score, decoder.sequence.result_len, decoder.sequence.avg_logprobs, decoder.sequence.entropy);

                    if (decoder.sequence.result_len > 32 && decoder.sequence.entropy < params.entropy_thold) {
                        PARAFORMER_PRINT_DEBUG("%s: decoder %2d: failed due to entropy %8.5f < %8.5f\n",
                                            __func__, j, decoder.sequence.entropy, params.entropy_thold);

                        decoder.failed = true;
                        state->n_fail_h++;

                        continue;
                    }

                    if (best_score < decoder.sequence.score) {
                        best_score = decoder.sequence.score;
                        best_decoder_id = j;
                    }
                }

                PARAFORMER_PRINT_DEBUG("%s: best decoder = %d\n", __func__, best_decoder_id);
            }

            // was the decoding successful for the current temperature?
            // do fallback only if:
            // - we are not at the last temperature
            // - we are not at the end of the audio (3 sec)
            if (it != (int) temperatures.size() - 1 &&
                seek_end - seek > 10*PARAFORMER_CHUNK_SIZE) {
                bool success = true;

                const auto & decoder = state->decoders[best_decoder_id];

                if (decoder.failed || decoder.sequence.avg_logprobs < params.logprob_thold) {
                    success = false;
                    state->n_fail_p++;
                }

                if (success) {
                    //for (auto & token : ctx->decoders[best_decoder_id].sequence.tokens) {
                    //    PARAFORMER_PRINT_DEBUG("%s: token = %d, p = %6.3f, pt = %6.3f, ts = %s, str = %s\n", __func__, token.id, token.p, token.pt, ctx->vocab.id_to_token.at(token.tid).c_str(), ctx->vocab.id_to_token.at(token.id).c_str());
                    //}

                    break;
                }
            }

            PARAFORMER_PRINT_DEBUG("\n%s: failed to decode with temperature = %.2f\n", __func__, t_cur);
        }

        // output results through a user-provided callback
        {
            const auto & best_decoder = state->decoders[best_decoder_id];

            const auto seek_delta = best_decoder.seek_delta;
            const auto result_len = best_decoder.sequence.result_len;

            const auto & tokens_cur = best_decoder.sequence.tokens;

            //PARAFORMER_PRINT_DEBUG("prompt_init.size() = %d, prompt.size() = %d, result_len = %d, seek_delta = %d\n", prompt_init.size(), prompt.size(), result_len, seek_delta);

            // update prompt_past
            prompt_past.clear();
            if (prompt.front() == paraformer_token_prev(ctx)) {
                prompt_past.insert(prompt_past.end(), prompt.begin() + 1, prompt.end() - prompt_init.size());
            }

            for (int i = 0; i < result_len; ++i) {
                prompt_past.push_back(tokens_cur[i].id);
            }

            if (!tokens_cur.empty() && ctx->model.n_loaded > 0) {
                int  i0 = 0;
                auto t0 = seek + 2*(tokens_cur.front().tid - paraformer_token_beg(ctx));

                std::string text;
                bool speaker_turn_next = false;

                for (int i = 0; i < (int) tokens_cur.size(); i++) {
                    //printf("%s: %18s %6.3f %18s %6.3f\n", __func__,
                    //        ctx->vocab.id_to_token[tokens_cur[i].id].c_str(), tokens_cur[i].p,
                    //        ctx->vocab.id_to_token[tokens_cur[i].tid].c_str(), tokens_cur[i].pt);

                    if (params.print_special || tokens_cur[i].id < paraformer_token_eot(ctx)) {
                        text += paraformer_token_to_str(ctx, tokens_cur[i].id);
                    }

                    // [TDRZ] record if speaker turn was predicted after current segment
                    if (params.tdrz_enable && tokens_cur[i].id == paraformer_token_solm(ctx)) {
                        speaker_turn_next = true;
                    }

                    if (tokens_cur[i].id > paraformer_token_beg(ctx) && !params.single_segment) {
                        const auto t1 = seek + 2*(tokens_cur[i].tid - paraformer_token_beg(ctx));

                        if (!text.empty()) {
                            const auto tt0 = params.speed_up ? 2*t0 : t0;
                            const auto tt1 = params.speed_up ? 2*t1 : t1;

                            if (params.print_realtime) {
                                if (params.print_timestamps) {
                                    printf("[%s --> %s]  %s\n", to_timestamp(tt0).c_str(), to_timestamp(tt1).c_str(), text.c_str());
                                } else {
                                    printf("%s", text.c_str());
                                    fflush(stdout);
                                }
                            }

                            //printf("tt0 = %d, tt1 = %d, text = %s, token = %s, token_id = %d, tid = %d\n", tt0, tt1, text.c_str(), ctx->vocab.id_to_token[tokens_cur[i].id].c_str(), tokens_cur[i].id, tokens_cur[i].tid);

                            result_all.push_back({ tt0, tt1, text, {}, speaker_turn_next });
                            for (int j = i0; j <= i; j++) {
                                result_all.back().tokens.push_back(tokens_cur[j]);
                            }

                            int n_new = 1;

                            if (params.token_timestamps) {
                                paraformer_exp_compute_token_level_timestamps(
                                        *ctx, *state, result_all.size() - 1, params.thold_pt, params.thold_ptsum);

                                if (params.max_len > 0) {
                                    n_new = paraformer_wrap_segment(*ctx, *state, params.max_len, params.split_on_word);
                                }
                            }
                            if (params.new_segment_callback) {
                                params.new_segment_callback(ctx, state, n_new, params.new_segment_callback_user_data);
                            }
                        }
                        text = "";
                        while (i < (int) tokens_cur.size() && tokens_cur[i].id > paraformer_token_beg(ctx)) {
                            i++;
                        }
                        i--;
                        t0 = t1;
                        i0 = i + 1;
                        speaker_turn_next = false;
                    }
                }

                if (!text.empty()) {
                    const auto t1 = seek + seek_delta;

                    const auto tt0 = params.speed_up ? 2*t0 : t0;
                    const auto tt1 = params.speed_up ? 2*t1 : t1;

                    if (params.print_realtime) {
                        if (params.print_timestamps) {
                            printf("[%s --> %s]  %s\n", to_timestamp(tt0).c_str(), to_timestamp(tt1).c_str(), text.c_str());
                        } else {
                            printf("%s", text.c_str());
                            fflush(stdout);
                        }
                    }

                    result_all.push_back({ tt0, tt1, text, {} , speaker_turn_next });
                    for (int j = i0; j < (int) tokens_cur.size(); j++) {
                        result_all.back().tokens.push_back(tokens_cur[j]);
                    }

                    int n_new = 1;

                    if (params.token_timestamps) {
                        paraformer_exp_compute_token_level_timestamps(
                                *ctx, *state, result_all.size() - 1, params.thold_pt, params.thold_ptsum);

                        if (params.max_len > 0) {
                            n_new = paraformer_wrap_segment(*ctx, *state, params.max_len, params.split_on_word);
                        }
                    }
                    if (params.new_segment_callback) {
                        params.new_segment_callback(ctx, state, n_new, params.new_segment_callback_user_data);
                    }
                }
            }

            // update audio window
            seek += seek_delta;

            PARAFORMER_PRINT_DEBUG("seek = %d, seek_delta = %d\n", seek, seek_delta);
        }
    }

    return 0;
}

int paraformer_full(
        struct paraformer_context * ctx,
        struct paraformer_full_params   params,
        const float * samples,
        int   n_samples) {
    return paraformer_full_with_state(ctx, ctx->state, params, samples, n_samples);
}

int paraformer_full_parallel(
        struct paraformer_context * ctx,
        struct paraformer_full_params params,
        const float * samples,
        int n_samples,
        int n_processors) {
    if (n_processors == 1) {
        return paraformer_full(ctx, params, samples, n_samples);
    }
    int ret = 0;

    // prepare separate states for each thread
    std::vector<paraformer_state*> states;

    const int offset_samples = (PARAFORMER_SAMPLE_RATE*params.offset_ms)/1000;
    const int n_samples_per_processor = (n_samples - offset_samples)/n_processors;

    // the calling thread will process the first chunk
    // while the other threads will process the remaining chunks

    std::vector<std::thread> workers(n_processors - 1);
    for (int i = 0; i < n_processors - 1; ++i) {
        // create a new state for each thread
        states.push_back(paraformer_init_state(ctx));

        const int start_samples = offset_samples + (i + 1)*n_samples_per_processor;
        const int n_samples_cur = (i == n_processors - 2) ? n_samples - start_samples : n_samples_per_processor;

        auto params_cur = params;

        params_cur.offset_ms = 0;
        params_cur.print_progress = false;
        params_cur.print_realtime = false;

        params_cur.new_segment_callback = nullptr;
        params_cur.new_segment_callback_user_data = nullptr;

        params_cur.progress_callback = nullptr;
        params_cur.progress_callback_user_data = nullptr;

        workers[i] = std::thread(paraformer_full_with_state, ctx, states[i], std::move(params_cur), samples + start_samples, n_samples_cur);
    }

    {
        auto params_cur = params;

        // We need to disable the print real-time for this one as well, otherwise it will show only for the first chunk.
        params_cur.print_realtime = false;

        // Run the first transformation using default state but only for the first chunk.
        ret = paraformer_full_with_state(ctx, ctx->state, std::move(params_cur), samples, offset_samples + n_samples_per_processor);
    }

    for (int i = 0; i < n_processors - 1; ++i) {
        workers[i].join();
    }

    const int64_t offset_t = (int64_t) params.offset_ms/10.0;

    // combine results into result_state->result_all from all other states
    for (int i = 0; i < n_processors - 1; ++i) {
        auto& results_i = states[i]->result_all;

        for (auto& result : results_i) {
            // correct the segment timestamp taking into account the offset
            result.t0 += 100 * ((i + 1) * n_samples_per_processor) / PARAFORMER_SAMPLE_RATE + offset_t;
            result.t1 += 100 * ((i + 1) * n_samples_per_processor) / PARAFORMER_SAMPLE_RATE + offset_t;

            // make sure that segments are not overlapping
            if (!ctx->state->result_all.empty()) {
                result.t0 = std::max(result.t0, ctx->state->result_all.back().t1);
            }

            ctx->state->result_all.push_back(std::move(result));

            // call the new_segment_callback for each segment
            if (params.new_segment_callback) {
                params.new_segment_callback(ctx, ctx->state, 1, params.new_segment_callback_user_data);
            }
        }

        ctx->state->t_mel_us += states[i]->t_mel_us;

        ctx->state->t_sample_us += states[i]->t_sample_us;
        ctx->state->t_encode_us += states[i]->t_encode_us;
        ctx->state->t_decode_us += states[i]->t_decode_us;
        ctx->state->t_prompt_us += states[i]->t_prompt_us;

        ctx->state->n_sample += states[i]->n_sample;
        ctx->state->n_encode += states[i]->n_encode;
        ctx->state->n_decode += states[i]->n_decode;
        ctx->state->n_prompt += states[i]->n_prompt;

        paraformer_free_state(states[i]);
    }

    // average the timings
    ctx->state->t_mel_us    /= n_processors;
    ctx->state->t_sample_us /= n_processors;
    ctx->state->t_encode_us /= n_processors;
    ctx->state->t_decode_us /= n_processors;

    // print information about the audio boundaries
    log("\n");
    log("%s: the audio has been split into %d chunks at the following times:\n", __func__, n_processors);
    for (int i = 0; i < n_processors - 1; ++i) {
        log("%s: split %d - %s\n", __func__, (i + 1), to_timestamp(100*((i + 1)*n_samples_per_processor)/PARAFORMER_SAMPLE_RATE + offset_t).c_str());
    }
    log("%s: the transcription quality may be degraded near these boundaries\n", __func__);

    return ret;
}

int paraformer_full_n_segments_from_state(struct paraformer_state * state) {
    return state->result_all.size();
}

int paraformer_full_n_segments(struct paraformer_context * ctx) {
    return ctx->state->result_all.size();
}

int paraformer_full_lang_id_from_state(struct paraformer_state * state) {
    return state->lang_id;
}

int paraformer_full_lang_id(struct paraformer_context * ctx) {
    return ctx->state->lang_id;
}

int64_t paraformer_full_get_segment_t0_from_state(struct paraformer_state * state, int i_segment) {
    return state->result_all[i_segment].t0;
}

int64_t paraformer_full_get_segment_t0(struct paraformer_context * ctx, int i_segment) {
    return ctx->state->result_all[i_segment].t0;
}

int64_t paraformer_full_get_segment_t1_from_state(struct paraformer_state * state, int i_segment) {
    return state->result_all[i_segment].t1;
}

int64_t paraformer_full_get_segment_t1(struct paraformer_context * ctx, int i_segment) {
    return ctx->state->result_all[i_segment].t1;
}

bool paraformer_full_get_segment_speaker_turn_next(struct paraformer_context * ctx, int i_segment) {
    return ctx->state->result_all[i_segment].speaker_turn_next;
}

const char * paraformer_full_get_segment_text_from_state(struct paraformer_state * state, int i_segment) {
    return state->result_all[i_segment].text.c_str();
}

const char * paraformer_full_get_segment_text(struct paraformer_context * ctx, int i_segment) {
    return ctx->state->result_all[i_segment].text.c_str();
}

int paraformer_full_n_tokens_from_state(struct paraformer_state * state, int i_segment) {
    return state->result_all[i_segment].tokens.size();
}

int paraformer_full_n_tokens(struct paraformer_context * ctx, int i_segment) {
    return ctx->state->result_all[i_segment].tokens.size();
}

const char * paraformer_full_get_token_text_from_state(struct paraformer_context * ctx, struct paraformer_state * state, int i_segment, int i_token) {
    return ctx->vocab.id_to_token[state->result_all[i_segment].tokens[i_token].id].c_str();
}

const char* paraformer_full_get_token_text(struct paraformer_context * ctx, int i_segment, int i_token) {
    return ctx->vocab.id_to_token[ctx->state->result_all[i_segment].tokens[i_token].id].c_str();
}

paraformer_token paraformer_full_get_token_id_from_state(struct paraformer_state * state, int i_segment, int i_token) {
    return state->result_all[i_segment].tokens[i_token].id;
}

paraformer_token paraformer_full_get_token_id(struct paraformer_context * ctx, int i_segment, int i_token) {
    return ctx->state->result_all[i_segment].tokens[i_token].id;
}

struct paraformer_token_data paraformer_full_get_token_data_from_state(struct paraformer_state * state, int i_segment, int i_token) {
    return state->result_all[i_segment].tokens[i_token];
}

struct paraformer_token_data paraformer_full_get_token_data(struct paraformer_context * ctx, int i_segment, int i_token) {
    return ctx->state->result_all[i_segment].tokens[i_token];
}

float paraformer_full_get_token_p_from_state(struct paraformer_state * state, int i_segment, int i_token) {
    return state->result_all[i_segment].tokens[i_token].p;
}

float paraformer_full_get_token_p(struct paraformer_context * ctx, int i_segment, int i_token) {
    return ctx->state->result_all[i_segment].tokens[i_token].p;
}

