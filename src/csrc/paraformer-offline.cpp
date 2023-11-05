//
// Created by lovemefan on 2023/11/4.
//

#include "paraformer-offline.h"

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
static bool paraformer_model_load(struct paraformer_model_loader *loader,
                                  paraformer_context &wctx) {
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
        read_safe(loader, hparams.n_audio_ctx);
        read_safe(loader, hparams.n_audio_state);
        read_safe(loader, hparams.n_audio_head);
        read_safe(loader, hparams.n_audio_layer);
        read_safe(loader, hparams.n_text_ctx);
        read_safe(loader, hparams.n_text_state);
        read_safe(loader, hparams.n_text_head);
        read_safe(loader, hparams.n_text_layer);
        read_safe(loader, hparams.n_mels);
        read_safe(loader, hparams.ftype);

        assert(hparams.n_text_state == hparams.n_audio_state);

        if (hparams.n_audio_layer == 4) {
            model.type = e_model::MODEL_TINY;
        }

        if (hparams.n_audio_layer == 6) {
            model.type = e_model::MODEL_BASE;
        }

        if (hparams.n_audio_layer == 12) {
            model.type = e_model::MODEL_SMALL;
        }

        if (hparams.n_audio_layer == 24) {
            model.type = e_model::MODEL_MEDIUM;
        }

        if (hparams.n_audio_layer == 32) {
            model.type = e_model::MODEL_LARGE;
        }

        const int32_t qntvr = hparams.ftype / GGML_QNT_VERSION_FACTOR;

        hparams.ftype %= GGML_QNT_VERSION_FACTOR;

        // for the big tensors, we have the option to store the data in 16-bit
        // floats or quantized in order to save memory and also to speed up the
        // computation
        wctx.wtype = ggml_ftype_to_ggml_type((ggml_ftype)(model.hparams.ftype));
        if (wctx.wtype == GGML_TYPE_COUNT) {
            log("%s: invalid model (bad ftype value %d)\n", __func__,
                model.hparams.ftype);
            return false;
        }

        const size_t scale = model.hparams.ftype ? 1 : 2;

        log("%s: n_vocab       = %d\n", __func__, hparams.n_vocab);
        log("%s: n_audio_ctx   = %d\n", __func__, hparams.n_audio_ctx);
        log("%s: n_audio_state = %d\n", __func__, hparams.n_audio_state);
        log("%s: n_audio_head  = %d\n", __func__, hparams.n_audio_head);
        log("%s: n_audio_layer = %d\n", __func__, hparams.n_audio_layer);
        log("%s: n_text_ctx    = %d\n", __func__, hparams.n_text_ctx);
        log("%s: n_text_state  = %d\n", __func__, hparams.n_text_state);
        log("%s: n_text_head   = %d\n", __func__, hparams.n_text_head);
        log("%s: n_text_layer  = %d\n", __func__, hparams.n_text_layer);
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
        wctx.model.buf->resize(scale *
                               MEM_REQ_MODEL.at(wctx.wtype).at(model.type));

        // we skip initialization of the state until it is needed
        // because it might be that state will always be provided externally.
    }

    // load mel filters
    {
        auto &filters = wctx.model.filters;

        read_safe(loader, filters.n_mel);
        read_safe(loader, filters.n_fft);

        filters.data.resize(filters.n_mel * filters.n_fft);
        loader->read(loader->context, filters.data.data(),
                     filters.data.size() * sizeof(float));
        BYTESWAP_FILTERS(filters);
    }

    // load vocab
    {
        int32_t n_vocab = 0;
        read_safe(loader, n_vocab);

        // if (n_vocab != model.hparams.n_vocab) {
        //     log("%s: invalid model file '%s' (bad vocab size %d != %d)\n",
        //             __func__, fname.c_str(), n_vocab, model.hparams.n_vocab);
        //     return false;
        // }

        std::string word;
        std::vector<char> tmp;

        tmp.reserve(128);

        for (int i = 0; i < n_vocab; i++) {
            uint32_t len;
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
        if (vocab.is_multilingual()) {
            vocab.token_eot++;
            vocab.token_sot++;
            vocab.token_translate++;
            vocab.token_transcribe++;
            vocab.token_solm++;
            vocab.token_prev++;
            vocab.token_nosp++;
            vocab.token_not++;
            vocab.token_beg++;
        }

        if (n_vocab < model.hparams.n_vocab) {
            log("%s: adding %d extra tokens\n", __func__,
                model.hparams.n_vocab - n_vocab);
            for (int i = n_vocab; i < model.hparams.n_vocab; i++) {
                if (i > vocab.token_beg) {
                    word = "[_TT_" + std::to_string(i - vocab.token_beg) + "]";
                } else if (i == vocab.token_eot) {
                    word = "[_EOT_]";
                } else if (i == vocab.token_sot) {
                    word = "[_SOT_]";
                } else if (i == vocab.token_solm) {
                    word = "[_SOLM_]";
                } else if (i == vocab.token_prev) {
                    word = "[_PREV_]";
                } else if (i == vocab.token_nosp) {
                    word = "[_NOSP_]";
                } else if (i == vocab.token_not) {
                    word = "[_NOT_]";
                } else if (i == vocab.token_beg) {
                    word = "[_BEG_]";
                } else {
                    word = "[_extra_token_" + std::to_string(i) + "]";
                }
                vocab.token_to_id[word] = i;
                vocab.id_to_token[i] = word;
            }
        }
    }

    size_t ctx_size = 0;

    const ggml_type wtype = wctx.wtype;
    const ggml_type vtype = wctx.wtype == GGML_TYPE_F32
                                ? GGML_TYPE_F32
                                : GGML_TYPE_F16;  // conv type

    {
        const auto &hparams = model.hparams;

        const int n_vocab = hparams.n_vocab;

        const int n_audio_ctx = hparams.n_audio_ctx;
        const int n_audio_state = hparams.n_audio_state;
        const int n_audio_layer = hparams.n_audio_layer;

        const int n_text_ctx = hparams.n_text_ctx;
        const int n_text_state = hparams.n_text_state;
        const int n_text_layer = hparams.n_text_layer;

        const int n_mels = hparams.n_mels;

        // encoder
        {
            ctx_size += n_audio_ctx * n_audio_state *
                        ggml_type_sizef(GGML_TYPE_F32);  // e_pe;

            ctx_size += 3 * n_mels * n_audio_state *
                        ggml_type_sizef(vtype);  // e_conv_1_w
            ctx_size +=
                n_audio_state * ggml_type_sizef(GGML_TYPE_F32);  // e_conv_1_b

            ctx_size += 3 * n_audio_state * n_audio_state *
                        ggml_type_sizef(vtype);  // e_conv_2_w
            ctx_size +=
                n_audio_state * ggml_type_sizef(GGML_TYPE_F32);  // e_conv_2_b

            ctx_size +=
                n_audio_state * ggml_type_sizef(GGML_TYPE_F32);  // e_ln_w;
            ctx_size +=
                n_audio_state * ggml_type_sizef(GGML_TYPE_F32);  // e_ln_b;
        }

        // decoder
        {
            ctx_size += n_text_ctx * n_text_state *
                        ggml_type_sizef(GGML_TYPE_F32);  // d_pe;

            ctx_size +=
                n_vocab * n_text_state * ggml_type_sizef(wtype);  // d_te;

            ctx_size +=
                n_text_state * ggml_type_sizef(GGML_TYPE_F32);  // d_ln_w;
            ctx_size +=
                n_text_state * ggml_type_sizef(GGML_TYPE_F32);  // d_ln_b;
        }

        // encoder layers
        {
            ctx_size +=
                n_audio_layer *
                (n_audio_state * ggml_type_sizef(GGML_TYPE_F32));  // mlp_ln_w
            ctx_size +=
                n_audio_layer *
                (n_audio_state * ggml_type_sizef(GGML_TYPE_F32));  // mlp_ln_b

            ctx_size += n_audio_layer * (4 * n_audio_state * n_audio_state *
                                         ggml_type_sizef(wtype));  // mlp_0_w
            ctx_size +=
                n_audio_layer * (4 * n_audio_state *
                                 ggml_type_sizef(GGML_TYPE_F32));  // mlp_0_b

            ctx_size += n_audio_layer * (4 * n_audio_state * n_audio_state *
                                         ggml_type_sizef(wtype));  // mlp_1_w
            ctx_size +=
                n_audio_layer *
                (n_audio_state * ggml_type_sizef(GGML_TYPE_F32));  // mlp_1_b

            ctx_size += n_audio_layer *
                        (n_audio_state *
                         ggml_type_sizef(GGML_TYPE_F32));  // attn_ln_0_w
            ctx_size += n_audio_layer *
                        (n_audio_state *
                         ggml_type_sizef(GGML_TYPE_F32));  // attn_ln_0_b

            ctx_size += n_audio_layer * (n_audio_state * n_audio_state *
                                         ggml_type_sizef(wtype));  // attn_q_w
            ctx_size +=
                n_audio_layer *
                (n_audio_state * ggml_type_sizef(GGML_TYPE_F32));  // attn_q_b

            ctx_size += n_audio_layer * (n_audio_state * n_audio_state *
                                         ggml_type_sizef(wtype));  // attn_k_w

            ctx_size += n_audio_layer * (n_audio_state * n_audio_state *
                                         ggml_type_sizef(wtype));  // attn_v_w
            ctx_size +=
                n_audio_layer *
                (n_audio_state * ggml_type_sizef(GGML_TYPE_F32));  // attn_v_b

            ctx_size +=
                n_audio_layer * (n_audio_state * n_audio_state *
                                 ggml_type_sizef(wtype));  // attn_ln_1_w
            ctx_size += n_audio_layer *
                        (n_audio_state *
                         ggml_type_sizef(GGML_TYPE_F32));  // attn_ln_1_b
        }

        // decoder layers
        {
            ctx_size +=
                n_text_layer *
                (n_text_state * ggml_type_sizef(GGML_TYPE_F32));  // mlp_ln_w
            ctx_size +=
                n_text_layer *
                (n_text_state * ggml_type_sizef(GGML_TYPE_F32));  // mlp_ln_b

            ctx_size += n_text_layer * (4 * n_text_state * n_text_state *
                                        ggml_type_sizef(wtype));  // mlp_0_w
            ctx_size +=
                n_text_layer *
                (4 * n_text_state * ggml_type_sizef(GGML_TYPE_F32));  // mlp_0_b

            ctx_size += n_text_layer * (4 * n_text_state * n_text_state *
                                        ggml_type_sizef(wtype));  // mlp_1_w
            ctx_size +=
                n_text_layer *
                (n_text_state * ggml_type_sizef(GGML_TYPE_F32));  // mlp_1_b

            ctx_size +=
                n_text_layer *
                (n_text_state * ggml_type_sizef(GGML_TYPE_F32));  // attn_ln_0_w
            ctx_size +=
                n_text_layer *
                (n_text_state * ggml_type_sizef(GGML_TYPE_F32));  // attn_ln_0_b

            ctx_size += n_text_layer * (n_text_state * n_text_state *
                                        ggml_type_sizef(wtype));  // attn_q_w
            ctx_size +=
                n_text_layer *
                (n_text_state * ggml_type_sizef(GGML_TYPE_F32));  // attn_q_b

            ctx_size += n_text_layer * (n_text_state * n_text_state *
                                        ggml_type_sizef(wtype));  // attn_k_w

            ctx_size += n_text_layer * (n_text_state * n_text_state *
                                        ggml_type_sizef(wtype));  // attn_v_w
            ctx_size +=
                n_text_layer *
                (n_text_state * ggml_type_sizef(GGML_TYPE_F32));  // attn_v_b

            ctx_size += n_text_layer * (n_text_state * n_text_state *
                                        ggml_type_sizef(wtype));  // attn_ln_1_w
            ctx_size +=
                n_text_layer *
                (n_text_state * ggml_type_sizef(GGML_TYPE_F32));  // attn_ln_1_b
                                                                  //
            ctx_size += n_text_layer *
                        (n_text_state *
                         ggml_type_sizef(GGML_TYPE_F32));  // cross_attn_ln_0_w
            ctx_size += n_text_layer *
                        (n_text_state *
                         ggml_type_sizef(GGML_TYPE_F32));  // cross_attn_ln_0_b

            ctx_size +=
                n_text_layer * (n_text_state * n_text_state *
                                ggml_type_sizef(wtype));  // cross_attn_q_w
            ctx_size += n_text_layer *
                        (n_text_state *
                         ggml_type_sizef(GGML_TYPE_F32));  // cross_attn_q_b

            ctx_size +=
                n_text_layer * (n_text_state * n_text_state *
                                ggml_type_sizef(wtype));  // cross_attn_k_w

            ctx_size +=
                n_text_layer * (n_text_state * n_text_state *
                                ggml_type_sizef(wtype));  // cross_attn_v_w
            ctx_size += n_text_layer *
                        (n_text_state *
                         ggml_type_sizef(GGML_TYPE_F32));  // cross_attn_v_b

            ctx_size +=
                n_text_layer * (n_text_state * n_text_state *
                                ggml_type_sizef(wtype));  // cross_attn_ln_1_w
            ctx_size += n_text_layer *
                        (n_text_state *
                         ggml_type_sizef(GGML_TYPE_F32));  // cross_attn_ln_1_b
        }

        ctx_size += (15 + 15 * n_audio_layer + 24 * n_text_layer) *
                    512;  // object overhead

        log("%s: model ctx     = %7.2f MB\n", __func__,
            ctx_size / (1024.0 * 1024.0));
    }

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

    // prepare memory for the weights
    {
        auto &ctx = model.ctx;

        const auto &hparams = model.hparams;

        const int n_vocab = hparams.n_vocab;

        const int n_audio_ctx = hparams.n_audio_ctx;
        const int n_audio_state = hparams.n_audio_state;
        const int n_audio_layer = hparams.n_audio_layer;

        const int n_text_ctx = hparams.n_text_ctx;
        const int n_text_state = hparams.n_text_state;
        const int n_text_layer = hparams.n_text_layer;

        const int n_mels = hparams.n_mels;

        model.layers_encoder.resize(n_audio_layer);
        model.layers_decoder.resize(n_text_layer);

        // encoder
        {
            model.e_pe = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_audio_state,
                                            n_audio_ctx);

            model.e_conv_1_w =
                ggml_new_tensor_3d(ctx, vtype, 3, n_mels, n_audio_state);
            model.e_conv_1_b =
                ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n_audio_state);

            model.e_conv_2_w =
                ggml_new_tensor_3d(ctx, vtype, 3, n_audio_state, n_audio_state);
            model.e_conv_2_b =
                ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n_audio_state);

            model.e_ln_w =
                ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state);
            model.e_ln_b =
                ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state);

            // map by name
            model.tensors["encoder.positional_embedding"] = model.e_pe;

            model.tensors["encoder.conv1.weight"] = model.e_conv_1_w;
            model.tensors["encoder.conv1.bias"] = model.e_conv_1_b;

            model.tensors["encoder.conv2.weight"] = model.e_conv_2_w;
            model.tensors["encoder.conv2.bias"] = model.e_conv_2_b;

            model.tensors["encoder.ln_post.weight"] = model.e_ln_w;
            model.tensors["encoder.ln_post.bias"] = model.e_ln_b;

            for (int i = 0; i < n_audio_layer; ++i) {
                auto &layer = model.layers_encoder[i];

                layer.mlp_ln_w =
                    ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state);
                layer.mlp_ln_b =
                    ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state);

                layer.mlp_0_w = ggml_new_tensor_2d(ctx, wtype, n_audio_state,
                                                   4 * n_audio_state);
                layer.mlp_0_b =
                    ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4 * n_audio_state);

                layer.mlp_1_w = ggml_new_tensor_2d(
                    ctx, wtype, 4 * n_audio_state, n_audio_state);
                layer.mlp_1_b =
                    ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state);

                layer.attn_ln_0_w =
                    ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state);
                layer.attn_ln_0_b =
                    ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state);

                layer.attn_q_w = ggml_new_tensor_2d(ctx, wtype, n_audio_state,
                                                    n_audio_state);
                layer.attn_q_b =
                    ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state);

                layer.attn_k_w = ggml_new_tensor_2d(ctx, wtype, n_audio_state,
                                                    n_audio_state);

                layer.attn_v_w = ggml_new_tensor_2d(ctx, wtype, n_audio_state,
                                                    n_audio_state);
                layer.attn_v_b =
                    ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state);

                layer.attn_ln_1_w = ggml_new_tensor_2d(
                    ctx, wtype, n_audio_state, n_audio_state);
                layer.attn_ln_1_b =
                    ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state);

                // map by name
                model.tensors["encoder.blocks." + std::to_string(i) +
                              ".mlp_ln.weight"] = layer.mlp_ln_w;
                model.tensors["encoder.blocks." + std::to_string(i) +
                              ".mlp_ln.bias"] = layer.mlp_ln_b;

                model.tensors["encoder.blocks." + std::to_string(i) +
                              ".mlp.0.weight"] = layer.mlp_0_w;
                model.tensors["encoder.blocks." + std::to_string(i) +
                              ".mlp.0.bias"] = layer.mlp_0_b;

                model.tensors["encoder.blocks." + std::to_string(i) +
                              ".mlp.2.weight"] = layer.mlp_1_w;
                model.tensors["encoder.blocks." + std::to_string(i) +
                              ".mlp.2.bias"] = layer.mlp_1_b;

                model.tensors["encoder.blocks." + std::to_string(i) +
                              ".attn_ln.weight"] = layer.attn_ln_0_w;
                model.tensors["encoder.blocks." + std::to_string(i) +
                              ".attn_ln.bias"] = layer.attn_ln_0_b;

                model.tensors["encoder.blocks." + std::to_string(i) +
                              ".attn.query.weight"] = layer.attn_q_w;
                model.tensors["encoder.blocks." + std::to_string(i) +
                              ".attn.query.bias"] = layer.attn_q_b;

                model.tensors["encoder.blocks." + std::to_string(i) +
                              ".attn.key.weight"] = layer.attn_k_w;

                model.tensors["encoder.blocks." + std::to_string(i) +
                              ".attn.value.weight"] = layer.attn_v_w;
                model.tensors["encoder.blocks." + std::to_string(i) +
                              ".attn.value.bias"] = layer.attn_v_b;

                model.tensors["encoder.blocks." + std::to_string(i) +
                              ".attn.out.weight"] = layer.attn_ln_1_w;
                model.tensors["encoder.blocks." + std::to_string(i) +
                              ".attn.out.bias"] = layer.attn_ln_1_b;
            }
        }

        // decoder
        {
            model.d_pe = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_text_state,
                                            n_text_ctx);

            model.d_te = ggml_new_tensor_2d(ctx, wtype, n_text_state, n_vocab);

            model.d_ln_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state);
            model.d_ln_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state);

            // map by name
            model.tensors["decoder.positional_embedding"] = model.d_pe;

            model.tensors["decoder.token_embedding.weight"] = model.d_te;

            model.tensors["decoder.ln.weight"] = model.d_ln_w;
            model.tensors["decoder.ln.bias"] = model.d_ln_b;

            for (int i = 0; i < n_text_layer; ++i) {
                auto &layer = model.layers_decoder[i];

                layer.mlp_ln_w =
                    ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state);
                layer.mlp_ln_b =
                    ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state);

                layer.mlp_0_w = ggml_new_tensor_2d(ctx, wtype, n_text_state,
                                                   4 * n_text_state);
                layer.mlp_0_b =
                    ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4 * n_text_state);

                layer.mlp_1_w = ggml_new_tensor_2d(ctx, wtype, 4 * n_text_state,
                                                   n_text_state);
                layer.mlp_1_b =
                    ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state);

                layer.attn_ln_0_w =
                    ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state);
                layer.attn_ln_0_b =
                    ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state);

                layer.attn_q_w =
                    ggml_new_tensor_2d(ctx, wtype, n_text_state, n_text_state);
                layer.attn_q_b =
                    ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state);

                layer.attn_k_w =
                    ggml_new_tensor_2d(ctx, wtype, n_text_state, n_text_state);

                layer.attn_v_w =
                    ggml_new_tensor_2d(ctx, wtype, n_text_state, n_text_state);
                layer.attn_v_b =
                    ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state);

                layer.attn_ln_1_w =
                    ggml_new_tensor_2d(ctx, wtype, n_text_state, n_text_state);
                layer.attn_ln_1_b =
                    ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state);

                layer.cross_attn_ln_0_w =
                    ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state);
                layer.cross_attn_ln_0_b =
                    ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state);

                layer.cross_attn_q_w =
                    ggml_new_tensor_2d(ctx, wtype, n_text_state, n_text_state);
                layer.cross_attn_q_b =
                    ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state);

                layer.cross_attn_k_w =
                    ggml_new_tensor_2d(ctx, wtype, n_text_state, n_text_state);

                layer.cross_attn_v_w =
                    ggml_new_tensor_2d(ctx, wtype, n_text_state, n_text_state);
                layer.cross_attn_v_b =
                    ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state);

                layer.cross_attn_ln_1_w =
                    ggml_new_tensor_2d(ctx, wtype, n_text_state, n_text_state);
                layer.cross_attn_ln_1_b =
                    ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state);

                // map by name
                model.tensors["decoder.blocks." + std::to_string(i) +
                              ".mlp_ln.weight"] = layer.mlp_ln_w;
                model.tensors["decoder.blocks." + std::to_string(i) +
                              ".mlp_ln.bias"] = layer.mlp_ln_b;

                model.tensors["decoder.blocks." + std::to_string(i) +
                              ".mlp.0.weight"] = layer.mlp_0_w;
                model.tensors["decoder.blocks." + std::to_string(i) +
                              ".mlp.0.bias"] = layer.mlp_0_b;

                model.tensors["decoder.blocks." + std::to_string(i) +
                              ".mlp.2.weight"] = layer.mlp_1_w;
                model.tensors["decoder.blocks." + std::to_string(i) +
                              ".mlp.2.bias"] = layer.mlp_1_b;

                model.tensors["decoder.blocks." + std::to_string(i) +
                              ".attn_ln.weight"] = layer.attn_ln_0_w;
                model.tensors["decoder.blocks." + std::to_string(i) +
                              ".attn_ln.bias"] = layer.attn_ln_0_b;

                model.tensors["decoder.blocks." + std::to_string(i) +
                              ".attn.query.weight"] = layer.attn_q_w;
                model.tensors["decoder.blocks." + std::to_string(i) +
                              ".attn.query.bias"] = layer.attn_q_b;

                model.tensors["decoder.blocks." + std::to_string(i) +
                              ".attn.key.weight"] = layer.attn_k_w;

                model.tensors["decoder.blocks." + std::to_string(i) +
                              ".attn.value.weight"] = layer.attn_v_w;
                model.tensors["decoder.blocks." + std::to_string(i) +
                              ".attn.value.bias"] = layer.attn_v_b;

                model.tensors["decoder.blocks." + std::to_string(i) +
                              ".attn.out.weight"] = layer.attn_ln_1_w;
                model.tensors["decoder.blocks." + std::to_string(i) +
                              ".attn.out.bias"] = layer.attn_ln_1_b;

                model.tensors["decoder.blocks." + std::to_string(i) +
                              ".cross_attn_ln.weight"] =
                    layer.cross_attn_ln_0_w;
                model.tensors["decoder.blocks." + std::to_string(i) +
                              ".cross_attn_ln.bias"] = layer.cross_attn_ln_0_b;

                model.tensors["decoder.blocks." + std::to_string(i) +
                              ".cross_attn.query.weight"] =
                    layer.cross_attn_q_w;
                model.tensors["decoder.blocks." + std::to_string(i) +
                              ".cross_attn.query.bias"] = layer.cross_attn_q_b;

                model.tensors["decoder.blocks." + std::to_string(i) +
                              ".cross_attn.key.weight"] = layer.cross_attn_k_w;

                model.tensors["decoder.blocks." + std::to_string(i) +
                              ".cross_attn.value.weight"] =
                    layer.cross_attn_v_w;
                model.tensors["decoder.blocks." + std::to_string(i) +
                              ".cross_attn.value.bias"] = layer.cross_attn_v_b;

                model.tensors["decoder.blocks." + std::to_string(i) +
                              ".cross_attn.out.weight"] =
                    layer.cross_attn_ln_1_w;
                model.tensors["decoder.blocks." + std::to_string(i) +
                              ".cross_attn.out.bias"] = layer.cross_attn_ln_1_b;
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
                log("%s: unknown tensor '%s' in model file\n", __func__,
                    name.data());
                return false;
            }

            auto tensor = model.tensors[name.data()];
            if (ggml_nelements(tensor) != nelements) {
                log("%s: tensor '%s' has wrong size in model file\n", __func__,
                    name.data());
                log("%s: shape: [%d, %d, %d], expected: [%d, %d, %d]\n",
                    __func__, ne[0], ne[1], ne[2], (int)tensor->ne[0],
                    (int)tensor->ne[1], (int)tensor->ne[2]);
                return false;
            }

            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1] ||
                tensor->ne[2] != ne[2]) {
                log("%s: tensor '%s' has wrong shape in model file: got [%d, "
                    "%d, %d], expected [%d, %d, %d]\n",
                    __func__, name.data(), (int)tensor->ne[0],
                    (int)tensor->ne[1], (int)tensor->ne[2], ne[0], ne[1],
                    ne[2]);
                return false;
            }

            const size_t bpe = ggml_type_size(ggml_type(ttype));

            if ((nelements * bpe) / ggml_blck_size(tensor->type) !=
                ggml_nbytes(tensor)) {
                log("%s: tensor '%s' has wrong size in model file: got %zu, "
                    "expected %zu\n",
                    __func__, name.data(), ggml_nbytes(tensor),
                    nelements * bpe);
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

        log("%s: model size    = %7.2f MB\n", __func__,
            total_size / 1024.0 / 1024.0);

        if (model.n_loaded == 0) {
            log("%s: WARN no tensors loaded from model file - assuming empty "
                "model for testing\n",
                __func__);
        } else if (model.n_loaded != (int)model.tensors.size()) {
            log("%s: ERROR not all tensors loaded from model file - expected "
                "%zu, got %d\n",
                __func__, model.tensors.size(), model.n_loaded);
            return false;
        }
    }

    wctx.t_load_us = ggml_time_us() - t_start_us;

    return true;
}