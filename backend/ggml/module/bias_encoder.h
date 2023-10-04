//
// Created by lovemefan on 2023/10/1.
//

#ifndef PARAFORMER_CPP_BIAS_ENCODER_H
#define PARAFORMER_CPP_BIAS_ENCODER_H


struct paraformer_bias_encoder {
    // bias encoder is a lstm model

    struct ggml_tensor * bias_embed;

    // bias_encoder.weight_ih_l0
    struct ggml_tensor * ih_l_w;
    struct ggml_tensor * ihl_b;

    // bias_encoder.weight_hh_l0
    struct ggml_tensor * hh_l_w;
    struct ggml_tensor * hh_l_b;


};


#endif //PARAFORMER_CPP_BIAS_ENCODER_H
