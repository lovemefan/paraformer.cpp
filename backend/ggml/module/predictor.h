//
// Created by lovemefan on 2023/10/1.
//

#ifndef PARAFORMER_CPP_PREDICTOR_H
#define PARAFORMER_CPP_PREDICTOR_H




struct paraformer_predictor {
    // predictor.cif_conv1d.weight
    struct ggml_tensor * cif_conv1d_w;
    struct ggml_tensor * cif_conv1d_b;

    struct ggml_tensor * cif_ln_out_w;
    struct ggml_tensor * cif_ln_out_b;

};


#endif //PARAFORMER_CPP_PREDICTOR_H