//
// Created by lovemefan on 2023/11/5.
//

#include <fstream>

#include "paraformer-contextual-offline.h"

int main() {
    std::string path_model =
        "/Users/cenglingfan/Code/cpp-project/paraformer.cpp/resource/model/"
        "paraformer-ggml-model-fp16.bin";
    printf("%s: loading model from '%s'\n", __func__, path_model.c_str());

    struct paraformer_context *context = paraformer_init_from_file(path_model.c_str());

    //    ggml_cgraph *gf = paraformer_build_graph_encoder(*context, *context->state);
    //    gf->nodes;
}