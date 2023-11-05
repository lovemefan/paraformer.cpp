//
// Created by lovemefan on 2023/11/5.
//

#include <fstream>

#include "../paraformer-offline.h"

int main() {
    std::string path_model =
        "/Users/cenglingfan/Code/cpp-project/paraformer.cpp/resource/model/"
        "paraformer-ggml-model.bin";
    printf("%s: loading model from '%s'\n", __func__, path_model.c_str());

    auto fin = std::ifstream(path_model, std::ios::binary);
    if (!fin) {
        printf("%s: failed to open '%s'\n", __func__, path_model.c_str());
    }

    paraformer_model_loader loader = {};

    struct paraformer_model_loader model_loader = {};
    model_loader.context = &fin;
    model_loader.read = [](void *ctx, void *output, size_t read_size) {
        auto fin = (std::ifstream *)ctx;
        fin->read((char *)output, read_size);
        return read_size;
    };

    model_loader.eof = [](void *ctx) {
        auto *fin = (std::ifstream *)ctx;
        return fin->eof();
    };

    model_loader.close = [](void *ctx) {
        auto *fin = (std::ifstream *)ctx;
        fin->close();
    };
    struct paraformer_context context;
    paraformer_model_load(&model_loader, context);
}