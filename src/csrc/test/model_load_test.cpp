//
// Created by lovemefan on 2023/11/5.
//

#include "../paraformer-offline.h"

int main() {
    struct paraformer_model_loader model_loader {};
    struct paraformer_context context;
    paraformer_model_load(&model_loader, context);
}