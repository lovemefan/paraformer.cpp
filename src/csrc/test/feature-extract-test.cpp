//
// Created by lovemefan on 2023/11/18.
//
#include <time.h>
#include <unistd.h>

#include <random>

#include "paraformer-frontend.h"

int main() {
    int n_samples = 16000;
    std::vector<float> samples;
    std::default_random_engine e;
    e.seed(time(0));
    std::normal_distribution<float> u(0, 1);
    samples.resize(n_samples);
    for (int i = 0; i < n_samples; i++) {
        samples[i] = u(e);
    }
    int frame_size = 25;
    int frame_step = 10;
    int n_mel = 80;
    int n_thread = 4;
    char *cmvn_file =
        "/Users/cenglingfan/Code/cpp-project/paraformer.cpp/resource/model/"
        "speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404/am.mvn";
    std::vector<float> cmvn_means;
    std::vector<float> cmvn_var;
    load_cmvn(cmvn_file, cmvn_means, cmvn_var);
    paraformer_mel mel;
    fill_sin_cos_table();
    clock_t start = clock();
    fbank_lfr_cmvn_feature(samples, n_samples, frame_size, frame_step, n_mel, n_thread, true, cmvn_means, cmvn_var,
                           mel);
    clock_t end = clock();
    printf("it costed %f s", (double)(end - start) / CLOCKS_PER_SEC);
}