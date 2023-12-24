//
// Created by lovemefan on 2023/11/18.
//
#include <time.h>

#include "paraformer-frontend.h"

int main() {
    std::vector<double> samples;
    int sample_rate;
    char *wav_file = "/Users/cenglingfan/Code/cpp-project/paraformer.cpp/resource/model/test.wav";
    load_wav_file(wav_file, &sample_rate, samples);
    int n_samples = samples.size();
    int frame_size = 25;
    int frame_step = 10;
    int n_mel = 80;
    int n_thread = 2;
    char *cmvn_file = "/Users/cenglingfan/Code/cpp-project/paraformer.cpp/resource/model/am.mvn";
    struct paraformer_cmvn cmvn;
    load_cmvn(cmvn_file, cmvn);
    paraformer_mel mel;
    clock_t start = clock();
    fbank_lfr_cmvn_feature(samples, n_samples, frame_size, frame_step, n_mel, n_thread, true, cmvn, mel);
    clock_t end = clock();
    printf("it costed %f s", (double)(end - start) / CLOCKS_PER_SEC);
}