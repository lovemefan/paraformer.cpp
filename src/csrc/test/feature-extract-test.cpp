//
// Created by lovemefan on 2023/11/18.
//
#include <time.h>

#include <chrono>

#include "paraformer-frontend.h"

int main() {
    std::vector<double> samples;
    int sample_rate;
    std::string wav_file = "/Users/cenglingfan/Code/cpp-project/paraformer.cpp/resource/model/test.wav";
    load_wav_file(wav_file.c_str(), &sample_rate, samples);
    int n_samples = samples.size();
    int frame_size = 25;
    int frame_step = 10;
    int n_mel = 80;
    int n_thread = 4;
    std::string cmvn_file = "/Users/cenglingfan/Code/cpp-project/paraformer.cpp/resource/model/am.mvn";
    struct paraformer_cmvn cmvn;
    load_cmvn(cmvn_file.c_str(), cmvn);
    paraformer_feature mel;
    auto start = std::chrono::steady_clock::now();
    fbank_lfr_cmvn_feature(samples, n_samples, frame_size, frame_step, n_mel, n_thread, true, cmvn, mel);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    printf("it costed %f s", diff.count());
}