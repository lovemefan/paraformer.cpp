//
// Created by lovemefan on 2023/10/3.
//
#pragma once
#include <fstream>
#include <thread>

#define PARAFORMER_SAMPLE_RATE 16000
struct paraformer_mel {
    int n_len;
    int n_len_org;
    int n_mel;

    std::vector<float> data;
};

struct paraformer_filters {
    int32_t n_mel;
    int32_t n_fft;

    std::vector<float> data;
};

bool fbank_lfr_cmvn_feature(const float *samples, const int n_samples, const int frame_size, const int frame_step,
                            const int n_mel, const int n_threads, paraformer_filters &filters, const bool debug,
                            paraformer_mel &mel);