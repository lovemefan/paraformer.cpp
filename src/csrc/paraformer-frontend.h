//
// Created by lovemefan on 2023/10/3.
//
#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>
#define PARAFORMER_SAMPLE_RATE 16000
#define PREEMPH_COEFF 0.97
struct paraformer_mel {
    int n_len;
    int n_len_org;
    int n_mel;
    float low_freq = 20.0f;
    float high_freq = 0.0f;
    float vtln_high = -500.0f;
    float vtln_low = 100.0f;
    int lfr_n = 6;
    int lfr_m = 7;
    std::vector<float> data;
};

struct paraformer_filters {
    int32_t n_mel;
    int32_t n_fft;

    std::vector<float> data;
};
void fill_sin_cos_table();
void load_cmvn(const char *filename, std::vector<float> &means_list, std::vector<float> &vars_list);
bool fbank_lfr_cmvn_feature(const std::vector<float> &samples, const int n_samples, const int frame_size,
                            const int frame_step, const int n_mel, const int n_threads, const bool debug,
                            const std::vector<float> &cmvn_mean, std::vector<float> &cmvn_var, paraformer_mel &mel);