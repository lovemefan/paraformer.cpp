//
// Created by lovemefan on 2023/11/18.
//
#include "paraformer-frontend.h"

int main() {
    int n_samples = 16000;
    float *samples = new float[n_samples];
    int frame_size = 25;
    int frame_step = 10;
    int n_mel = 80;
    int n_thread = 1;
    paraformer_filters filter;

    paraformer_mel mel;
    fbank_lfr_cmvn_feature(samples, n_samples, frame_size, frame_step, n_mel, n_thread, filter, true, mel);
}