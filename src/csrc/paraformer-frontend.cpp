//
// Created by lovemefan on 2023/10/3.
//

#include "paraformer-frontend.h"
#define SIN_COS_N_COUNT 80

static float sin_vals[SIN_COS_N_COUNT];
static float cos_vals[SIN_COS_N_COUNT];

#define M_2PI 6.283185307179586476925286766559005

// In FFT, we frequently use sine and cosine operations with the same values.
// We can use precalculated values to speed up the process.
void fill_sin_cos_table() {
    static bool is_filled = false;
    if (is_filled) return;
    for (int i = 0; i < SIN_COS_N_COUNT; i++) {
        double theta = (2 * M_PI * i) / SIN_COS_N_COUNT;
        sin_vals[i] = sinf(theta);
        cos_vals[i] = cosf(theta);
    }
    is_filled = true;
}

static void pre_emphasize(float *d, int32_t n, float pre_emph_coeff) {
    if (pre_emph_coeff == 0.0) {
        return;
    }

    assert(pre_emph_coeff >= 0.0 && pre_emph_coeff <= 1.0);

    for (int32_t i = n - 1; i > 0; --i) {
        d[i] -= pre_emph_coeff * d[i - 1];
    }
    d[0] -= pre_emph_coeff * d[0];
}

// naive Discrete Fourier Transform
// input is real-valued
// output is complex-valued
static void dft(const std::vector<float> &in, std::vector<float> &out) {
    int N = in.size();

    out.resize(N * 2);
    const int sin_cos_step = SIN_COS_N_COUNT / N;

    for (int k = 0; k < N; k++) {
        float re = 0;
        float im = 0;

        for (int n = 0; n < N; n++) {
            int idx = (k * n * sin_cos_step) % (SIN_COS_N_COUNT);  // t = 2*M_PI*k*n/N
            re += in[n] * cos_vals[idx];                           // cos(t)
            im -= in[n] * sin_vals[idx];                           // sin(t)
        }

        out[k * 2 + 0] = re;
        out[k * 2 + 1] = im;
    }
}

// Cooley-Tukey FFT
// poor man's implementation - use something better
// input is real-valued
// output is complex-valued
// ref: // https://github.com/ggerganov/whisper.cpp/blob/master/whisper.cpp#L2373C1-L2429C1
static void fft(const std::vector<float> &in, std::vector<float> &out) {
    out.resize(in.size() * 2);

    int N = in.size();

    if (N == 1) {
        out[0] = in[0];
        out[1] = 0;
        return;
    }

    if (N % 2 == 1) {
        dft(in, out);
        return;
    }

    std::vector<float> even;
    std::vector<float> odd;

    even.reserve(N / 2);
    odd.reserve(N / 2);

    for (int i = 0; i < N; i++) {
        if (i % 2 == 0) {
            even.push_back(in[i]);
        } else {
            odd.push_back(in[i]);
        }
    }

    std::vector<float> even_fft;
    std::vector<float> odd_fft;

    fft(even, even_fft);
    fft(odd, odd_fft);

    const int sin_cos_step = SIN_COS_N_COUNT / N;
    for (int k = 0; k < N / 2; k++) {
        int idx = k * sin_cos_step;  // t = 2*M_PI*k/N
        float re = cos_vals[idx];    // cos(t)
        float im = -sin_vals[idx];   // sin(t)

        float re_odd = odd_fft[2 * k + 0];
        float im_odd = odd_fft[2 * k + 1];

        out[2 * k + 0] = even_fft[2 * k + 0] + re * re_odd - im * im_odd;
        out[2 * k + 1] = even_fft[2 * k + 1] + re * im_odd + im * re_odd;

        out[2 * (k + N / 2) + 0] = even_fft[2 * k + 0] - re * re_odd + im * im_odd;
        out[2 * (k + N / 2) + 1] = even_fft[2 * k + 1] - re * im_odd - im * re_odd;
    }
}

inline int32_t round_to_nearest_power_two(int32_t n) {
    // copied from kaldi/src/base/kaldi-math.cc
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

static bool hamming_window(int length, bool periodic, std::vector<float> &output) {
    if (output.size() < static_cast<size_t>(length)) {
        output.resize(length);
    }
    int offset = -1;
    if (periodic) {
        offset = 0;
    }
    for (int i = 0; i < length; i++) {
        output[i] = 0.54 - 0.46 * cosf((M_2PI * i) / (length + offset));
    }

    return true;
}

static inline float inverse_mel_scale(float mel_freq) { return 700.0f * (expf(mel_freq / 1127.0f) - 1.0f); }

static inline float mel_scale(float freq) { return 1127.0f * logf(1.0f + freq / 700.0f); }

static void fbank_feature_worker_thread(int ith, const std::vector<float> &hamming, const std::vector<float> &samples,
                                        int n_samples, int frame_size, int frame_step, int n_threads,
                                        paraformer_mel &mel) {
    std::vector<float> fft_out(2 * frame_step);
    // make sure n_fft == 1 + (PARAFORMER_N_FFT / 2), bin_0 to bin_nyquist
    int n_fft = 1 + (frame_size / 2);
    int i = ith;

    std::vector<float> window;
    const int padded_window_size = round_to_nearest_power_two(frame_size);
    window.resize(padded_window_size);

    // calculate FFT only when fft_in are not all zero
    for (; i < std::min(n_samples / frame_step + 1, mel.n_len); i += n_threads) {
        const int offset = i * frame_step;

        std::copy(samples.begin() + offset, samples.begin() + offset + frame_size, window.begin());

        // remove dc offset
        {
            float sum = 0;
            for (int32_t i = 0; i != frame_size; ++i) {
                sum += window[i];
            }
            float mean = sum / frame_size;
            for (int32_t i = 0; i != frame_size; ++i) {
                window[i] -= mean;
            }
        }
        // pre-emphasis
        {
            for (int32_t i = frame_size - 1; i > 0; --i) {
                window[i] -= PREEMPH_COEFF * window[i - 1];
            }
            window[0] -= PREEMPH_COEFF * window[0];
        }

        // apply Hamming window
        {
            for (int j = 0; j < frame_size; j++) {
                window[j] *= hamming[j];
            }
        }

        // pad columns with zero
        if (padded_window_size - frame_size > 0) {
            std::fill(window.begin() + (padded_window_size - frame_size), window.end(), 0.0);
        }

        // FFT
        fft(window, fft_out);

        // Calculate modulus^2 of complex numbers，Power Spectrum
        // Use pow(fft_out[2 * j + 0], 2) + pow(fft_out[2 * j + 1], 2) causes inference quality problem? Interesting.
        for (int j = 0; j < padded_window_size; j++) {
            fft_out[j] = (fft_out[2 * j + 0] * fft_out[2 * j + 0] + fft_out[2 * j + 1] * fft_out[2 * j + 1]);
        }

        // log-Mel filter bank energies aka: "fbank"
        {
            auto num_fft_bins = padded_window_size / 2;
            auto nyquist = 0.5 * PARAFORMER_SAMPLE_RATE;
            auto low_freq = mel.low_freq;
            auto high_freq = mel.high_freq;
            auto vtln_high = mel.vtln_high;
            auto vtln_low = mel.vtln_low;

            if (high_freq <= 0) high_freq += nyquist;

            // fft-bin width [think of it as Nyquist-freq / half-window-length]
            float fft_bin_width = PARAFORMER_SAMPLE_RATE * 1.0 / padded_window_size;
            float mel_low_freq = mel_scale(low_freq);
            float mel_high_freq = mel_scale(high_freq);

            // divide by num_bins+1 in next line because of end-effects where the bins
            // spread out to the sides.
            auto mel_freq_delta = (mel_high_freq - mel_low_freq) / (mel.n_mel + 1);
            if (vtln_high < 0.0f) vtln_high += nyquist;

            for (int j = 0; j < mel.n_mel; j++) {
                double sum = 0.0;
                float left_mel = mel_low_freq + j * mel_freq_delta;
                float center_mel = mel_low_freq + (j + 1.0) * mel_freq_delta;
                float right_mel = mel_low_freq + (j + 2.0) * mel_freq_delta;

                for (int k = 0; k < num_fft_bins; k++) {
                    auto mel_num = mel_scale(fft_bin_width * k);
                    auto up_slope = (mel_num - left_mel) / (center_mel - left_mel);
                    auto down_slope = (right_mel - mel_num) / (right_mel - center_mel);
                    auto filter = std::max(0.0f, std::min(up_slope, down_slope));
                    sum += fft_out[k] * filter;
                }

                sum = log10(std::max(sum, 1e-10));

                mel.data[j * mel.n_len + i] = sum;
            }
        }
    }

    // Otherwise fft_out are all zero
    double sum = log10(1e-10);
    for (; i < mel.n_len; i += n_threads) {
        for (int j = 0; j < mel.n_mel; j++) {
            mel.data[j * mel.n_len + i] = sum;
        }
    }
}

bool fbank_lfr_cmvn_feature(const std::vector<float> &samples, const int n_samples, const int frame_size,
                            const int frame_step, const int n_mel, const int n_threads, const bool debug,
                            paraformer_mel &mel) {
    //    const int64_t t_start_us = ggml_time_us();

    const int32_t n_frames_per_ms = PARAFORMER_SAMPLE_RATE * 0.001f;
    mel.n_mel = n_mel;
    mel.n_len = 1 + ((n_samples - frame_size * n_frames_per_ms) / (frame_step * n_frames_per_ms));
    mel.data.resize(mel.n_mel * mel.n_len);

    std::vector<float> hamming;
    hamming_window(frame_size, true, hamming);

    {
        std::vector<std::thread> workers(n_threads - 1);
        for (int iw = 0; iw < n_threads - 1; ++iw) {
            workers[iw] =
                std::thread(fbank_feature_worker_thread, iw + 1, std::cref(hamming), samples, n_samples,
                            frame_size * n_frames_per_ms, frame_step * n_frames_per_ms, n_threads, std::ref(mel));
        }

        // main thread
        fbank_feature_worker_thread(0, hamming, samples, n_samples, frame_size * n_frames_per_ms,
                                    frame_step * n_frames_per_ms, n_threads, mel);

        for (int iw = 0; iw < n_threads - 1; ++iw) {
            workers[iw].join();
        }
    }

    // clamping and normalization
    double mmax = -1e20;
    for (int i = 0; i < mel.n_mel * mel.n_len; i++) {
        if (mel.data[i] > mmax) {
            mmax = mel.data[i];
        }
    }

    mmax -= 8.0;

    for (int i = 0; i < mel.n_mel * mel.n_len; i++) {
        if (mel.data[i] < mmax) {
            mel.data[i] = mmax;
        }

        mel.data[i] = (mel.data[i] + 4.0) / 4.0;
    }

    //    wstate.t_mel_us += ggml_time_us() - t_start_us;

    // Dump fbank_lfr_cmvn_feature
    if (debug) {
        std::ofstream outFile("fbank_lfr_cmvn_feature.json");
        outFile << "[";
        for (uint64_t i = 0; i < mel.data.size() - 1; i++) {
            outFile << mel.data[i] << ", ";
        }
        outFile << mel.data[mel.data.size() - 1] << "]";
        outFile.close();
    }

    // todo  apply lrf, merge lfr_m frames as one,lfr_n frames per window
    // ref: https://github.com/alibaba-damo-academy/FunASR/blob/main/runtime/onnxruntime/src/paraformer.cpp#L409-L440

    // todo apply cvmn
    return true;
}