//
// Created by lovemefan on 2023/10/3.
//

#include "paraformer-frontend.h"
#define SIN_COS_N_COUNT 80

static float sin_vals[SIN_COS_N_COUNT];
static float cos_vals[SIN_COS_N_COUNT];

// In FFT, we frequently use sine and cosine operations with the same values.
// We can use precalculated values to speed up the process.
static void fill_sin_cos_table() {
    static bool is_filled = false;
    if (is_filled) return;
    for (int i = 0; i < SIN_COS_N_COUNT; i++) {
        double theta = (2 * M_PI * i) / SIN_COS_N_COUNT;
        sin_vals[i] = sinf(theta);
        cos_vals[i] = cosf(theta);
    }
    is_filled = true;
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

static bool hann_window(int length, bool periodic, std::vector<float> &output) {
    if (output.size() < static_cast<size_t>(length)) {
        output.resize(length);
    }
    int offset = -1;
    if (periodic) {
        offset = 0;
    }
    for (int i = 0; i < length; i++) {
        output[i] = 0.5 * (1.0 - cosf((2.0 * M_PI * i) / (length + offset)));
    }

    return true;
}

static void fbank_lfr_cmvn_feature_worker_thread(int ith, const std::vector<float> &hann,
                                                 const std::vector<float> &samples, int n_samples, int frame_size,
                                                 int frame_step, int n_threads, const paraformer_filters &filters,
                                                 paraformer_mel &mel) {
    std::vector<float> fft_in(frame_size, 0.0);
    std::vector<float> fft_out(2 * frame_step);
    // make sure n_fft == 1 + (PARAFORMER_N_FFT / 2), bin_0 to bin_nyquist
    int n_fft = 1 + (frame_size / 2);
    int i = ith;

    // calculate FFT only when fft_in are not all zero
    for (; i < std::min(n_samples / frame_step + 1, mel.n_len); i += n_threads) {
        const int offset = i * frame_step;

        // apply Hanning window (~10% faster)
        for (int j = 0; j < std::min(frame_size, n_samples - offset); j++) {
            fft_in[j] = hann[j] * samples[offset + j];
        }
        // fill the rest with zeros
        if (n_samples - offset < frame_size) {
            std::fill(fft_in.begin() + (n_samples - offset), fft_in.end(), 0.0);
        }

        // FFT
        fft(fft_in, fft_out);

        // Calculate modulus^2 of complex numbers
        // Use pow(fft_out[2 * j + 0], 2) + pow(fft_out[2 * j + 1], 2) causes inference quality problem? Interesting.
        for (int j = 0; j < frame_size; j++) {
            fft_out[j] = (fft_out[2 * j + 0] * fft_out[2 * j + 0] + fft_out[2 * j + 1] * fft_out[2 * j + 1]);
        }

        // mel spectrogram
        for (int j = 0; j < mel.n_mel; j++) {
            double sum = 0.0;

            // unroll loop (suggested by GH user @lunixbochs)
            int k = 0;
            for (k = 0; k < n_fft - 3; k += 4) {
                sum += fft_out[k + 0] * filters.data[j * n_fft + k + 0] +
                       fft_out[k + 1] * filters.data[j * n_fft + k + 1] +
                       fft_out[k + 2] * filters.data[j * n_fft + k + 2] +
                       fft_out[k + 3] * filters.data[j * n_fft + k + 3];
            }

            // handle n_fft remainder
            for (; k < n_fft; k++) {
                sum += fft_out[k] * filters.data[j * n_fft + k];
            }

            sum = log10(std::max(sum, 1e-10));

            mel.data[j * mel.n_len + i] = sum;
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

// ref: https://github.com/ggerganov/ggml/blob/master/examples/whisper/whisper.cpp#L2833C24-L2833C24
bool fbank_lfr_cmvn_feature(const float *samples, const int n_samples, const int frame_size, const int frame_step,
                            const int n_mel, const int n_threads, paraformer_filters &filters, const bool debug,
                            paraformer_mel &mel) {
    //    const int64_t t_start_us = ggml_time_us();

    // Hanning window (Use cosf to eliminate difference)
    // ref: https://pytorch.org/docs/stable/generated/torch.hann_window.html
    // ref: https://github.com/openai/whisper/blob/main/paraformer/audio.py#L147
    std::vector<float> hann;
    hann_window(frame_size, true, hann);

    // Calculate the length of padding
    int64_t stage_1_pad = PARAFORMER_SAMPLE_RATE * 30;
    int64_t stage_2_pad = frame_size / 2;

    // Initialize a vector and copy data from C array to it.
    std::vector<float> samples_padded;
    samples_padded.resize(n_samples + stage_1_pad + stage_2_pad * 2);
    std::copy(samples, samples + n_samples, samples_padded.begin() + stage_2_pad);

    // pad 30 seconds of zeros at the end of audio (480,000 samples) + reflective pad 200 samples at the end of audio
    std::fill(samples_padded.begin() + n_samples + stage_2_pad,
              samples_padded.begin() + n_samples + stage_1_pad + 2 * stage_2_pad, 0);

    // reflective pad 200 samples at the beginning of audio
    std::reverse_copy(samples + 1, samples + 1 + stage_2_pad, samples_padded.begin());

    mel.n_mel = n_mel;
    // https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/SpectralOps.cpp#L936
    // Calculate number of frames + remove the last frame
    mel.n_len = (samples_padded.size() - frame_size) / frame_step;
    // Calculate semi-padded sample length to ensure compatibility
    mel.n_len_org = 1 + (n_samples + stage_2_pad - frame_size) / frame_step;
    mel.data.resize(mel.n_mel * mel.n_len);

    {
        std::vector<std::thread> workers(n_threads - 1);
        for (int iw = 0; iw < n_threads - 1; ++iw) {
            workers[iw] = std::thread(fbank_lfr_cmvn_feature_worker_thread, iw + 1, std::cref(hann), samples_padded,
                                      n_samples + stage_2_pad, frame_size, frame_step, n_threads, std::cref(filters),
                                      std::ref(mel));
        }

        // main thread
        fbank_lfr_cmvn_feature_worker_thread(0, hann, samples_padded, n_samples + stage_2_pad, frame_size, frame_step,
                                             n_threads, filters, mel);

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

    return true;
}