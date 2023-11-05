//
// Created by lovemefan on 2023/10/4.
//

#include "utils.h"
#include <vector>
#include <functional>
#include <fstream>
//
// ggml helpers
//

static void ggml_graph_compute_helper(std::vector<uint8_t> & buf, ggml_cgraph * graph, int n_threads) {
    struct ggml_cplan plan = ggml_graph_plan(graph, n_threads);

    if (plan.work_size > 0) {
        buf.resize(plan.work_size);
        plan.work_data = buf.data();
    }

    ggml_graph_compute(graph, &plan);
}

// faster matrix multiplications for tensors that do not have dimension 0 divisible by "pad"
// the idea is to represent the original matrix multiplication:
//
//   Z = X @ Y
//
// with the sum of two matrix multiplications:
//
//   Z = (X_0 @ Y_0) + (X_1 @ Y_1)
//
// here X_0 and Y_0 are views of X and Y that have dimension 0 divisible by "pad"
// and X_1 and Y_1 are the remaining views. X_1 and Y_1 end up being small matrices that can be processed with more
// general-purpose kernels
//


static struct ggml_tensor * ggml_mul_mat_pad(struct ggml_context * ctx, struct ggml_tensor * x, struct ggml_tensor * y, int pad = 32) {
    // use padding only if dimension 0 is at least 8 times larger than the padding
    // else we won't get much benefit from the optimization
    const int n_pad_req = 8;

    if (x->ne[0] % pad == 0 || x->ne[0] / pad < n_pad_req) {
        return ggml_mul_mat(ctx, x, y);
    }

    struct ggml_tensor * x_0 = ggml_view_3d(ctx, x, (x->ne[0]/pad)*pad, x->ne[1], x->ne[2], x->nb[1], x->nb[2], 0);
    struct ggml_tensor * x_1 = ggml_view_3d(ctx, x,  x->ne[0]%pad,      x->ne[1], x->ne[2], x->nb[1], x->nb[2], x_0->ne[0]*x_0->nb[0]);

    struct ggml_tensor * y_0 = ggml_view_3d(ctx, y, (y->ne[0]/pad)*pad, y->ne[1], y->ne[2], y->nb[1], y->nb[2], 0);
    struct ggml_tensor * y_1 = ggml_view_3d(ctx, y,  y->ne[0]%pad,      y->ne[1], y->ne[2], y->nb[1], y->nb[2], y_0->ne[0]*y_0->nb[0]);

    return ggml_add(ctx,
                    ggml_mul_mat(ctx, x_0, y_0),
                    ggml_mul_mat(ctx, x_1, y_1));
}

// TODO: check if other platforms can benefit from this optimization
#if defined(GGML_USE_METAL)
#define ggml_mul_mat ggml_mul_mat_pad
#endif



static size_t paraformer_allocr_size(struct paraformer_allocr & allocr) {
    return allocr.meta.size() + allocr.data.size();
}

// measure the memory usage of a graph and prepare the allocr's internal data buffer
static void paraformer_allocr_graph_init(struct paraformer_allocr & allocr, std::function<struct ggml_cgraph *()> && get_graph) {
    const int tensor_alignment = 32;

    auto & alloc = allocr.alloc;
    auto & meta  = allocr.meta;
    auto & data  = allocr.data;

    meta.resize(ggml_tensor_overhead()*GGML_MAX_NODES + ggml_graph_overhead());

    alloc = ggml_allocr_new_measure(tensor_alignment);

    const size_t alloc_size = ggml_allocr_alloc_graph(alloc, get_graph()) + tensor_alignment;

    ggml_allocr_free(alloc);

    data.resize(alloc_size);

    alloc = ggml_allocr_new(data.data(), data.size(), tensor_alignment);
}

static void paraformer_allocr_free(struct paraformer_allocr & allocr) {
    if (allocr.alloc) {
        ggml_allocr_free(allocr.alloc);
        allocr.alloc = nullptr;
    }
}


// =================================================================================================

//
// Temporary interface needed for exposing ggml interface
// Will be removed in the future when ggml becomes a separate library
//

PARAFORMER_API int paraformer_bench_memcpy(int n_threads) {
    fputs(paraformer_bench_memcpy_str(n_threads), stderr);
    return 0;
}

PARAFORMER_API const char * paraformer_bench_memcpy_str(int n_threads) {
    static std::string s;
    s = "";
    char strbuf[256];

    ggml_time_init();

    size_t n    = 20;
    size_t arr  = n_threads > 0 ? 1024llu : n_threads; // trick to avoid compiler optimizations

    // 1GB MB array
    const size_t size = arr*1024llu*1024llu;

    // single-thread
    {
        char * src = (char *) malloc(size);
        char * dst = (char *) malloc(size);

        for (size_t i = 0; i < size; i++) src[i] = i;

        memcpy(dst, src, size); // heat-up

        double tsum = 0.0;
        double sum  = 0.0;

        for (size_t i = 0; i < n; i++) {
            const int64_t t0 = ggml_time_us();

            memcpy(dst, src, size);

            const int64_t t1 = ggml_time_us();

            tsum += (t1 - t0)*1e-6;

            src[rand() % size] = rand() % 256;
        }

        snprintf(strbuf, sizeof(strbuf), "memcpy: %.2f GB/s (1 thread)\n", (double) (n*size)/(tsum*1024llu*1024llu*1024llu));
        s += strbuf;

        // needed to prevent the compiler from optimizing the memcpy away
        {
            for (size_t i = 0; i < size; i++) sum += dst[i];

            snprintf(strbuf, sizeof(strbuf), "sum:    %f\n", sum);
            s += strbuf;
        }

        free(src);
        free(dst);
    }

    return s.c_str();
}

PARAFORMER_API int paraformer_bench_ggml_mul_mat(int n_threads) {
    fputs(paraformer_bench_ggml_mul_mat_str(n_threads), stderr);
    return 0;
}

PARAFORMER_API const char * paraformer_bench_ggml_mul_mat_str(int n_threads) {
    static std::string s;
    s = "";
    char strbuf[256];

    ggml_time_init();

    const int n_max = 128;

    const std::vector<size_t> sizes = {
            64, 128, 256, 512, 1024, 2048, 4096,
    };

    const size_t N_max = sizes.back();

    // a: N*N*sizeof(float)
    // b: N*N*sizeof(float)
    // c: N*N*sizeof(float)
    // when F16 is used, there is an extra work buffer of size N*N*sizeof(float)
    std::vector<uint8_t> buf(3llu*N_max*N_max*sizeof(float) + 3*ggml_tensor_overhead());
    std::vector<uint8_t> work;

    // put a bunch of random data in the buffer
    for (size_t i = 0; i < buf.size(); i++) buf[i] = i;

    for (int j = 0; j < (int) sizes.size(); j++) {
        int n_q4_0 = 0;
        int n_q4_1 = 0;
        int n_q5_0 = 0;
        int n_q5_1 = 0;
        int n_q8_0 = 0;
        int n_fp16 = 0;
        int n_fp32 = 0;

        // GFLOPS/s
        double s_q4_0 = 0.0;
        double s_q4_1 = 0.0;
        double s_q5_0 = 0.0;
        double s_q5_1 = 0.0;
        double s_q8_0 = 0.0;
        double s_fp16 = 0.0;
        double s_fp32 = 0.0;

        const size_t N = sizes[j];

        for (int k = 0; k < 7; ++k) {
            const ggml_type wtype =
                    k == 0 ? GGML_TYPE_Q4_0 :
                    k == 1 ? GGML_TYPE_Q4_1 :
                    k == 2 ? GGML_TYPE_Q5_0 :
                    k == 3 ? GGML_TYPE_Q5_1 :
                    k == 4 ? GGML_TYPE_Q8_0 :
                    k == 5 ? GGML_TYPE_F16  : GGML_TYPE_F32;

            double & s = k == 0 ? s_q4_0 : k == 1 ? s_q4_1 : k == 2 ? s_q5_0 : k == 3 ? s_q5_1 : k == 4 ? s_q8_0 : k == 5 ? s_fp16 : /*k == 6*/ s_fp32;
            int    & n = k == 0 ? n_q4_0 : k == 1 ? n_q4_1 : k == 2 ? n_q5_0 : k == 3 ? n_q5_1 : k == 4 ? n_q8_0 : k == 5 ? n_fp16 : /*k == 6*/ n_fp32;

            struct ggml_init_params gparams = {
                    /*.mem_size   =*/ buf.size(),
                    /*.mem_buffer =*/ buf.data(),
                    /*.no_alloc   =*/ false,
            };

            struct ggml_context * ctx0 = ggml_init(gparams);

            struct ggml_tensor * a = ggml_new_tensor_2d(ctx0, wtype,         N, N);
            struct ggml_tensor * b = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, N, N);

            struct ggml_tensor * c = ggml_mul_mat(ctx0, a, b);

            struct ggml_cgraph gf = ggml_build_forward(c);

            double tsum = 0.0;

            // heat-up
            ggml_graph_compute_helper(work, &gf, n_threads);

            for (int i = 0; i < n_max; ++i) {
                const int64_t t0 = ggml_time_us();

                ggml_graph_compute_helper(work, &gf, n_threads);

                const int64_t t1 = ggml_time_us();

                tsum += (t1 - t0)*1e-6;
                n++;

                if (tsum > 1.0 && n >= 3) {
                    break;
                }
            }

            ggml_free(ctx0);

            s = ((2.0*N*N*N*n)/tsum)*1e-9;
        }

        // Q4_0 | Q4_1
        snprintf(strbuf, sizeof(strbuf), "%4zu x %4zu: Q4_0 %7.1f GFLOPS (%3d runs) | Q4_1 %7.1f GFLOPS (%3d runs)\n",
                 N, N, s_q4_0, n_q4_0, s_q4_1, n_q4_1);
        s += strbuf;

        // Q5_0 | Q5_1 | Q8_0
        snprintf(strbuf, sizeof(strbuf), "%4zu x %4zu: Q5_0 %7.1f GFLOPS (%3d runs) | Q5_1 %7.1f GFLOPS (%3d runs) | Q8_0 %7.1f GFLOPS (%3d runs)\n",
                 N, N, s_q5_0, n_q5_0, s_q5_1, n_q5_1, s_q8_0, n_q8_0);
        s += strbuf;

        // F16 | F32
        snprintf(strbuf, sizeof(strbuf), "%4zu x %4zu: F16  %7.1f GFLOPS (%3d runs) | F32  %7.1f GFLOPS (%3d runs)\n",
                 N, N, s_fp16, n_fp16, s_fp32, n_fp32);
        s += strbuf;
    }

    return s.c_str();
}

// =================================================================================================

// =================================================================================================

//
// Experimental stuff below
//
// Not sure if these should be part of the library at all, because the quality of the results is not
// guaranteed. Might get removed at some point unless a robust algorithm implementation is found
//

// =================================================================================================

//
// token-level timestamps
//

static int timestamp_to_sample(int64_t t, int n_samples) {
    return std::max(0, std::min((int) n_samples - 1, (int) ((t*PARAFORMER_SAMPLE_RATE)/100)));
}

static int64_t sample_to_timestamp(int i_sample) {
    return (100ll*i_sample)/PARAFORMER_SAMPLE_RATE;
}

// a cost-function / heuristic that is high for text that takes longer to pronounce
// obviously, can be improved
static float voice_length(const std::string & text) {
    float res = 0.0f;

    for (char c : text) {
        if (c == ' ') {
            res += 0.01f;
        } else if (c == ',') {
            res += 2.00f;
        } else if (c == '.') {
            res += 3.00f;
        } else if (c == '!') {
            res += 3.00f;
        } else if (c == '?') {
            res += 3.00f;
        } else if (c >= '0' && c <= '9') {
            res += 3.00f;
        } else {
            res += 1.00f;
        }
    }

    return res;
}

// average the fabs of the signal
static std::vector<float> get_signal_energy(const float * signal, int n_samples, int n_samples_per_half_window) {
    const int hw = n_samples_per_half_window;

    std::vector<float> result(n_samples);

    for (int i = 0; i < n_samples; i++) {
        float sum = 0;
        for (int j = -hw; j <= hw; j++) {
            if (i + j >= 0 && i + j < n_samples) {
                sum += fabs(signal[i + j]);
            }
        }
        result[i] = sum/(2*hw + 1);
    }

    return result;
}

static void paraformer_exp_compute_token_level_timestamps(
        struct paraformer_context & ctx,
        struct paraformer_state & state,
        int   i_segment,
        float   thold_pt,
        float   thold_ptsum) {
    auto & segment = state.result_all[i_segment];
    auto & tokens  = segment.tokens;

    const int n_samples = state.energy.size();

    if (n_samples == 0) {
        log("%s: no signal data available\n", __func__);
        return;
    }

    const int64_t t0 = segment.t0;
    const int64_t t1 = segment.t1;

    const int n = tokens.size();

    if (n == 0) {
        return;
    }

    if (n == 1) {
        tokens[0].t0 = t0;
        tokens[0].t1 = t1;

        return;
    }

    auto & t_beg    = state.t_beg;
    auto & t_last   = state.t_last;
    auto & tid_last = state.tid_last;

    for (int j = 0; j < n; ++j) {
        auto & token = tokens[j];

        if (j == 0) {
            if (token.id == paraformer_token_beg(&ctx)) {
                tokens[j    ].t0 = t0;
                tokens[j    ].t1 = t0;
                tokens[j + 1].t0 = t0;

                t_beg    = t0;
                t_last   = t0;
                tid_last = paraformer_token_beg(&ctx);
            } else {
                tokens[j    ].t0 = t_last;
            }
        }

        const int64_t tt = t_beg + 2*(token.tid - paraformer_token_beg(&ctx));

        tokens[j].id    = token.id;
        tokens[j].tid   = token.tid;
        tokens[j].p     = token.p;
        tokens[j].pt    = token.pt;
        tokens[j].ptsum = token.ptsum;

        tokens[j].vlen = voice_length(ctx.vocab.id_to_token.at(token.id).c_str());

        if (token.pt > thold_pt && token.ptsum > thold_ptsum && token.tid > tid_last && tt <= t1) {
            if (j > 0) {
                tokens[j - 1].t1 = tt;
            }
            tokens[j].t0 = tt;
            tid_last = token.tid;
        }
    }

    tokens[n - 2].t1 = t1;
    tokens[n - 1].t0 = t1;
    tokens[n - 1].t1 = t1;

    t_last = t1;

    // find intervals of tokens with unknown timestamps
    // fill the timestamps by proportionally splitting the interval based on the token voice lengths
    {
        int p0 = 0;
        int p1 = 0;

        while (true) {
            while (p1 < n && tokens[p1].t1 < 0) {
                p1++;
            }

            if (p1 >= n) {
                p1--;
            }

            //printf("p0=%d p1=%d t0=%lld t1=%lld\n", p0, p1, tokens[p0].t0, tokens[p1].t1);

            if (p1 > p0) {
                double psum = 0.0;
                for (int j = p0; j <= p1; j++) {
                    psum += tokens[j].vlen;
                }

                //printf("analyzing %d - %d, psum = %f\n", p0, p1, psum);

                const double dt = tokens[p1].t1 - tokens[p0].t0;

                // split the time proportionally to the voice length
                for (int j = p0 + 1; j <= p1; j++) {
                    const double ct = tokens[j - 1].t0 + dt*tokens[j - 1].vlen/psum;

                    tokens[j - 1].t1 = ct;
                    tokens[j    ].t0 = ct;
                }
            }

            p1++;
            p0 = p1;
            if (p1 >= n) {
                break;
            }
        }
    }

    // fix up (just in case)
    for (int j = 0; j < n - 1; j++) {
        if (tokens[j].t1 < 0) {
            tokens[j + 1].t0 = tokens[j].t1;
        }

        if (j > 0) {
            if (tokens[j - 1].t1 > tokens[j].t0) {
                tokens[j].t0 = tokens[j - 1].t1;
                tokens[j].t1 = std::max(tokens[j].t0, tokens[j].t1);
            }
        }
    }

    // VAD
    // expand or contract tokens based on voice activity
    {
        const int hw = PARAFORMER_SAMPLE_RATE/8;

        for (int j = 0; j < n; j++) {
            if (tokens[j].id >= paraformer_token_eot(&ctx)) {
                continue;
            }

            int s0 = timestamp_to_sample(tokens[j].t0, n_samples);
            int s1 = timestamp_to_sample(tokens[j].t1, n_samples);

            const int ss0 = std::max(s0 - hw, 0);
            const int ss1 = std::min(s1 + hw, n_samples);

            const int ns = ss1 - ss0;

            float sum = 0.0f;

            for (int k = ss0; k < ss1; k++) {
                sum += state.energy[k];
            }

            const float thold = 0.5*sum/ns;

            {
                int k = s0;
                if (state.energy[k] > thold && j > 0) {
                    while (k > 0 && state.energy[k] > thold) {
                        k--;
                    }
                    tokens[j].t0 = sample_to_timestamp(k);
                    if (tokens[j].t0 < tokens[j - 1].t1) {
                        tokens[j].t0 = tokens[j - 1].t1;
                    } else {
                        s0 = k;
                    }
                } else {
                    while (state.energy[k] < thold && k < s1) {
                        k++;
                    }
                    s0 = k;
                    tokens[j].t0 = sample_to_timestamp(k);
                }
            }

            {
                int k = s1;
                if (state.energy[k] > thold) {
                    while (k < n_samples - 1 && state.energy[k] > thold) {
                        k++;
                    }
                    tokens[j].t1 = sample_to_timestamp(k);
                    if (j < ns - 1 && tokens[j].t1 > tokens[j + 1].t0) {
                        tokens[j].t1 = tokens[j + 1].t0;
                    } else {
                        s1 = k;
                    }
                } else {
                    while (state.energy[k] < thold && k > s0) {
                        k--;
                    }
                    s1 = k;
                    tokens[j].t1 = sample_to_timestamp(k);
                }
            }
        }
    }

    // fixed token expand (optional)
    //{
    //    const int t_expand = 0;

    //    for (int j = 0; j < n; j++) {
    //        if (j > 0) {
    //            tokens[j].t0 = std::max(0, (int) (tokens[j].t0 - t_expand));
    //        }
    //        if (j < n - 1) {
    //            tokens[j].t1 = tokens[j].t1 + t_expand;
    //        }
    //    }
    //}

    // debug info
    //for (int j = 0; j < n; ++j) {
    //    const auto & token = tokens[j];
    //    const auto tt = token.pt > thold_pt && token.ptsum > 0.01 ? paraformer_token_to_str(&ctx, token.tid) : "[?]";
    //    printf("%s: %10s %6.3f %6.3f %6.3f %6.3f %5d %5d '%s'\n", __func__,
    //            tt, token.p, token.pt, token.ptsum, token.vlen, (int) token.t0, (int) token.t1, paraformer_token_to_str(&ctx, token.id));

    //    if (tokens[j].id >= paraformer_token_eot(&ctx)) {
    //        continue;
    //    }
    //}
}

void paraformer_set_log_callback(paraformer_log_callback callback) {
    paraformer_log = callback;
}

int paraformer_pcm_to_mel_with_state(struct paraformer_context * ctx, struct paraformer_state * state, const float * samples, int n_samples, int n_threads) {
    if (!log_mel_spectrogram(*state, samples, n_samples, PARAFORMER_SAMPLE_RATE, PARAFORMER_N_FFT, PARAFORMER_HOP_LENGTH, PARAFORMER_N_MEL, n_threads, ctx->model.filters, false, state->mel)) {
        log("%s: failed to compute mel spectrogram\n", __func__);
        return -1;
    }

    return 0;
}

int paraformer_pcm_to_mel(struct paraformer_context * ctx, const float * samples, int n_samples, int n_threads) {
    return paraformer_pcm_to_mel_with_state(ctx, ctx->state, samples, n_samples, n_threads);
}

// same as paraformer_pcm_to_mel, but applies a Phase Vocoder to speed up the audio x2 (PV without phase lock is not good)
int paraformer_pcm_to_mel_phase_vocoder_with_state(struct paraformer_context * ctx, struct paraformer_state * state, const float * samples, int n_samples, int n_threads) {
    if (!log_mel_spectrogram(*state, samples, n_samples, PARAFORMER_SAMPLE_RATE, 2 * PARAFORMER_N_FFT, 2 * PARAFORMER_HOP_LENGTH, PARAFORMER_N_MEL, n_threads, ctx->model.filters, false, state->mel)) {
        log("%s: failed to compute mel spectrogram\n", __func__);
        return -1;
    }

    return 0;
}

// same as paraformer_pcm_to_mel, but applies a Phase Vocoder to speed up the audio x2 (PV without phase lock is not good)
int paraformer_pcm_to_mel_phase_vocoder(struct paraformer_context * ctx, const float * samples, int n_samples, int n_threads) {
    return paraformer_pcm_to_mel_phase_vocoder_with_state(ctx, ctx->state, samples, n_samples, n_threads);
}

// same as paraformer_pcm_to_mel, but applies WSOLA to speed up the audio x2
// TODO

// same as paraformer_pcm_to_mel, but applies HPTSM to speed up the audio x2
// TODO

// same as paraformer_pcm_to_mel, but applies PV (with phase lock) to speed up the audio x2
// TODO

int paraformer_set_mel_with_state(
        struct paraformer_context * /*ctx*/,
        struct paraformer_state * state,
        const float * data,
        int   n_len,
        int   n_mel) {
    if (n_mel != PARAFORMER_N_MEL) {
        log("%s: invalid number of mel bands: %d (expected %d)\n", __func__, n_mel, PARAFORMER_N_MEL);
        return -1;
    }

    state->mel.n_len     = n_len;
    state->mel.n_len_org = n_len;
    state->mel.n_mel     = n_mel;

    state->mel.data.resize(n_len*n_mel);
    memcpy(state->mel.data.data(), data, n_len*n_mel*sizeof(float));

    return 0;
}

int paraformer_set_mel(
        struct paraformer_context * ctx,
        const float * data,
        int n_len,
        int n_mel) {
    return paraformer_set_mel_with_state(ctx, ctx->state, data, n_len, n_mel);
}