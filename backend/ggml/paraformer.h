//
// Created by lovemefan on 2023/9/20.
//

#ifndef PARAFORMER_CPP_PARAFORMER_H
#define PARAFORMER_CPP_PARAFORMER_H

#endif //PARAFORMER_CPP_PARAFORMER_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef PARAFORMER_SHARED
#    ifdef _WIN32
#        ifdef PARAFORMER_BUILD
#            define PARAFORMER_API __declspec(dllexport)
#        else
#            define PARAFORMER_API __declspec(dllimport)
#        endif
#    else
#        define PARAFORMER_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define PARAFORMER_API
#endif

#define PARAFORMER_SAMPLE_RATE 16000
#define PARAFORMER_N_FFT       400
#define PARAFORMER_N_MEL       80
#define PARAFORMER_HOP_LENGTH  160
#define PARAFORMER_CHUNK_SIZE  30

#ifdef __cplusplus
extern "C" {
#endif

//
// C interface
//
// The following interface is thread-safe as long as the sample paraformer_context is not used by multiple threads
// concurrently.
//
// Basic usage:
//
//     #include "paraformer.h"
//
//     ...
//
//     struct paraformer_context * ctx = paraformer_init_from_file("/path/to/ggml-base.en.bin");
//
//     if (paraformer_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
//         fprintf(stderr, "failed to process audio\n");
//         return 7;
//     }
//
//     const int n_segments = paraformer_full_n_segments(ctx);
//     for (int i = 0; i < n_segments; ++i) {
//         const char * text = paraformer_full_get_segment_text(ctx, i);
//         printf("%s", text);
//     }
//
//     paraformer_free(ctx);
//
//     ...
//
// This is a demonstration of the most straightforward usage of the library.
// "pcmf32" contains the RAW audio data in 32-bit floating point format.
//
// The interface also allows for more fine-grained control over the computation, but it requires a deeper
// understanding of how the model works.
//

struct paraformer_context;
struct paraformer_state;
struct paraformer_full_params;

typedef int paraformer_token;

typedef struct paraformer_token_data {
    paraformer_token id;  // token id
    paraformer_token tid; // forced timestamp token id

    float p;           // probability of the token
    float plog;        // log probability of the token
    float pt;          // probability of the timestamp token
    float ptsum;       // sum of probabilities of all timestamp tokens

    // token-level timestamp data
    // do not use if you haven't computed token-level timestamps
    int64_t t0;        // start time of the token
    int64_t t1;        //   end time of the token

    float vlen;        // voice length of the token
} paraformer_token_data;

typedef struct paraformer_model_loader {
    void * context;

    size_t (*read)(void * ctx, void * output, size_t read_size);
    bool    (*eof)(void * ctx);
    void  (*close)(void * ctx);
} paraformer_model_loader;

// Various functions for loading a ggml paraformer model.
// Allocate (almost) all memory needed for the model.
// Return NULL on failure
PARAFORMER_API struct paraformer_context * paraformer_init_from_file(const char * path_model);
PARAFORMER_API struct paraformer_context * paraformer_init_from_buffer(void * buffer, size_t buffer_size);
PARAFORMER_API struct paraformer_context * paraformer_init(struct paraformer_model_loader * loader);

// These are the same as the above, but the internal state of the context is not allocated automatically
// It is the responsibility of the caller to allocate the state using paraformer_init_state() (#523)
PARAFORMER_API struct paraformer_context * paraformer_init_from_file_no_state(const char * path_model);
PARAFORMER_API struct paraformer_context * paraformer_init_from_buffer_no_state(void * buffer, size_t buffer_size);
PARAFORMER_API struct paraformer_context * paraformer_init_no_state(struct paraformer_model_loader * loader);

PARAFORMER_API struct paraformer_state * paraformer_init_state(struct paraformer_context * ctx);

// Given a context, enable use of OpenVINO for encode inference.
// model_path: Optional path to OpenVINO encoder IR model. If set to nullptr,
//                      the path will be generated from the ggml model path that was passed
//                      in to paraformer_init_from_file. For example, if 'path_model' was
//                      "/path/to/ggml-base.en.bin", then OpenVINO IR model path will be
//                      assumed to be "/path/to/ggml-base.en-encoder-openvino.xml".
// device: OpenVINO device to run inference on ("CPU", "GPU", etc.)
// cache_dir: Optional cache directory that can speed up init time, especially for
//                     GPU, by caching compiled 'blobs' there.
//                     Set to nullptr if not used.
// Returns 0 on success. If OpenVINO is not enabled in build, this simply returns 1.
PARAFORMER_API int paraformer_ctx_init_openvino_encoder(
        struct paraformer_context * ctx,
        const char * model_path,
        const char * device,
        const char * cache_dir);

// Frees all allocated memory
PARAFORMER_API void paraformer_free      (struct paraformer_context * ctx);
PARAFORMER_API void paraformer_free_state(struct paraformer_state * state);
PARAFORMER_API void paraformer_free_params(struct paraformer_full_params * params);

// Convert RAW PCM audio to log mel spectrogram.
// The resulting spectrogram is stored inside the default state of the provided paraformer context.
// Returns 0 on success
PARAFORMER_API int paraformer_pcm_to_mel(
        struct paraformer_context * ctx,
        const float * samples,
        int   n_samples,
        int   n_threads);

PARAFORMER_API int paraformer_pcm_to_mel_with_state(
        struct paraformer_context * ctx,
        struct paraformer_state * state,
        const float * samples,
        int   n_samples,
        int   n_threads);

// Convert RAW PCM audio to log mel spectrogram but applies a Phase Vocoder to speed up the audio x2.
// The resulting spectrogram is stored inside the default state of the provided paraformer context.
// Returns 0 on success
PARAFORMER_API int paraformer_pcm_to_mel_phase_vocoder(
        struct paraformer_context * ctx,
        const float * samples,
        int   n_samples,
        int   n_threads);

PARAFORMER_API int paraformer_pcm_to_mel_phase_vocoder_with_state(
        struct paraformer_context * ctx,
        struct paraformer_state * state,
        const float * samples,
        int   n_samples,
        int   n_threads);

// This can be used to set a custom log mel spectrogram inside the default state of the provided paraformer context.
// Use this instead of paraformer_pcm_to_mel() if you want to provide your own log mel spectrogram.
// n_mel must be 80
// Returns 0 on success
PARAFORMER_API int paraformer_set_mel(
        struct paraformer_context * ctx,
        const float * data,
        int   n_len,
        int   n_mel);

PARAFORMER_API int paraformer_set_mel_with_state(
        struct paraformer_context * ctx,
        struct paraformer_state * state,
        const float * data,
        int   n_len,
        int   n_mel);

// Run the Whisper encoder on the log mel spectrogram stored inside the default state in the provided paraformer context.
// Make sure to call paraformer_pcm_to_mel() or paraformer_set_mel() first.
// offset can be used to specify the offset of the first frame in the spectrogram.
// Returns 0 on success
PARAFORMER_API int paraformer_encode(
        struct paraformer_context * ctx,
        int   offset,
        int   n_threads);

PARAFORMER_API int paraformer_encode_with_state(
        struct paraformer_context * ctx,
        struct paraformer_state * state,
        int   offset,
        int   n_threads);

// Run the Whisper decoder to obtain the logits and probabilities for the next token.
// Make sure to call paraformer_encode() first.
// tokens + n_tokens is the provided context for the decoder.
// n_past is the number of tokens to use from previous decoder calls.
// Returns 0 on success
// TODO: add support for multiple decoders
PARAFORMER_API int paraformer_decode(
        struct paraformer_context * ctx,
        const paraformer_token * tokens,
        int   n_tokens,
        int   n_past,
        int   n_threads);

PARAFORMER_API int paraformer_decode_with_state(
        struct paraformer_context * ctx,
        struct paraformer_state * state,
        const paraformer_token * tokens,
        int   n_tokens,
        int   n_past,
        int   n_threads);

// Convert the provided text into tokens.
// The tokens pointer must be large enough to hold the resulting tokens.
// Returns the number of tokens on success, no more than n_max_tokens
// Returns -1 on failure
// TODO: not sure if correct
PARAFORMER_API int paraformer_tokenize(
        struct paraformer_context * ctx,
        const char * text,
        paraformer_token * tokens,
        int   n_max_tokens);

// Largest language id (i.e. number of available languages - 1)
PARAFORMER_API int paraformer_lang_max_id();

// Return the id of the specified language, returns -1 if not found
// Examples:
//   "de" -> 2
//   "german" -> 2
PARAFORMER_API int paraformer_lang_id(const char * lang);

// Return the short string of the specified language id (e.g. 2 -> "de"), returns nullptr if not found
PARAFORMER_API const char * paraformer_lang_str(int id);

// Use mel data at offset_ms to try and auto-detect the spoken language
// Make sure to call paraformer_pcm_to_mel() or paraformer_set_mel() first
// Returns the top language id or negative on failure
// If not null, fills the lang_probs array with the probabilities of all languages
// The array must be paraformer_lang_max_id() + 1 in size
// ref: https://github.com/openai/paraformer/blob/main/paraformer/decoding.py#L18-L69
PARAFORMER_API int paraformer_lang_auto_detect(
        struct paraformer_context * ctx,
        int   offset_ms,
        int   n_threads,
        float * lang_probs);

PARAFORMER_API int paraformer_lang_auto_detect_with_state(
        struct paraformer_context * ctx,
        struct paraformer_state * state,
        int   offset_ms,
        int   n_threads,
        float * lang_probs);

PARAFORMER_API int paraformer_n_len           (struct paraformer_context * ctx); // mel length
PARAFORMER_API int paraformer_n_len_from_state(struct paraformer_state * state); // mel length
PARAFORMER_API int paraformer_n_vocab         (struct paraformer_context * ctx);
PARAFORMER_API int paraformer_n_text_ctx      (struct paraformer_context * ctx);
PARAFORMER_API int paraformer_n_audio_ctx     (struct paraformer_context * ctx);
PARAFORMER_API int paraformer_is_multilingual (struct paraformer_context * ctx);

PARAFORMER_API int paraformer_model_n_vocab      (struct paraformer_context * ctx);
PARAFORMER_API int paraformer_model_n_audio_ctx  (struct paraformer_context * ctx);
PARAFORMER_API int paraformer_model_n_audio_state(struct paraformer_context * ctx);
PARAFORMER_API int paraformer_model_n_audio_head (struct paraformer_context * ctx);
PARAFORMER_API int paraformer_model_n_audio_layer(struct paraformer_context * ctx);
PARAFORMER_API int paraformer_model_n_text_ctx   (struct paraformer_context * ctx);
PARAFORMER_API int paraformer_model_n_text_state (struct paraformer_context * ctx);
PARAFORMER_API int paraformer_model_n_text_head  (struct paraformer_context * ctx);
PARAFORMER_API int paraformer_model_n_text_layer (struct paraformer_context * ctx);
PARAFORMER_API int paraformer_model_n_mels       (struct paraformer_context * ctx);
PARAFORMER_API int paraformer_model_ftype        (struct paraformer_context * ctx);
PARAFORMER_API int paraformer_model_type         (struct paraformer_context * ctx);

// Token logits obtained from the last call to paraformer_decode()
// The logits for the last token are stored in the last row
// Rows: n_tokens
// Cols: n_vocab
PARAFORMER_API float * paraformer_get_logits           (struct paraformer_context * ctx);
PARAFORMER_API float * paraformer_get_logits_from_state(struct paraformer_state * state);

// Token Id -> String. Uses the vocabulary in the provided context
PARAFORMER_API const char * paraformer_token_to_str(struct paraformer_context * ctx, paraformer_token token);
PARAFORMER_API const char * paraformer_model_type_readable(struct paraformer_context * ctx);


// Special tokens
PARAFORMER_API paraformer_token paraformer_token_eot (struct paraformer_context * ctx);
PARAFORMER_API paraformer_token paraformer_token_sot (struct paraformer_context * ctx);
PARAFORMER_API paraformer_token paraformer_token_solm(struct paraformer_context * ctx);
PARAFORMER_API paraformer_token paraformer_token_prev(struct paraformer_context * ctx);
PARAFORMER_API paraformer_token paraformer_token_nosp(struct paraformer_context * ctx);
PARAFORMER_API paraformer_token paraformer_token_not (struct paraformer_context * ctx);
PARAFORMER_API paraformer_token paraformer_token_beg (struct paraformer_context * ctx);
PARAFORMER_API paraformer_token paraformer_token_lang(struct paraformer_context * ctx, int lang_id);

// Task tokens
PARAFORMER_API paraformer_token paraformer_token_translate (struct paraformer_context * ctx);
PARAFORMER_API paraformer_token paraformer_token_transcribe(struct paraformer_context * ctx);

// Performance information from the default state.
PARAFORMER_API void paraformer_print_timings(struct paraformer_context * ctx);
PARAFORMER_API void paraformer_reset_timings(struct paraformer_context * ctx);

// Print system information
PARAFORMER_API const char * paraformer_print_system_info(void);

////////////////////////////////////////////////////////////////////////////

// Available sampling strategies
enum paraformer_sampling_strategy {
    PARAFORMER_SAMPLING_GREEDY,      // similar to OpenAI's GreedyDecoder
    PARAFORMER_SAMPLING_BEAM_SEARCH, // similar to OpenAI's BeamSearchDecoder
};

// Text segment callback
// Called on every newly generated text segment
// Use the paraformer_full_...() functions to obtain the text segments
typedef void (*paraformer_new_segment_callback)(struct paraformer_context * ctx, struct paraformer_state * state, int n_new, void * user_data);

// Progress callback
typedef void (*paraformer_progress_callback)(struct paraformer_context * ctx, struct paraformer_state * state, int progress, void * user_data);

// Encoder begin callback
// If not NULL, called before the encoder starts
// If it returns false, the computation is aborted
typedef bool (*paraformer_encoder_begin_callback)(struct paraformer_context * ctx, struct paraformer_state * state, void * user_data);

// Logits filter callback
// Can be used to modify the logits before sampling
// If not NULL, called after applying temperature to logits
typedef void (*paraformer_logits_filter_callback)(
        struct paraformer_context * ctx,
        struct paraformer_state * state,
        const paraformer_token_data * tokens,
        int   n_tokens,
        float * logits,
        void * user_data);

// Parameters for the paraformer_full() function
// If you change the order or add new parameters, make sure to update the default values in paraformer.cpp:
// paraformer_full_default_params()
struct paraformer_full_params {
    enum paraformer_sampling_strategy strategy;

    int n_threads;
    int n_max_text_ctx;     // max tokens to use from past text as prompt for the decoder
    int offset_ms;          // start offset in ms
    int duration_ms;        // audio duration to process in ms

    bool translate;
    bool no_context;        // do not use past transcription (if any) as initial prompt for the decoder
    bool single_segment;    // force single segment output (useful for streaming)
    bool print_special;     // print special tokens (e.g. <SOT>, <EOT>, <BEG>, etc.)
    bool print_progress;    // print progress information
    bool print_realtime;    // print results from within paraformer.cpp (avoid it, use callback instead)
    bool print_timestamps;  // print timestamps for each text segment when printing realtime

    // [EXPERIMENTAL] token-level timestamps
    bool  token_timestamps; // enable token-level timestamps
    float thold_pt;         // timestamp token probability threshold (~0.01)
    float thold_ptsum;      // timestamp token sum probability threshold (~0.01)
    int   max_len;          // max segment length in characters
    bool  split_on_word;    // split on word rather than on token (when used with max_len)
    int   max_tokens;       // max tokens per segment (0 = no limit)

    // [EXPERIMENTAL] speed-up techniques
    // note: these can significantly reduce the quality of the output
    bool speed_up;          // speed-up the audio by 2x using Phase Vocoder
    bool debug_mode;        // enable debug_mode provides extra info (eg. Dump log_mel)
    int  audio_ctx;         // overwrite the audio context size (0 = use default)

    // [EXPERIMENTAL] [TDRZ] tinydiarize
    bool tdrz_enable;       // enable tinydiarize speaker turn detection

    // tokens to provide to the paraformer decoder as initial prompt
    // these are prepended to any existing text context from a previous call
    const char * initial_prompt;
    const paraformer_token * prompt_tokens;
    int prompt_n_tokens;

    // for auto-detection, set to nullptr, "" or "auto"
    const char * language;
    bool detect_language;

    // common decoding parameters:
    bool suppress_blank;    // ref: https://github.com/openai/paraformer/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/paraformer/decoding.py#L89
    bool suppress_non_speech_tokens; // ref: https://github.com/openai/paraformer/blob/7858aa9c08d98f75575035ecd6481f462d66ca27/paraformer/tokenizer.py#L224-L253

    float temperature;      // initial decoding temperature, ref: https://ai.stackexchange.com/a/32478
    float max_initial_ts;   // ref: https://github.com/openai/paraformer/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/paraformer/decoding.py#L97
    float length_penalty;   // ref: https://github.com/openai/paraformer/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/paraformer/transcribe.py#L267

    // fallback parameters
    // ref: https://github.com/openai/paraformer/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/paraformer/transcribe.py#L274-L278
    float temperature_inc;
    float entropy_thold;    // similar to OpenAI's "compression_ratio_threshold"
    float logprob_thold;
    float no_speech_thold;  // TODO: not implemented

    struct {
        int best_of;    // ref: https://github.com/openai/paraformer/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/paraformer/transcribe.py#L264
    } greedy;

    struct {
        int beam_size;  // ref: https://github.com/openai/paraformer/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/paraformer/transcribe.py#L265

        float patience; // TODO: not implemented, ref: https://arxiv.org/pdf/2204.05424.pdf
    } beam_search;

    // called for every newly generated text segment
    paraformer_new_segment_callback new_segment_callback;
    void * new_segment_callback_user_data;

    // called on each progress update
    paraformer_progress_callback progress_callback;
    void * progress_callback_user_data;

    // called each time before the encoder starts
    paraformer_encoder_begin_callback encoder_begin_callback;
    void * encoder_begin_callback_user_data;

    // called by each decoder to filter obtained logits
    paraformer_logits_filter_callback logits_filter_callback;
    void * logits_filter_callback_user_data;
};

// NOTE: this function allocates memory, and it is the responsibility of the caller to free the pointer - see paraformer_free_params()
PARAFORMER_API struct paraformer_full_params * paraformer_full_default_params_by_ref(enum paraformer_sampling_strategy strategy);
PARAFORMER_API struct paraformer_full_params paraformer_full_default_params(enum paraformer_sampling_strategy strategy);

// Run the entire model: PCM -> log mel spectrogram -> encoder -> decoder -> text
// Not thread safe for same context
// Uses the specified decoding strategy to obtain the text.
PARAFORMER_API int paraformer_full(
        struct paraformer_context * ctx,
        struct paraformer_full_params   params,
        const float * samples,
        int   n_samples);

PARAFORMER_API int paraformer_full_with_state(
        struct paraformer_context * ctx,
        struct paraformer_state * state,
        struct paraformer_full_params   params,
        const float * samples,
        int   n_samples);

// Split the input audio in chunks and process each chunk separately using paraformer_full_with_state()
// Result is stored in the default state of the context
// Not thread safe if executed in parallel on the same context.
// It seems this approach can offer some speedup in some cases.
// However, the transcription accuracy can be worse at the beginning and end of each chunk.
PARAFORMER_API int paraformer_full_parallel(
        struct paraformer_context * ctx,
        struct paraformer_full_params   params,
        const float * samples,
        int   n_samples,
        int   n_processors);

// Number of generated text segments
// A segment can be a few words, a sentence, or even a paragraph.
PARAFORMER_API int paraformer_full_n_segments           (struct paraformer_context * ctx);
PARAFORMER_API int paraformer_full_n_segments_from_state(struct paraformer_state * state);

// Language id associated with the context's default state
PARAFORMER_API int paraformer_full_lang_id(struct paraformer_context * ctx);

// Language id associated with the provided state
PARAFORMER_API int paraformer_full_lang_id_from_state(struct paraformer_state * state);

// Get the start and end time of the specified segment
PARAFORMER_API int64_t paraformer_full_get_segment_t0           (struct paraformer_context * ctx, int i_segment);
PARAFORMER_API int64_t paraformer_full_get_segment_t0_from_state(struct paraformer_state * state, int i_segment);

PARAFORMER_API int64_t paraformer_full_get_segment_t1           (struct paraformer_context * ctx, int i_segment);
PARAFORMER_API int64_t paraformer_full_get_segment_t1_from_state(struct paraformer_state * state, int i_segment);

// Get whether the next segment is predicted as a speaker turn
PARAFORMER_API bool paraformer_full_get_segment_speaker_turn_next(struct paraformer_context * ctx, int i_segment);

// Get the text of the specified segment
PARAFORMER_API const char * paraformer_full_get_segment_text           (struct paraformer_context * ctx, int i_segment);
PARAFORMER_API const char * paraformer_full_get_segment_text_from_state(struct paraformer_state * state, int i_segment);

// Get number of tokens in the specified segment
PARAFORMER_API int paraformer_full_n_tokens           (struct paraformer_context * ctx, int i_segment);
PARAFORMER_API int paraformer_full_n_tokens_from_state(struct paraformer_state * state, int i_segment);

// Get the token text of the specified token in the specified segment
PARAFORMER_API const char * paraformer_full_get_token_text           (struct paraformer_context * ctx, int i_segment, int i_token);
PARAFORMER_API const char * paraformer_full_get_token_text_from_state(struct paraformer_context * ctx, struct paraformer_state * state, int i_segment, int i_token);

PARAFORMER_API paraformer_token paraformer_full_get_token_id           (struct paraformer_context * ctx, int i_segment, int i_token);
PARAFORMER_API paraformer_token paraformer_full_get_token_id_from_state(struct paraformer_state * state, int i_segment, int i_token);

// Get token data for the specified token in the specified segment
// This contains probabilities, timestamps, etc.
PARAFORMER_API paraformer_token_data paraformer_full_get_token_data           (struct paraformer_context * ctx, int i_segment, int i_token);
PARAFORMER_API paraformer_token_data paraformer_full_get_token_data_from_state(struct paraformer_state * state, int i_segment, int i_token);

// Get the probability of the specified token in the specified segment
PARAFORMER_API float paraformer_full_get_token_p           (struct paraformer_context * ctx, int i_segment, int i_token);
PARAFORMER_API float paraformer_full_get_token_p_from_state(struct paraformer_state * state, int i_segment, int i_token);

////////////////////////////////////////////////////////////////////////////

// Temporary helpers needed for exposing ggml interface

PARAFORMER_API int          paraformer_bench_memcpy          (int n_threads);
PARAFORMER_API const char * paraformer_bench_memcpy_str      (int n_threads);
PARAFORMER_API int          paraformer_bench_ggml_mul_mat    (int n_threads);
PARAFORMER_API const char * paraformer_bench_ggml_mul_mat_str(int n_threads);

// Control logging output; default behavior is to print to stderr

typedef void (*paraformer_log_callback)(const char * line);
PARAFORMER_API void paraformer_set_log_callback(paraformer_log_callback callback);

#ifdef __cplusplus
}
#endif

#endif