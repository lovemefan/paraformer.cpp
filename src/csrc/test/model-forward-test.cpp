//
// Created by lovemefan on 2023/11/5.
//

#include <paraformer-frontend.h>

#include <fstream>

#include "paraformer-offline.cpp"
#include "paraformer-offline.h"

void load_feature_from_wav_file(std::string &wav_file,
                                paraformer_feature &mel) {
  std::vector<double> samples;
  int sample_rate;
  load_wav_file(wav_file.c_str(), &sample_rate, samples);
  int n_samples = samples.size();
  int frame_size = 25;
  int frame_step = 10;
  int n_mel = 80;
  int n_thread = 4;
  std::string cmvn_file =
      "/Users/cenglingfan/Code/cpp-project/paraformer.cpp/resource/model/"
      "am.mvn";
  struct paraformer_cmvn cmvn;
  load_cmvn(cmvn_file.c_str(), cmvn);
  auto start = std::chrono::steady_clock::now();
  fbank_lfr_cmvn_feature(samples, n_samples, frame_size, frame_step, n_mel,
                         n_thread, true, cmvn, mel);
}

int main() {
  std::string path_model =
      "/Users/cenglingfan/Code/cpp-project/paraformer.cpp/resource/model/"
      "seaco-paraformer-ggml-model-fp16.bin";

  printf("%s: loading model from '%s'\n", __func__, path_model.c_str());

  struct paraformer_context_params cparams = {
      /*.use_gpu              =*/false,
      /*.gpu_device           =*/0,
  };
  struct paraformer_context *context =
      paraformer_init_from_file_with_params(path_model.c_str(), cparams);

  if (context == nullptr) {
    fprintf(stderr, "error: failed to initialize paraformer context\n");
    return 3;
  }
  std::string wav_file =
      "/Users/cenglingfan/Code/cpp-project/paraformer.cpp/resource/model/"
      "test.wav";

  load_feature_from_wav_file(wav_file, context->state->feature);

  {
    paraformer_full_params params =
        paraformer_full_default_params(PARAFORMER_SAMPLING_GREEDY);
    paraformer_full_with_state(context, context->state, params,
                               context->state->feature, 4);
  }

  //    ggml_cgraph *gf = paraformer_build_graph_encoder(*context,
  //    *context->state); gf->nodes;
}