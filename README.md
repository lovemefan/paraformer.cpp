([简体中文](./README-zh.md)|English)

## developing ...

paraformer is a chinese asr from funasr. this project provided a port of paraformer model
with [ggml](https://github.com/ggerganov/ggml/)

- [x] Define model structure
- [x] Initialize model
- [x] Load model parameters and vocabulary
- [x] fbank + lfr + cmvn feature extraction
- [ ] Build calculation forward graph
    - [x] encoder
    - [ ] predictor
    - [ ] decoder
    - [ ] bias encoder
    - [ ] seaco decoder

## usage

you need download model from modelscope and convert weight with script

```bash
git clone https://github.com/lovemefan/paraformer.cpp
cd git paraformer.cpp
git submodule sync && git submodule update --init --recursive

mkdir build && cd build
cmake ../src/csrc && make -j 8

# download model weight from modelscope
git clone https://www.modelscope.cn/damo/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch.git resource/model
# convert weight and vocab into gguf format
python src/python/convert-pt-to-gguf.py \
  --model /Users/cenglingfan/Code/cpp-project/paraformer.cpp/resource/model/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch \
  --outfile /Users/cenglingfan/Code/cpp-project/paraformer.cpp/resource/model/seaco-paraformer-ggml-model-fp16.bin
```

## Acknowledge

1. we borrowed and imitated most c++ code
   from [whisper.cpp](https://github.com/ggerganov/ggml/blob/master/examples/whisper/whisper.cpp)

2. we referenced model structure and forward detail from [FunASR](https://github.com/alibaba-damo-academy/FunASR)

3. we borrowed fbank feature extract algorithm code
   from [kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank)
   and lrf + cmvn algorithm code
   from[funasr](https://github.com/alibaba-damo-academy/FunASR/blob/main/runtime/onnxruntime/src/paraformer.cpp#L337C22-L372)