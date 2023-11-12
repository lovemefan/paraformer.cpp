([简体中文](./README-zh.md)|English)

## developing ...

paraformer is a chinese asr from funasr. this project provided a port of paraformer model
with [ggml](https://github.com/ggerganov/ggml/)

- [x] Define model structure
- [x] Initialize model
- [x] Load model parameters and vocabulary
- [ ] Build calculation forward graph

## usage

you need download model from modelscope and convert weight with script

```bash
git clone https://github.com/lovemefan/paraformer.cpp
cd git paraformer.cpp
git submodule sync && git submodule update --init --recursive
# download model weight from modelscope
git clone https://www.modelscope.cn/damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404.git resource/model
# convert weight and vocab into ggml format
python src/python/convert-pt-to-ggml.py -i resource/model/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404 -o resource/model --fp16
```

## Acknowledge

1. we borrowed and imitated most code
   from [whisper.cpp](https://github.com/ggerganov/ggml/blob/master/examples/whisper/whisper.cpp)

