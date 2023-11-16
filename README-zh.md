(简体中文|[English](./README.md))

## 开发中...

paraformer是阿里funasr开源的中文语音识别模型。 本项目基于ggml推理框架，提供一个c++的runtime

- [x] 定义模型结构
- [x] 初始化模型
- [x] 加载模型参数和词表
- [ ] 构建计算图
    - [x] encoder
    - [ ] predictor
    - [ ] decoder
    - [ ] bias encoder

## 使用

下载模型并转换模型格式

```bash
git clone https://github.com/lovemefan/paraformer.cpp
cd git paraformer.cpp
git submodule sync && git submodule update --init --recursive
# download model weight from modelscope
git clone https://www.modelscope.cn/damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404.git resource/model
# convert weight and vocab into ggml format
python src/python/convert-pt-to-ggml.py -i resource/model/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404 -o resource/model --fp16
```

## 感谢以下项目

1. 本项目借用并模仿来自[whisper.cpp](https://github.com/ggerganov/ggml/blob/master/examples/whisper/whisper.cpp)
   的大部分c++代码
2. 参考来自funasr的paraformer模型结构以及前向计算 [FunASR](https://github.com/alibaba-damo-academy/FunASR)
   

