(简体中文|[English](./README.md))

## 开发中...

paraformer是阿里funasr开源的中文语音识别模型。 本项目基于ggml推理框架，提供一个c++的runtime

- [x] 定义模型结构
- [x] 初始化模型
- [x] 加载模型参数和词表
- [ ] 构建计算图

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