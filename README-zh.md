(简体中文|[English](./README.md))

## 开发中...

paraformer是阿里funasr开源的中文语音识别模型。 本项目基于ggml推理框架，提供一个c++的runtime

- [x] 定义模型结构
- [x] 初始化模型
- [x] 加载模型参数和词表
- [x] fbank + lfr + cvmn 特征提取
- [ ] 构建计算图
    - [x] encoder
    - [ ] predictor
    - [ ] decoder
    - [ ] bias encoder
    - [ ] seaco decoder

## 特性

1. 基于ggml，不依赖其他第三方库, 致力于端侧部署
2. 特征提取参考[kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank)库，并使用多线程加速特征提取过程。
3. 参看[whisper.cpp](https://github.com/ggerganov/ggml/blob/master/examples/whisper/whisper.cpp)项目，使用flash attention
   解码

## 使用

下载模型并转换模型格式

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

## 感谢以下项目

1. 本项目借用并模仿来自[whisper.cpp](https://github.com/ggerganov/ggml/blob/master/examples/whisper/whisper.cpp)
   的大部分c++代码
2. 参考来自funasr的paraformer模型结构以及前向计算 [FunASR](https://github.com/alibaba-damo-academy/FunASR)
3. 本项目参考并借用 [kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank)中的fbank特征提取算法。
   [FunASR](https://github.com/alibaba-damo-academy/FunASR/blob/main/runtime/onnxruntime/src/paraformer.cpp#L337C22-L372)
   中的lrf + cmvn 算法 

