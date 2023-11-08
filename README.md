## 开发中...

- [x] 定义模型结构
- [x] 初始化模型
- [ ] 加载模型参数和词表
- [ ] 构建计算图

## 使用

下载模型并转换模型格式

```bash
git clone https://www.modelscope.cn/damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404.git resource/model
python src/python/convert-pt-to-ggml.py -i resource/model/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404 -o resource/model --fp16
```