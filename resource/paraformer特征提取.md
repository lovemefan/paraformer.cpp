## 特征提取

代码参考[paraformer-frontend.cpp](https://github.com/lovemefan/paraformer.cpp/blob/main/src/csrc/paraformer-frontend.cpp)

```
waveform(16000) -> fbank(98, 80) -> pad fbank (128, 80) -> lfr (22, 560) -> cvmn (22, 560) 
```

参考 [kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank)实现

kaldi-native-fbank 参数

```yaml
frontend_conf:
  frame_rate: 16000
  window: hamming
  n_mels: 80
  frame_length: 25
  frame_shift: 10
  lfr_m: 7
  lfr_n: 6
  dither: 1.0
  preemph_coeff: 0.97
  remove_dc_offset: true
  round_to_power_of_two: true
  blackman_coeff: 0.42
  energy_floor: 0

  snip_edges: true
  use_energy: false
  raw_energy: true
  htk_compat: false
  use_log_fbank: true
  use_power: true

```

### 1. fbank

### 1.1 直流偏移消除

**定义：** 信号的直流分量指的是信号的平均值，信号中的直流部分，就称为直流偏移（DC offset）。
** 目的：** 如果一个信号带有直流分量，它就会呈现出“不对称”的特性，会减少我们处理音频时可以使用的动态范围。

对于直流偏移的信号，我们只要把它整体减去它的偏移量

```c++
// window 
std::vector<double> window(512);
float sum = 0;
for (int32_t i = 0; i != frame_size; ++i) {
    sum += window[i];
}
float mean = sum / frame_size;
for (int32_t i = 0; i != frame_size; ++i) {
    window[i] -= mean;
}
```

#### 1.2 预加重

**定义：** 预加重即对语音的高频部分进行加重。

**预加重的目的：**

* 平衡频谱，因为高频通常与较低频率相比具有较小的幅度，提升高频部分，使信号的频谱变得平坦，保持在低频到高频的整个频带中，能用同样的噪声比（SNR）求频谱。
* 也是为了消除发生过程中声带和嘴唇的效应，来补偿语音信号受到发音系统所抑制的高频部分，也为了突出高频的共振峰。

预加重处理其实是将语音信号通过一个高通滤波器：

$$
y(t) = x(t) - \alpha * x(t-1)
$$

其中$\alpha$ 为 0.97

```c++
#define PREEMPH_COEFF 0.97

for (int32_t i = frame_size - 1; i > 0; --i) {
    window[i] -= PREEMPH_COEFF * window[i - 1];
}
window[0] -= PREEMPH_COEFF * window[0];
```

#### 1.3 分帧

目的：

由于语音信号是一个非平稳态过程，不能用处理平稳信号的信号处理技术对其进行分析处理。语音信号是短时平稳信号。因此我们在短时帧上进行傅里叶变换，通过连接相邻帧来获得信号频率轮廓的良好近似。

过程：

为了方便对语音分析，可以将语音分成一个个小段，称之为：帧。先将N个采样点集合成一个观测单位，称为帧。通常情况下N的值为256或512，涵盖的时间约为20~
30ms左右。为了避免相邻两帧的变化过大，因此会让两相邻帧之间有一段重叠区域，此重叠区域包含了M个取样点，通常M的值约为N的1/2或1/3。通常语音识别所采用语音信号的采样频率为8KHz或16KHz，以8KHz来说，若帧长度为256个采样点，则对应的时间长度是256/8000×1000=32ms。简单表述就两点：

短时分析将语音流分为一帧来处理，帧长：10～30ms，20ms常见；
帧移：STRIDE，0～1/2帧长，帧与帧之间的平滑长度；

#### 1.3 加窗

在分帧之后，通常需要对每帧的信号进行加窗处理。目的是让帧两端平滑地衰减，这样可以降低后续傅里叶变换后旁瓣的强度，
取得更高质量的频谱。常用的窗有：矩形窗、汉明（Hamming）窗、汉宁窗（Hanning），以汉明窗为例，其窗函数为：

$$
w(n) = 0.54 - 0.46 * cos (\frac{2\pi n}{N - 1})
$$
这里 0 < n <= N-1， N是窗的宽度。

```c++
#define M_2PI 6.283185307179586476925286766559005
static bool hamming_window(int length, bool periodic, std::vector<double> &output) {
    if (output.size() < static_cast<size_t>(length)) {
        output.resize(length);
    }
    int offset = -1;
    if (periodic) {
        offset = 0;
    }
    for (int i = 0; i < length; i++) {
        output[i] = 0.54 - 0.46 * cosf((M_2PI * i) / (length + offset));
    }

    return true;
}

std::vector<double> hamming;
hamming_window(frame_size * n_frames_per_ms, true, hamming);
for (int j = 0; j < frame_size; j++) {
    window[j] *= hamming[j];
 }
```

#### 1.4 傅立叶变换

对于每一帧的加窗信号，进行N点RFFT变换，也称短时傅里叶变换（STFT），N这里取512。
本项目使用的是第三方库[General Purpose FFT (Fast Fourier/Cosine/Sine Transform) Package](https://www.kurims.kyoto-u.ac.jp/~ooura/fft.html)
代码在本项目中的src/csrc/fftsg.cpp

### 2. lfr

低帧率（Low Frame Rate，LFR）技术是阿里开源语音识别引擎DFSMN中提出的一种技术，该技术
通过合并多帧语音帧，可以获得2-3倍的训练以及解码的加速，可以显著的减少我们的系统实际应用时所需要的计算资源。

主要思想类似于滑动窗口，窗口大小为7帧，滑动距离为6帧。
将7帧80维的mel向量合并为560维的特征向量。

```c++
int T = mel.n_len;
int lfr_m = mel.lfr_m;  // 7
int lfr_n = mel.lfr_n;  // 6
int T_lrf = ceil(1.0 * T / mel.lfr_n);
int left_pad = (mel.lfr_m - 1) / 2;
int left_pad_offset = (lfr_m - left_pad) * mel.n_mel;
// Merge lfr_m frames as one,lfr_n frames per window
T = T + (lfr_m - 1) / 2;
std::vector<double> p;
for (int i = 0; i < T_lrf; i++) {
    // the first frames need left padding
    if (i == 0) {
        // left padding
        for (int j = 0; j < left_pad; j++) {
            p.insert(p.end(), mel.data.begin(), mel.data.begin() + mel.n_mel);
        }
        p.insert(p.end(), mel.data.begin(), mel.data.begin() + left_pad_offset);
        out_feats.emplace_back(p);
        p.clear();
    } else {
        if (lfr_m <= T - i * lfr_n) {
            p.insert(p.end(), mel.data.begin() + (i * lfr_n - left_pad) * mel.n_mel,
                     mel.data.begin() + (i * lfr_n - left_pad + lfr_m) * mel.n_mel);
            out_feats.emplace_back(p);
            p.clear();
        } else {
            // Fill to lfr_m frames at last window if less than lfr_m frames  (copy last frame)
            int num_padding = lfr_m - (T - i * lfr_n);
            for (int j = 0; j < (mel.n_len - i * lfr_n); j++) {
                p.insert(p.end(), mel.data.begin() + (i * lfr_n - left_pad) * mel.n_mel, mel.data.end());
            }
            for (int j = 0; j < num_padding; j++) {
                p.insert(p.end(), mel.data.end() - mel.n_mel, mel.data.end());
            }
            out_feats.emplace_back(p);
            p.clear();
        }
    }
}
```

### 3. cvmn

cmvn：倒谱均值方差归一化

提取声学特征以后，将声学特征从一个空间转变成另一个空间，使得在这个空间下更特征参数更符合某种概率分布，压缩了特征参数值域的动态范围，减少了训练和测试环境的不匹配等
提升模型的鲁棒性，其实就是归一化的操作。

本项目直接使用funasr提供的resource/model/am.mvn文件，其中包含数组大小为560的浮点数均值数组，
数组大小为560为的浮点数方差数组

```c++
// apply cvmn
for (auto &out_feat : out_feats) {
    for (int j = 0; j < cmvn.cmvn_means.size(); j++) {
        out_feat[j] = (out_feat[j] + cmvn.cmvn_means[j]) * cmvn.cmvn_vars[j];
    }
}
```
