## 特征提取

```
waveform(16000, )->fbank(98, 80) -> lfr -> cvmn 
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

#### 1.1 分帧

#### 1.2 预加重

#### 1.3 加窗

#### 1.4 傅立叶变换

### 2. lfr

### 3. cvmn