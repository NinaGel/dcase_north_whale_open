# Dataset Synthesis Guide / 数据集合成指南

This guide explains how to synthesize datasets for the Whale and DCASE tracks.

本指南介绍如何为 Whale 和 DCASE 合成数据集。

## Overview / 概述

Both tracks use [Scaper](https://github.com/justinsalamon/scaper) for polyphonic sound event synthesis with SNR control.

两个轨道都使用 [Scaper](https://github.com/justinsalamon/scaper) 进行带SNR控制的复音声音事件合成。

## Prerequisites / 前置要求

```bash
pip install scaper
```

---

## Whale Track / 鲸鱼轨道

### Source Data / 源数据

- **DCLDE 2024 (Detection, Classification, Localization, and Density Estimation)**
- North Atlantic Right Whale (NARW) 北大西洋露脊鲸录音
- Reference / 参考: [DCLDE 2024 Data Sets](https://www.dclde2024.com/data-sets/)

### Download Dataset / 下载数据集

数据集下载地址 / Dataset download:
- 链接: https://sites.drdc-rddc.gc.ca/proj/OS_NARW/
- 用户名 / Username: `DCLDE24C@gmail.com`
- 密码 / Password: `Conference24`

详细说明参见 `dclde2024_dataset_v5_0-1.pdf` 。

### Data Preparation / 数据准备

从 DCLDE 数据集中提取前景声和背景声：
- **Foreground / 前景声**: 鲸鱼叫声片段（upcall, gunshot, scream, moancall）
- **Background / 背景声**: 海洋环境噪声

提取完成后，使用 Scaper 进行合成。

### Synthesis Script / 合成脚本

Location / 位置: `Data/dclde_synthetic/soundscaper.py`

提取的前景声和背景声通过 soundscaper 脚本进行多事件合成，支持自定义 SNR 范围。

### SNR Groups / SNR分组

| Group / 分组 | SNR Range / SNR范围 | Description / 描述 |
|--------------|---------------------|---------------------|
| high | 15-25 dB | Clean signals / 干净信号 |
| medium | 5-15 dB | Moderate noise / 中等噪声 |
| low | -5-5 dB | Challenging / 有挑战性 |
| very_low | -15--5 dB | Very challenging / 非常有挑战性 |

### Usage / 使用方法

```python
from Data.dclde_synthetic.soundscaper import WhaleScapeSynthesizer

synthesizer = WhaleScapeSynthesizer(
    foreground_path='path/to/whale_calls',
    background_path='path/to/background_noise',
    output_path='path/to/output'
)

# Generate dataset with specific SNR / 生成特定SNR的数据集
synthesizer.generate(
    num_files=1000,
    snr_range=(5, 15),  # medium SNR
    duration=10.0  # 10 seconds per file
)
```

### Output Structure / 输出结构

```
Data/dclde_synthetic/
├── high_snr/
│   ├── audio/
│   └── labels/
├── medium_snr/
├── low_snr/
└── very_low_snr/
```

---

## DCASE Track / DCASE轨道

### Source Data / 源数据

- **DESED (Domestic Environment Sound Event Detection) SoundBank**
- Reference / 参考: [DESED GitHub](https://github.com/turpaultn/DESED)

### Download SoundBank / 下载SoundBank

**方法1: 使用desed库自动下载**

```bash
pip install desed
```

```python
import desed
desed.download_desed_soundbank("./data/soundbank")
```

**方法2: 手动下载**

如果自动下载失败，从 Zenodo 手动下载：
- 链接: https://zenodo.org/records/6026841
- 文件: `DESED_synth_soundbank.tar.gz`

```bash
wget https://zenodo.org/records/6026841/files/DESED_synth_soundbank.tar.gz
tar -xzf DESED_synth_soundbank.tar.gz -C ./data/soundbank
```

### SoundBank 组成

| Split | Background | Foreground |
|-------|------------|------------|
| Training | 2060 files (SINS) | 1009 files (Freesound) |
| Eval | 12 (Freesound) + 5 (Youtube) | 314 files (Freesound) |

- **Background**: 背景音频，用于混合
- **Foreground**: 前景事件音频，10类声音事件

### Sound Classes / 声音类别

1. Alarm_bell_ringing - 警铃响
2. Blender - 搅拌机
3. Cat - 猫
4. Dishes - 餐具
5. Dog - 狗
6. Electric_shaver_toothbrush - 电动剃须刀/牙刷
7. Frying - 油炸
8. Running_water - 流水
9. Speech - 语音
10. Vacuum_cleaner - 吸尘器

### Synthesis Script / 合成脚本

Location / 位置: `Data/dcase_synthetic/generate_snr_grouped_dcase_scaper.py`

### DESED Standard Configuration / DESED标准配置

- Sample Rate / 采样率: 16 kHz
- SNR Range / SNR范围: 6-30 dB (DESED standard)
- Duration / 时长: 10 seconds per clip / 每个片段10秒

### Usage / 使用方法

```python
from Data.dcase_synthetic.generate_snr_grouped_dcase_scaper import DCASEScapeSynthesizer

synthesizer = DCASEScapeSynthesizer(
    soundbank_path='path/to/DCASE2021_soundbank',
    output_path='path/to/output'
)

# Generate DESED-standard dataset / 生成DESED标准数据集
synthesizer.generate(
    num_files=10000,
    snr_range=(6, 30),
    sample_rate=16000
)
```

### Output Structure / 输出结构

```
Data/dcase_synthetic_10k/
└── dcase_snr_desed_standard/
    ├── train/
    │   ├── audio/
    │   └── labels/
    ├── val/
    └── test/
```

---

## Label Format / 标签格式

Both tracks use TSV format / 两个轨道都使用TSV格式:

```
onset	offset	event_label
0.5	2.3	Cat
1.2	3.8	Dog
2.0	4.5	Speech
```

- `onset`: Start time in seconds / 开始时间（秒）
- `offset`: End time in seconds / 结束时间（秒）
- `event_label`: Sound event class / 声音事件类别

---

## Tips / 提示

1. **Balance / 平衡**: Ensure balanced class distribution
   确保类别分布均衡

2. **Validation / 验证**: Check synthesized audio quality manually
   手动检查合成音频质量

3. **Storage / 存储**: Expect ~50GB for 10k DCASE samples
   10k个DCASE样本预计需要约50GB存储空间

---

## References / 参考文献

- [Scaper Documentation](https://scaper.readthedocs.io/)
- [DCASE Challenge](https://dcase.community/)
- [DESED Dataset & SoundBank](https://github.com/turpaultn/DESED)
- [DESED SoundBank Download (Zenodo)](https://zenodo.org/records/6026841)
- [DCLDE 2024 Workshop](https://www.dclde2024.com/)
- [DCLDE 2024 Data Sets](https://www.dclde2024.com/data-sets/)
