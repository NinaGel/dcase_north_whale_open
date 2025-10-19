# Sound Event Detection: Whale & DCASE Challenge

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-red)](https://pytorch.org/)

Advanced deep learning models for overlapping sound event detection with focus on whale acoustics (underwater) and DCASE2021 domestic sound events.

**Paper**: *Dynamic Attention-Asymmetric Perceptron Network for Overlapping Sound Event Detection*
- Key innovation: DAAPNet with BA-Conv, TFDP, and DDSA modules

## 📋 目录

- [快速开始](#-快速开始)
- [项目结构](#-项目结构)
- [数据集合成](#-dataset-synthesis)
- [评估](#-evaluation)
- [实验结果](#-expected-results)
- [模型](#-model-variants)
- [配置](#-configuration)
- [许可证](#-许可证)

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/NinaGel/dcase_north_whale_open.git
cd dcase_north_whale_open
pip install -r requirements.txt
```

详细依赖说明请参阅 [requirements.txt](requirements.txt)。

### Whale轨道 - 运行注意力对比实验

```bash
# 主要注意力机制对比
python experiments/attention_comparison.py --epochs 50 --batch_size 32

# 其他实验
python experiments/model_comparison.py            # 模型对比
python experiments/conv_module_comparison.py      # 卷积模块消融
python experiments/dynamic_conv_comparison.py     # 动态卷积研究
```

### DCASE轨道 - 运行多种子实验

```bash
# 标准多种子实验（推荐）
python run_multi_seed_experiments.py --model conformer --seeds 42 123 456 --epochs 60

# 其他支持的模型
python run_multi_seed_experiments.py --model daapnet --seeds 42 123 456 --epochs 80
python run_multi_seed_experiments.py --model faf_heavy --seeds 42 123 456 --epochs 60
```

更多实验细节请参阅 `experiments/` 目录下的实验脚本。

## 📊 项目结构

```
dcase_north_whale_open/
├── configs/                  # 配置文件
│   ├── whale.py             # Whale轨道配置
│   └── dcase.py             # DCASE轨道配置
├── Model/                   # 深度学习模型
│   ├── BA_Conv.py           # 双分支非对称卷积
│   ├── MultiScale_Ldsa.py   # 多尺度LDSA注意力
│   ├── Conformer_DCASE.py   # Conformer模型
│   ├── FAF_Filt.py          # 傅里叶滤波器模型
│   ├── experimental/        # 实验性模型
│   └── losses/              # 损失函数
├── Train/                   # 训练工具
│   ├── train_utils.py       # 训练器
│   ├── train_utils_dcase.py # DCASE训练器
│   └── dcase_trainer.py     # DCASE专用训练器
├── Data/                    # 数据处理
│   ├── audio_dataset.py     # Whale数据集
│   ├── dcase_dataset.py     # DCASE数据集
│   └── augmentation/        # 数据增强
├── evaluators/              # 评估工具模块
├── experiments/             # 实验脚本
├── config.py                # Whale配置入口
├── config_dcase.py          # DCASE配置入口
├── train.py                 # Whale训练入口
├── run_multi_seed_experiments.py  # DCASE多种子实验
├── eval_single_model.py     # 单模型评估
├── eval_batch_models.py     # 批量模型评估
├── evaluation_metrics.py    # Whale评估指标
└── evaluation_metrics_dcase.py  # DCASE评估指标
```

## 🔧 Dataset Synthesis

**Important**: Before running experiments, you need to synthesize datasets. See [DATASET_SYNTHESIS.md](DATASET_SYNTHESIS.md) for detailed instructions on:

1. **Whale Dataset Synthesis**:
   - Uses Scaper + DCLDE whale recordings
   - Generates SNR-grouped data (high/medium/low/very_low)
   - Located in `Data/dclde_synthetic/soundscaper.py`

2. **DCASE Dataset Synthesis**:
   - Requires DCASE2021 SoundBank audio files
   - Uses Scaper for polyphonic mixing
   - Generates SNR-stratified splits
   - Located in `Data/dcase_synthetic/generate_snr_grouped_dcase_scaper.py`

## 📊 Evaluation

### 单模型评估

```bash
# Whale数据集
python eval_single_model.py --seed 63 --model daap
python eval_single_model.py --seed 64 --model faf

# DCASE数据集
python eval_single_model.py --dataset dcase --seed 42 --model conformer
python eval_single_model.py --dataset dcase --model_path path/to/model.pth --model faf
```

### 批量评估

```bash
# 评估多个seed的模型
python eval_batch_models.py --seeds 63 64 65 --model daap
python eval_batch_models.py --seeds 63 64 65 --model daap faf conformer

# DCASE数据集批量评估
python eval_batch_models.py --dataset dcase --seeds 42 43 44 --model conformer
```

### 评估指标

- **Whale Track**: PSDS (Polyphonic Sound Detection Score), F1, Precision, Recall
- **DCASE Track**: Segment-based F1, Event-based F1, Frame-level accuracy

## 📈 Expected Results

### Whale Track (Low SNR)
- **Models compared**: CRNN, CNN-Transformer, Conformer, RA-Conv, Inception-Conv, LDSA

### DCASE Track
- **Best F1 Score**: Conformer and FAF-Filt variants
- **10 classes**: Alarm, Blender, Cat, Dishes, Dog, Electric_shaver, Frying, Running_water, Speech, Vacuum

## 🎨 Model Variants

**Whale Track**:
- `Whale_Model_Attention_MultiScale_Ldsa` (DAAPNet)

**DCASE Track**:
- `DCASE_Model_Attention_MultiScale` (DAAPNet for DCASE)
- `Conformer_DCASE` / `Conformer_DCASE_Optimized`
- `FAF_Filt_Model` (Frequency-aware Fourier filters)

## 📝 Configuration

Key settings in `config.py` (Whale) and `config_dcase.py` (DCASE):

**Whale Track**:
- Audio: 8 kHz, 1024 FFT, 256 hop
- Batch: 64, OneCycleLR scheduler
- Mixed precision FP16

**DCASE Track**:
- Audio: 16 kHz, 512 freq bins
- Batch: 64, CosineAnnealingLR
- 10-class classification

**Multi-Seed Fusion Config**:
- LR: 2.3e-4, Warmup: 8 epochs
- EMA: 0.995, Early stopping: patience=18

## 📚 References

- Paper: "Dynamic Attention-Asymmetric Perceptron Network for Overlapping Sound Event Detection"
- DCASE Challenge: https://dcase.community/
- DCLDE: Detection and Classification of Whale Recordings

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE)



