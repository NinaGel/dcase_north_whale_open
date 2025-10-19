import torch
from torch import nn
from pathlib import Path

# 项目根目录 (configs的上级目录)
ROOT_DIR = Path(__file__).parent.parent

# ===== DCASE2020实验路径配置 =====
EXPERIMENT_PATHS = {
    'dcase_attention_comparison': ROOT_DIR / 'experiments' / 'DCASE_Results' / 'attention_comparison',
    'dcase_model_comparison': ROOT_DIR / 'experiments' / 'DCASE_Results' / 'model_comparison',
    'dcase_conv_comparison': ROOT_DIR / 'experiments' / 'DCASE_Results' / 'conv_comparison',
    'dcase_dynamic_conv': ROOT_DIR / 'experiments' / 'DCASE_Results' / 'dynamic_conv'
}

# 实验子目录
EXPERIMENT_SUBDIRS = {
    'models': 'models',
    'checkpoints': 'checkpoints',
    'logs': 'logs',
    'curves': 'curves',
    'analysis': 'analysis',
    'reports': 'reports',
    'metrics': 'metrics'
}

# ===== DCASE路径配置 =====
DCASE_PATH_CONFIG = {
    # SNR分层数据集路径（DESED标准：16kHz, SNR 6-30dB）
    'snr_data_path': ROOT_DIR / 'Data' / 'dcase_synthetic_10k' / 'dcase_snr_desed_standard',

    # 结果保存路径
    'model_save_path': ROOT_DIR / 'experiments_dcase' / 'DCASE_Results',
    'image_save_path': ROOT_DIR / 'Result' / 'dcase_images'
}

# ===== DCASE音频参数 =====
DCASE_AUDIO_CONFIG = {
    'sr': 16000,  # DESED标准采样率
    'max_freq': 8000,  # 最大频率 (Hz) - 使用完整频谱以保留所有声音事件信息
    'freq': int(1024 * 8000 / 16000),  # 512 bins (0-8kHz 完整频谱)
    'frame': 311,  # int((16000*10 - 1024) / 512) + 1 = 311
    'window': 'hann',
    'n_fft': 1024,
    'hop_length': 512,
    'n_mels': 128,
    'fmax': 8000  # Mel频谱的最大频率
}

# ===== DCASE模型配置 =====
DCASE_MODEL_CONFIG = {
    # DCASE事件类别（按字母顺序）
    'class_names': [
        'Alarm_bell_ringing',
        'Blender',
        'Cat',
        'Dishes',
        'Dog',
        'Electric_shaver_toothbrush',
        'Frying',
        'Running_water',
        'Speech',
        'Vacuum_cleaner'
    ],
    'num_classes': 10,

    # BSA_Conv1参数 (输入: [batch, 1, 512, 311])
    'bsa_conv1': {
        'in_channel': 1,
        'c1_out': 4,
        'c2_out': 4,
        'c3_out': 8,
        'use_1x1conv': True,
        'strides': (2, 1),
        'avg_kernel_size': (4, 1)  # 从(2,1)增加到(4,1)，加速训练
    },

    # BSA_Conv2参数
    'bsa_conv2': {
        'in_channel': 8,
        'c1_out': 16,
        'c2_out': 16,
        'c3_out': 32,
        'use_1x1conv': True,
        'strides': (2, 1),
        'avg_kernel_size': (2, 1)
    },

    # Spatial_shift参数
    'spatial_shift': {
        'n': 4,
        'dropout': 0.5  # 从0.45增加到0.5，增强正则化
    },

    # GRU参数
    'gru': {
        'hidden_size': 256,
        'num_layers': 4,
        'dropout': 0.6  # 从0.5增加到0.6，对抗过拟合
    },

    # 注意力参数
    'attention': {
        'n_head': 8,
        'context_size': 55,
        'dropout': 0.4  # 从0.35增加到0.4，增强正则化
    }
}

# ===== 特征维度计算 =====
def calculate_dcase_feature_dims(bsa_conv1_stride, bsa_conv2_stride,
                                bsa_conv1_avg_kernel_size, bsa_conv2_avg_kernel_size,
                                bsa_conv2_out_channels):
    """计算DCASE模型中的特征维度

    输入维度: [batch, 1, 512, 311] (512 freq bins for 0-8kHz @ 16kHz sr)
    经过BSA_Conv1 (stride=(2,1), avg_pool=(4,1)): ← 增强池化
        freq: 512 -> 256 -> 64
        time: 311 -> 311 -> 311
    经过BSA_Conv2 (stride=(2,1), avg_pool=(2,1)):
        freq: 64 -> 32 -> 16
        time: 311 -> 311 -> 311
    最终特征: [batch, 32, 16, 311] -> reshape -> [batch, 512, 311]

    优势：GRU输入维度512（与原始128 freq bins配置相同），训练速度提升~2x
    """
    freq_dim = DCASE_AUDIO_CONFIG['freq']  # 512
    time_dim = DCASE_AUDIO_CONFIG['frame']  # 311

    # BSA_Conv1降采样 (512 -> 256 -> 64)
    freq_dim = freq_dim // bsa_conv1_stride[0] // bsa_conv1_avg_kernel_size[0]  # 64
    time_dim = time_dim // bsa_conv1_stride[1] // bsa_conv1_avg_kernel_size[1]  # 311

    # BSA_Conv2降采样 (64 -> 32 -> 16)
    freq_dim = freq_dim // bsa_conv2_stride[0] // bsa_conv2_avg_kernel_size[0]  # 16
    time_dim = time_dim // bsa_conv2_stride[1] // bsa_conv2_avg_kernel_size[1]  # 311

    # 计算最终特征维度
    n_feat = bsa_conv2_out_channels * freq_dim  # 32 * 16 = 512

    return n_feat, freq_dim, time_dim

# 计算并添加特征维度到配置
n_feat, freq_dim, time_dim = calculate_dcase_feature_dims(
    DCASE_MODEL_CONFIG['bsa_conv1']['strides'],
    DCASE_MODEL_CONFIG['bsa_conv2']['strides'],
    DCASE_MODEL_CONFIG['bsa_conv1']['avg_kernel_size'],
    DCASE_MODEL_CONFIG['bsa_conv2']['avg_kernel_size'],
    DCASE_MODEL_CONFIG['bsa_conv2']['c3_out']
)

DCASE_MODEL_CONFIG['attention']['n_feat'] = n_feat
DCASE_MODEL_CONFIG['feature_dims'] = {
    'n_feat': n_feat,
    'freq_dim': freq_dim,
    'time_dim': time_dim
}

# ===== 注意力机制配置 =====
# LDSA配置
DCASE_LDSA_CONFIG = {
    'n_head': 8,
    'dropout_rate': 0.3,
    'context_size': 55,
    'use_bias': False
}

# DDSA配置（多尺度）
DCASE_DDSA_CONFIG = {
    'n_head': 8,
    'dropout_rate': 0.3,
    'use_bias': False,
    'context_sizes': [21, 41, 61, 81]
}

# ===== 训练配置 =====
# 优化器配置（基于诊断报告优化）
DCASE_OPTIMIZER_CONFIG = {
    'lr': 1e-4,  # 从1e-5提升10倍（诊断报告建议）
    'weight_decay': 0.01,  # 从0.05降低到0.01
    'betas': (0.9, 0.95),
    'eps': 1e-8
}

# 训练配置（基于诊断报告优化）
DCASE_TRAIN_CONFIG = {
    # 基础训练参数
    'epochs': 200,  # 从120延长到200（诊断报告建议）
    'batch_size': 64,  # 从8增大到32（诊断报告建议）
    'optimizer': torch.optim.AdamW,
    'optimizer_params': DCASE_OPTIMIZER_CONFIG,

    # 混合精度训练
    'mixed_precision': True,

    # 梯度相关
    'grad_clip': True,
    'grad_clip_value': 1.0,
    'gradient_accumulation_steps': 4,  # 从1增加到4，有效batch=128

    # 早停和调度器
    'patience': 30,  # 从12增加到30（更长训练周期）
    'warm_up_epochs': 20,  # 从10增加到20
    'scheduler': {
        'type': 'cosine',  # 从one_cycle改为cosine（更适合长训练）
        'T_max': 200,
        'eta_min': 1e-6,
        'warmup_epochs': 20
    }
}

# GPU配置
DCASE_GPU_CONFIG = {
    'device': 'cuda',
    'precision': 'float16',
    'cudnn_benchmark': True,
    'cudnn_deterministic': True,
    'num_workers': 8,
    'pin_memory': True
}

# ===== 数据增强配置 =====
DCASE_AUGMENTATION_CONFIG = {
    'enabled': True,
    'augment_prob': 0.5,

    # 时域增强
    'time_domain': {
        'enabled': True,
        'prob': 0.3,
        'time_stretch': {'enabled': True, 'range': (0.8, 1.2)},
        'pitch_shift': {'enabled': True, 'range': (-2, 2)},
        'gain': {'enabled': True, 'range': (-6.0, 6.0)},
        'noise': {
            'enabled': True,
            'types': ['gaussian', 'pink'],
            'snr_range': (10.0, 20.0)
        }
    },

    # 频域增强 (SpecAugment)
    'spec_augment': {
        'enabled': True,
        'prob': 0.7,
        'freq_mask': {'param': 30, 'num_masks': 2},
        'time_mask': {'param': 30, 'num_masks': 2}
    },

    # 混合增强
    'mixing': {
        'enabled': True,
        'mixup': {'enabled': True, 'prob': 0.5, 'alpha': 0.5},
        'cutmix': {'enabled': True, 'prob': 0.5, 'alpha': 0.5},
        'random_erase': {
            'enabled': True,
            'prob': 0.3,
            'area_ratio': (0.02, 0.2),
            'aspect_ratio': (0.3, 3.3),
            'noise_std': 0.1
        }
    }
}

# ===== 评估配置 =====
DCASE_EVALUATION_CONFIG = {
    'metrics': [
        'precision', 'recall', 'f1_score',
        'segment_based_f1', 'event_based_f1',
        'psds_score'
    ],
    'thresholds': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'collar_tolerance': 0.2,
    'percentage_tolerance': 0.1
}

# ===== 损失函数和优化器 =====
# Option 1: Standard BCE Loss (baseline)
loss_fn = nn.BCEWithLogitsLoss(reduction='mean')

# Option 2: Unified Loss Function (ULF) - Recommended for imbalanced data
# from Model.losses import create_ulf_loss

# DCASE dataset with paper settings (alpha=0.5, gamma=4.0)
# loss_fn = create_ulf_loss('dcase')

# Custom settings example (uncomment to use):
# loss_fn = create_ulf_loss('dcase', custom_params={
#     'alpha': 0.7,   # Increase active frame weight
#     'gamma': 5.0,   # Stronger hard example focusing
#     'xi': 5.0
# })

optimizer = torch.optim.AdamW

# 为实验脚本提供兼容别名
OPTIMIZER_CONFIG = DCASE_OPTIMIZER_CONFIG
TRAIN_CONFIG = DCASE_TRAIN_CONFIG
AUDIO_CONFIG = DCASE_AUDIO_CONFIG
MODEL_CONFIG = DCASE_MODEL_CONFIG
GPU_CONFIG = DCASE_GPU_CONFIG
AUGMENTATION_CONFIG = DCASE_AUGMENTATION_CONFIG
EVALUATION_CONFIG = DCASE_EVALUATION_CONFIG
PATH_CONFIG = DCASE_PATH_CONFIG
LDSA_CONFIG = DCASE_LDSA_CONFIG
DDSA_CONFIG = DCASE_DDSA_CONFIG
