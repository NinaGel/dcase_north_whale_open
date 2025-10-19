import torch
from torch import nn
from pathlib import Path

# 项目根目录 (configs的上级目录)
ROOT_DIR = Path(__file__).parent.parent

# ===== 实验路径配置 =====
EXPERIMENT_PATHS = {
    'attention_comparison': ROOT_DIR / 'experiments' / 'Snr_Results' / 'attention_comparison',
    'ablation_study': ROOT_DIR / 'experiments' / 'Snr_Results' / 'ablation_study',
    'model_comparison': ROOT_DIR / 'experiments' / 'Snr_Results' / 'model_comparison',
    'conv_comparison': ROOT_DIR / 'experiments' / 'Snr_Results' / 'conv_comparison',
    'dynamic_conv': ROOT_DIR / 'experiments' / 'Snr_Results' / 'dynamic_conv'
}

# 实验子目录
EXPERIMENT_SUBDIRS = {
    'models': 'models',           # 模型保存目录
    'checkpoints': 'checkpoints', # 检查点保存目录
    'logs': 'logs',              # 日志保存目录
    'curves': 'curves',          # 训练曲线保存目录
    'analysis': 'analysis',      # 分析结果保存目录
    'reports': 'reports',         # 报告保存目录
    'metrics': 'metrics'         # 指标保存目录
}

# config.py
GENERATION_CONFIG = {
    'paths': {
        'pure_events_dir': 'data/pure_events',
        'backgrounds_dir': 'data/backgrounds',
        'output_dir': 'output/dataset',
        'analysis_file': 'output/overlap_analysis.json'
    },
    'audio': {
        'sr': 8000,
        'duration': 10.0
    },
    'generation': {
        'snr_groups': {
            'low': {'count': 3000, 'range': (15, 20)},
            'medium': {'count': 3000, 'range': (20, 25)},
            'high': {'count': 3000, 'range': (25, 30)}
        },
        'augmentation': {
            'none': 0.3,
            'single': 0.4,
            'mixed': 0.3
        }
    }
}

# 路径配置，所有路径都相对于项目根目录
PATH_CONFIG = {
    'test_path': ROOT_DIR / 'Data' / 'snr_scaper_audio' / 'test',
    'val_path': ROOT_DIR / 'Data' / 'snr_scaper_audio' / 'val',
    'image_save_path': ROOT_DIR / 'Result' / 'image',
    'txt_save_path': ROOT_DIR / 'Result' / 'act_img_11.13_eval_result',
    'model_save_path': ROOT_DIR / 'experiments' / 'Snr_Results',
    'snr_data_path': ROOT_DIR / 'Data' / 'snr_scaper_audio'
}

# 音频参数
AUDIO_CONFIG = {
    'sr': 8000,
    'freq': int(1024 * 2000 / 8000),  # 修改为2kHz对应的频率bin数
    'frame': 309,
    'window': 'hann',
    'n_fft': 1024,
    'hop_length': 256,
}


# 特征维度计算函数
def calculate_feature_dims(bsa_conv1_stride, bsa_conv2_stride, bsa_conv1_avg_kernel_size, bsa_conv2_avg_kernel_size,
                           bsa_conv2_out_channels):
    """计算模型中的特征维度

    Args:
        bsa_conv1_stride: BSA_Conv1的步长，如(2, 1)
        bsa_conv2_stride: BSA_Conv2的步长，如(2, 1)
        bsa_conv2_out_channels: BSA_Conv2的输出通道数

    Returns:
        n_feat: 特征维度
        freq_dim: 最终频率维度
        time_dim: 最终时间维度
    """
    # 初始维度
    freq_dim = AUDIO_CONFIG['freq']  # 现在是2kHz对应的bin数
    time_dim = AUDIO_CONFIG['frame']

    # BSA_Conv1降采样
    freq_dim = freq_dim // bsa_conv1_avg_kernel_size[0]
    freq_dim = freq_dim // bsa_conv1_stride[0]
    time_dim = time_dim // bsa_conv1_avg_kernel_size[1]
    time_dim = time_dim // bsa_conv1_stride[1]
    # BSA_Conv2降采样
    freq_dim = freq_dim // bsa_conv2_avg_kernel_size[0]
    freq_dim = freq_dim // bsa_conv2_stride[0]
    time_dim = time_dim // bsa_conv2_avg_kernel_size[1]
    time_dim = time_dim // bsa_conv2_stride[1]

    # 计算最终特征维度
    n_feat = bsa_conv2_out_channels * freq_dim

    return n_feat, freq_dim, time_dim


# 模型架构参数
MODEL_CONFIG = {
    # 添加类别名称列表
    'class_names': ['upcall', 'gunshot', 'scream', 'moancall'],
    
    # BSA_Conv1参数 - 第一次降采样
    'bsa_conv1': {
        'in_channel': 1,
        'c1_out': 4,  # 增加通道数
        'c2_out': 4,
        'c3_out': 8,
        'use_1x1conv': True,
        'strides': (2, 1),
        'avg_kernel_size': (2, 1)
    },

    # BSA_Conv2参数 - 第二次降采样
    'bsa_conv2': {
        'in_channel': 8,
        'c1_out': 16,  # 增加通道数
        'c2_out': 16,
        'c3_out': 32,  # 保持64，因为64 * 16 = 1024，可以被16整除
        'use_1x1conv': True,
        'strides': (2, 1),
        'avg_kernel_size': (2, 1)
    },

    # Spatial_shift参数
    'spatial_shift': {
        'n': 4,  # 增加空间位移次数
        'dropout': 0.4  # 降低dropout
    },

    # GRU参数
    'gru': {
        'hidden_size': 256,  # 增加隐藏层大小
        'num_layers': 4,     # 减少层数但增加每层容量
        'dropout': 0.3       # 降低dropout
    },

    # 注意力参数
    'attention': {
        'n_head': 8,       # 将注意力头数改为16
        'context_size': 45,
        'dropout': 0.3      # 降低dropout
    },

    'num_classes': 4
}

# 对应LDSA的配置
LDSA_CONFIG = {
    'n_head': 8,
    'dropout_rate': 0.3,
    'context_size': 45,
    'use_bias': False
}


# 对应DDSA的配置
DDSA_CONFIG = {
    'n_head': 8,  # 将注意力头数改为16，使其能整除特征维度1024
    'dropout_rate': 0.5,  # 降低dropout
    'use_bias': False,  # 启用偏置项
    'context_sizes': [17, 33, 55],  # 增加一个中间尺度
    'regularization': {
        'l2_weight': 0.02,           # 降低L2正则化
        'attention_dropout': 0.4,     # 降低dropout
        'feature_dropout': 0.3,
        'stochastic_depth_rate': 0.2, # 降低随机深度
        'layer_drop': 0.1,          # 降低层dropout
        'gradient_clip': 0.3         # 放宽梯度裁剪
    }
}

# 计算并添加特征维度到配置中
n_feat, freq_dim, time_dim = calculate_feature_dims(
    MODEL_CONFIG['bsa_conv1']['strides'],
    MODEL_CONFIG['bsa_conv2']['strides'],
    MODEL_CONFIG['bsa_conv1']['avg_kernel_size'],
    MODEL_CONFIG['bsa_conv2']['avg_kernel_size'],
    MODEL_CONFIG['bsa_conv2']['c3_out']
)

MODEL_CONFIG['attention']['n_feat'] = n_feat
MODEL_CONFIG['feature_dims'] = {
    'n_feat': n_feat,
    'freq_dim': freq_dim,
    'time_dim': time_dim
}

# 学习率配置
LR_CONFIG = {
    'initial_lr': 1.86e-5,
    'min_lr': 1e-6,
    'warmup_epochs': 5
}

# 更新优化器配置
OPTIMIZER_CONFIG = {
    'lr': LR_CONFIG['initial_lr'] * (128/32)**0.5,  # 根据batch_size调整基础学习率
    'weight_decay': 6.56e-6,
    'betas': (0.9, 0.999),  # Adam优化器参数
    'eps': 1e-8
}

# 训练参数
dropout = 0.3
batch_size = 1  # 增大批大小以充分利用4090显存
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW

# 训练配置
TRAIN_CONFIG = {
    'epochs': 2,  # 增加训练轮数
    'batch_size': 64,
    'optimizer': optimizer,
    'optimizer_params': {
        'lr': 2e-4,  # 略微提高学习率
        'weight_decay': 0.01,  # 降低L2正则化强度
        'betas': (0.9, 0.95),  # 调整beta2以加快学习
        'eps': 1e-8
    },
    'mixed_precision': True,  # 启用混合精度训练
    'gradient_accumulation_steps': 2,  # 减少梯度累积步数以加快收敛
    'max_grad_norm': 1.0,    # 放宽梯度裁剪阈值
    'grad_clip': True,
    'grad_clip_value': 1.0,
    'patience': 20,          # 增加早停耐心值
    'warm_up_epochs': 10,     # 减少预热轮数
    'memory_efficient': True,
    'backward_cleanup': True,
    'scheduler': {
        'type': 'one_cycle',  # 使用OneCycleLR调度器
        'max_lr': 2e-4,
        'pct_start': 0.3,
        'div_factor': 25.0,
        'final_div_factor': 1e4
    },
    # 增强数据增强策略
    'augmentation': {
        # 混合增强
        'mixup_alpha': 0.4,        # 增加mixup强度
        'cutmix_alpha': 0.4,       # 增加cutmix强度
        'label_smoothing': 0.15,   # 增加标签平滑
        'random_erasing_prob': 0.3, # 增加随机擦除概率
        
        # SpecAugment增强
        'spec_augment': {
            'freq_mask_param': 30,   # 增加频率掩码范围
            'time_mask_param': 30,   # 增加时间掩码范围
            'num_freq_mask': 3,      # 增加频率掩码数量
            'num_time_mask': 3,      # 增加时间掩码数量
            'mask_prob': 0.5         # 掩码概率
        },
        
        # 添加新的增强方法
        'noise_augment': {
            'gaussian_noise_prob': 0.3,
            'gaussian_noise_scale': 0.01,
            'pink_noise_prob': 0.3,
            'pink_noise_scale': 0.01
        },
        
        'time_stretch': {
            'prob': 0.3,
            'rate_range': (0.8, 1.2)
        },
        
        'pitch_shift': {
            'prob': 0.3,
            'shift_range': (-2, 2)
        }
    }
}

# 添加GPU相关配置
GPU_CONFIG = {
    'device': 'cuda',
    'precision': 'float16',  # 使用FP16
    'cudnn_benchmark': True, # 启用cuDNN基准测试
    'cudnn_deterministic': True,  # 关闭确定性模式以提高性能
    'num_workers': 4,       # 数据加载器的工作进程数
    'pin_memory': True,     # 启用内存页锁定
    'memory_cleanup_freq': 10,  # 内存清理频率
    'max_memory_cached': 0.8,   # 最大缓存内存比例
}

# 添加验证配置
VALIDATION_CONFIG = {
    'eval_frequency': 1,     # 每轮评估一次
    'save_best_only': True,  # 只保存最佳模型
    'early_stopping': {
        'monitor': 'val_loss',
        'mode': 'min',
        'patience': 15,
        'min_delta': 1e-4
    }
}

# 数据增强配置
AUGMENTATION_CONFIG = {
    # 基础参数
    'enabled': True,
    'augment_prob': 0.5,  # 降低总体增强概率
    
    # 时域增强
    'time_domain': {
        'enabled': True,
        'prob': 0.3,  # 降低时域增强概率
        'time_stretch': {
            'enabled': True,
            'range': (0.8, 1.2)  # 缩小时间拉伸范围
        },
        'pitch_shift': {
            'enabled': True,
            'range': (-2, 2)  # 缩小音高偏移范围
        },
        'gain': {
            'enabled': True,
            'range': (-6.0, 6.0)  # 缩小增益范围
        },
        'noise': {
            'enabled': True,
            'types': ['gaussian', 'pink'],
            'snr_range': (10.0, 20.0)  # 提高信噪比范围
        }
    },
    
    # 频域增强 (SpecAugment)
    'spec_augment': {
        'enabled': True,
        'prob': 0.5,  # 降低概率
        'freq_mask': {
            'param': 30,  # 减小掩码长度
            'num_masks': 2  # 减少掩码数量
        },
        'time_mask': {
            'param': 30,
            'num_masks': 2
        }
    },
    
    # 混合增强
    'mixing': {
        'enabled': True,
        'mixup': {
            'enabled': True,
            'prob': 0.4,
            'alpha': 0.4  # 降低mixup强度
        },
        'cutmix': {
            'enabled': True,
            'prob': 0.4,
            'alpha': 0.4
        },
        'random_erase': {
            'enabled': True,
            'prob': 0.3,
            'area_ratio': (0.02, 0.2),  # 缩小擦除区域范围
            'aspect_ratio': (0.3, 3.3),  # 调整长宽比范围
            'noise_std': 0.1  # 降低噪声强度
        }
    },
    
    # SNR相关增强
    'snr_based': {
        'enabled': True,
        'high_snr': {  # SNR > 5dB
            'noise_prob': 0.2,
            'spec_augment_prob': 0.3,
            'mixing_prob': 0.3
        },
        'medium_snr': {  # 0dB < SNR <= 5dB
            'noise_prob': 0.3,
            'spec_augment_prob': 0.4,
            'mixing_prob': 0.4
        },
        'low_snr': {  # -5dB < SNR <= 0dB
            'noise_prob': 0.4,
            'spec_augment_prob': 0.5,
            'mixing_prob': 0.5
        },
        'very_low_snr': {  # SNR <= -5dB
            'noise_prob': 0.5,
            'spec_augment_prob': 0.6,
            'mixing_prob': 0.6
        }
    }
}

# 添加LDSA优化配置
LDSA_OPTIMIZED_CONFIG = {
    'use_bias': False,  # 是否使用偏置项
    'dropout_rate': 0.3,  # dropout率
    'context_size': 45,  # 上下文窗口大小
    'n_head': 8,  # 注意力头数
    'regularization': {
        'l2_weight': 0.01,  # L2正则化权重
        'attention_dropout': 0.3,  # 注意力dropout
        'feature_dropout': 0.2,  # 特征dropout
        'gradient_clip': 1.0  # 梯度裁剪值
    }
}

# 添加多尺度LDSA配置
MULTISCALE_CONFIG = {
    'context_sizes': [31, 61],  # 上下文窗口大小列表
    'n_head': 8,  # 注意力头数
    'dropout_rate': 0.3,  # dropout率
    'regularization': {
        'l2_weight': 0.01,  # L2正则化权重
        'attention_dropout': 0.3,  # 注意力dropout
        'feature_dropout': 0.2,  # 特征dropout
        'gradient_clip': 1.0  # 梯度裁剪值
    }
}
