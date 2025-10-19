"""
多种子实验运行脚本 - 统一框架
Multi-seed Experiment Runner - Unified Framework

支持模型 / Supported Models:
- conformer_optimized: 优化版Conformer模型
- faf_heavy: FAF-Filt Heavy模型 (ICASSP 2025)
- daapnet: DAAPNet模型 (BCE损失)
- daapnet_ulf: DAAPNet + Unified Loss Function

功能 / Features:
1. 多随机种子训练（默认种子：42, 123, 456）
   Multi-seed training for reproducibility
2. 记录完整实验信息（模型参数、FLOPs、训练配置）
   Complete experiment logging (params, FLOPs, config)
3. 详细评估（3个SNR组 × 7个指标）
   Detailed evaluation (3 SNR groups × 7 metrics)
4. 自动保存训练历史和评估结果
   Auto-save training history and evaluation results
"""

# 必须在所有导入之前设置警告过滤
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*pkg_resources.*')

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import matplotlib
matplotlib.use('Agg')
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.cuda.amp as amp  # CUDA AMP utilities (GradScaler/autocast)
import random
import json
import pandas as pd
from datetime import datetime

# 导入配置和模块
import config_dcase as cfg
from Data.dcase_dataset import load_train_val_data, load_test_data
from evaluation_metrics_dcase import DCASEEventDetectionEvaluator, load_ground_truth_dcase


def set_seed(seed):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EMA:
    """指数移动平均（EMA）用于平滑模型权重"""
    def __init__(self, model, decay=0.995):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # 初始化shadow参数
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """更新EMA参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """应用EMA参数（用于评估）"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """恢复原始参数（用于训练）"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class LabelSmoothingBCELoss(torch.nn.Module):
    """带Label Smoothing的BCE损失"""
    def __init__(self, smoothing=0.005):
        super().__init__()
        self.smoothing = smoothing
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred, target):
        # Label smoothing: y_smooth = y * (1 - smoothing) + 0.5 * smoothing
        target_smooth = target * (1.0 - self.smoothing) + 0.5 * self.smoothing
        loss = self.bce(pred, target_smooth)
        return loss.mean()


def apply_batch_mixup(features, labels, alpha=0.5, prob=0.5):
    """在Batch级别应用Mixup增强
    
    Args:
        features: [batch, channel, freq, time]
        labels: [batch, time, num_classes]
        alpha: Beta分布参数，越大混合越均匀
        prob: 应用Mixup的概率
    
    Returns:
        mixed_features, mixed_labels
    """
    if random.random() > prob or features.size(0) < 2:
        return features, labels
    
    # 从Beta分布采样混合比例
    lam = np.random.beta(alpha, alpha)
    
    batch_size = features.size(0)
    # 随机打乱索引
    index = torch.randperm(batch_size, device=features.device)
    
    # 混合特征和标签
    mixed_features = lam * features + (1 - lam) * features[index]
    mixed_labels = lam * labels + (1 - lam) * labels[index]
    
    return mixed_features, mixed_labels


def initialize_model(model_type, model_config=None):
    """初始化模型

    Args:
        model_type: 模型类型 ('conformer_optimized', 'faf_heavy', 'daapnet', 'daapnet_ulf')
        model_config: 模型配置字典（用于FAF模型）

    Returns:
        model: 初始化的模型
        model_info: 模型信息字典
    """
    if model_type == 'conformer_optimized':
        from Model.Conformer_DCASE_Optimized import ConformerDCASE_Optimized
        num_classes = len(cfg.DCASE_MODEL_CONFIG['class_names'])
        model = ConformerDCASE_Optimized(num_classes=num_classes)
        model_config = {
            'num_classes': num_classes,
            'd_model': 144,
            'conformer_blocks': 3,
            'attention_heads': 4,
            'architecture': '7-layer CNN + 3 Conformer blocks (d_model=144)',
            'paper': 'Optimized version based on Barahona et al., IEEE/ACM TASLP 2024'
        }

    elif model_type == 'faf_heavy':
        from Model.FAF_Filt import FAF_Filt_Model

        # FAF Heavy 配置（基于ICASSP 2025论文）
        # Paper: "FAF-Filt: Frequency-aware Fourier Filter for Sound Event Detection"
        # Authors: Siyu Sun et al., ByteDance China
        num_classes = len(cfg.DCASE_MODEL_CONFIG['class_names'])

        # Heavy配置：更大的通道数以达到接近论文的参数量
        # 论文报告4.85M参数，当前配置约为4-5M参数
        conv_channels = [64, 128, 256, 256, 256]  # 5个卷积块
        gru_hidden = 256  # BiGRU隐藏层大小

        print(f"\n[FAF-Filt配置]")
        print(f"  输入频率维度: {cfg.DCASE_AUDIO_CONFIG['freq']} (DCASE ACT)")
        print(f"  投影目标维度: 128 (降低计算量)")
        print(f"  卷积通道数: {conv_channels}")
        print(f"  GRU隐藏层: {gru_hidden}")
        print(f"  Reduction ratio: 4 (论文标准)")

        model = FAF_Filt_Model(
            num_classes=num_classes,
            input_freq_bins=cfg.DCASE_AUDIO_CONFIG['freq'],  # 512 for DCASE ACT
            conv_channels=conv_channels,
            gru_hidden=gru_hidden,
            gru_layers=2,  # BiGRU层数
            reduction_ratio=4,  # SE模块降维比例（论文使用4）
            use_projection=True,  # 启用频率投影512→128
            projection_method='conv1d',  # 使用Conv1d投影（高效）
            projection_target=128  # 投影到128维
        )

        model_config = {
            'num_classes': num_classes,
            'config': 'heavy',
            'conv_channels': conv_channels,
            'gru_hidden': gru_hidden,
            'gru_layers': 2,
            'reduction_ratio': 4,
            'use_projection': True,
            'projection_method': 'conv1d',
            'projection_target': 128,
            'paper': 'FAF-Filt (ICASSP 2025)',
            'architecture': 'Conv2d→BN→CG→Pool + [FAF-Filt→FA-Conv→BN→CG→Pool]×4 + BiGRU→Linear'
        }

    elif model_type == 'daapnet':
        # DAAPNet - 使用DCASE_Model_Attention_MultiScale
        from Model.DCASE_Model_Attention_MultiScale import DCASE_Model_Attention_MultiScale
        model = DCASE_Model_Attention_MultiScale()
        model_config = {}

    elif model_type == 'daapnet_ulf':
        # DAAPNet + ULF (Standard) - 使用标准ULF参数
        from Model.DCASE_Model_Attention_MultiScale import DCASE_Model_Attention_MultiScale
        model = DCASE_Model_Attention_MultiScale()
        model_config = {'loss': 'Unified Loss Function (Standard)'}

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # 获取模型信息
    model_info = {}
    if hasattr(model, 'get_model_info'):
        model_info = model.get_model_info()
    else:
        # 手动计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_info = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_name': model_type
        }

    # 添加额外信息
    model_info['model_type'] = model_type
    if model_config:
        model_info['model_config'] = model_config

    return model, model_info


def get_loss_function(model_type, label_smoothing=0.005):
    """获取损失函数

    Args:
        model_type: 模型类型
        label_smoothing: Label smoothing强度

    Returns:
        loss_fn: 损失函数
        loss_info: 损失函数信息字典
    """
    if model_type == 'daapnet_ulf':
        from Model.losses import create_ulf_loss

        # DAAPNet + ULF (Standard): 使用标准DCASE配置
        loss_fn = create_ulf_loss('dcase')

        loss_info = {
            'type': 'Unified Loss Function (Standard)',
            'config': 'dcase',
            'alpha': 0.5,
            'beta': 1.0,
            'rho': 1.0,
            'tau': 1.0,
            'gamma': 4.0,
            'xi': 4.0
        }
    else:
        # BCE 损失 + Label Smoothing（conformer, faf_heavy, daapnet都使用）
        if label_smoothing > 0:
            loss_fn = LabelSmoothingBCELoss(smoothing=label_smoothing)
            loss_info = {
                'type': 'BCEWithLogitsLoss + Label Smoothing',
                'label_smoothing': label_smoothing,
                'reduction': 'mean'
            }
        else:
            loss_fn = cfg.loss_fn
            loss_info = {
                'type': 'BCEWithLogitsLoss',
                'reduction': 'mean'
            }

    return loss_fn, loss_info


def calculate_model_flops(model, model_type='unknown'):
    """计算模型FLOPs（使用thop）

    Args:
        model: PyTorch模型
        model_type: 模型类型（用于确定正确的输入形状）

    Returns:
        dict: FLOPs信息
    """
    try:
        from thop import profile, clever_format

        device = next(model.parameters()).device

        # 根据模型类型设置正确的输入形状
        # DCASE数据集：ACT特征为512频率bins，311时间帧
        if model_type in ['faf_heavy', 'conformer_optimized', 'daapnet_ulf']:
            # FAF-Filt和优化版Conformer期望: [batch, freq, time] 或 [batch, 1, freq, time]
            input_shape = (1, 512, 311)  # [batch=1, freq=512, time=311]
        elif model_type == 'daapnet':
            # DAAPNet期望: [batch, 1, freq, time]
            input_shape = (1, 1, 256, 311)  # [batch=1, channels=1, freq=256, time=311]
        else:
            # 默认形状
            input_shape = (1, 1, 256, 311)

        dummy_input = torch.randn(*input_shape).to(device)

        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        flops_readable, params_readable = clever_format([flops, params], "%.3f")

        return {
            'flops': flops,
            'flops_readable': flops_readable,
            'params': params,
            'params_readable': params_readable,
            'input_shape': input_shape
        }
    except Exception as e:
        print(f"  警告: FLOPs计算失败: {e}")
        return {
            'flops': 'N/A',
            'flops_readable': 'N/A',
            'params': 'N/A',
            'params_readable': 'N/A',
            'input_shape': 'N/A'
        }


def train_single_seed(
    model_type,
    seed,
    epochs=80,
    batch_size=32,
    lr=2.3e-4,  # 融合版配置
    weight_decay=0.01,
    scheduler_type='plateau',  # ReduceLROnPlateau
    gradient_clip=0.5,  # 融合版配置
    early_stop_patience=18,  # 融合版配置
    use_warmup=True,  # 融合版配置
    warmup_epochs=8,  # 融合版配置
    ema_decay=0.995,  # 融合版配置
    label_smoothing=0.005,  # 融合版配置
    scheduler_patience=6,  # 融合版配置
    scheduler_factor=0.6,  # 融合版配置
    model_config=None,
    exp_base_dir="experiments_dcase/multi_seed"
):
    """训练单个种子的模型

    Returns:
        dict: 包含训练历史、测试结果、模型路径等信息
    """
    print("\n" + "="*80)
    print(f"训练 {model_type.upper()} - Seed {seed}")
    print("="*80)

    # 设置种子
    set_seed(seed)

    # 创建实验目录
    exp_dir = Path(exp_base_dir) / model_type / f"seed_{seed}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    # 启用CUDA优化
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # 1. 加载数据
    print("\n" + "="*80)
    print("步骤 1: 加载数据集")
    print("="*80)

    train_loader, val_loaders = load_train_val_data(batch_size=batch_size)
    test_loaders = load_test_data(batch_size=batch_size)

    print(f"\n[OK] 训练集大小: {len(train_loader.dataset)} 样本")
    print(f"[OK] 验证集组数: {len(val_loaders)}")
    for snr_group, loader in val_loaders.items():
        print(f"  - {snr_group}: {len(loader.dataset)} 样本")

    # 2. 创建模型
    print("\n" + "="*80)
    print("步骤 2: 创建模型")
    print("="*80)

    model, model_info = initialize_model(model_type, model_config)
    model = model.to(device)

    print(f"\n[OK] 模型类型: {model_type}")
    print(f"[OK] 总参数: {model_info.get('total_params', 'N/A'):,}")
    print(f"[OK] 可训练参数: {model_info.get('trainable_params', 'N/A'):,}")

    # 计算FLOPs
    flops_info = calculate_model_flops(model, model_type=model_type)
    print(f"[OK] FLOPs: {flops_info['flops_readable']}")
    if 'input_shape' in flops_info and flops_info['input_shape'] != 'N/A':
        print(f"[OK] FLOPs输入形状: {flops_info['input_shape']}")

    # 3. 设置优化器和损失函数
    print("\n" + "="*80)
    print("步骤 3: 设置训练组件")
    print("="*80)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )

    loss_fn, loss_info = get_loss_function(model_type, label_smoothing=label_smoothing)
    print(f"\n[OK] 损失函数: {loss_info['type']}")
    if 'label_smoothing' in loss_info:
        print(f"[OK] Label Smoothing: {loss_info['label_smoothing']}")

    # 创建EMA
    ema = None
    if ema_decay is not None and ema_decay > 0:
        ema = EMA(model, decay=ema_decay)
        print(f"[OK] EMA已启用 (decay={ema_decay})")
    else:
        print("[OK] EMA已禁用")

    # 学习率调度器
    if scheduler_type == 'plateau':
        # ReduceLROnPlateau + Warmup
        main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_factor,
            patience=scheduler_patience,
            verbose=True,
            min_lr=1e-7
        )
        if use_warmup:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=warmup_epochs
            )
        else:
            warmup_scheduler = None
        scheduler_step_mode = 'epoch'
        print(f"[OK] 学习率调度器: ReduceLROnPlateau (factor={scheduler_factor}, patience={scheduler_patience})")
        if use_warmup:
            print(f"[OK] Warmup: {warmup_epochs} epochs")
    elif scheduler_type == 'onecycle':
        main_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr * 2,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1e4
        )
        warmup_scheduler = None
        scheduler_step_mode = 'batch'
    elif use_warmup:
        warmup_epoch_count = warmup_epochs if 'warmup_epochs' in locals() else int(epochs * 0.1)
        main_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.1, total_iters=warmup_epoch_count
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs-warmup_epoch_count, eta_min=1e-7
                )
            ],
            milestones=[warmup_epoch_count]
        )
        warmup_scheduler = None
        scheduler_step_mode = 'epoch'
    else:
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=1e-7
        )
        warmup_scheduler = None
        scheduler_step_mode = 'epoch'

    # 早停设置
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_model_state = None

    # 训练历史
    train_losses = []
    val_losses = {snr: [] for snr in val_loaders.keys()}
    learning_rates = []  # 记录每个epoch的学习率
    epoch_times = []  # 记录每个epoch的训练时间

    # 启用混合精度
    use_amp = device.type == 'cuda'
    scaler = amp.GradScaler(enabled=use_amp)

    # 4. 训练循环
    print("\n" + "="*80)
    print(f"步骤 4: 训练模型 ({epochs} epochs)")
    print("="*80)

    import time
    total_training_start = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        # 训练
        model.train()
        epoch_train_loss = 0.0
        train_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in pbar:
            features, labels, _ = batch

            # 输入维度处理
            # - DCASE数据集返回: [batch, freq, time]
            # - 大多数模型期望: [batch, 1, freq, time]
            # - FAF_Filt可接受: [batch, freq, time] 或 [batch, 1, freq, time]
            if features.dim() == 3:
                features = features.unsqueeze(1)  # [batch, freq, time] -> [batch, 1, freq, time]

            features = features.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)

            # 应用Batch级别Mixup增强（使用config_dcase中的配置）
            mixup_cfg = cfg.DCASE_AUGMENTATION_CONFIG['mixing']['mixup']
            if mixup_cfg['enabled']:
                features, labels = apply_batch_mixup(
                    features, labels, 
                    alpha=mixup_cfg['alpha'],
                    prob=mixup_cfg['prob']
                )

            optimizer.zero_grad()

            # 混合精度训练
            if use_amp:
                with amp.autocast(enabled=True):
                    outputs = model(features)
                    loss = loss_fn(outputs, labels)

                scaler.scale(loss).backward()

                if gradient_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

                scaler.step(optimizer)
                scaler.update()

                # 更新EMA
                if ema is not None:
                    ema.update()
            else:
                outputs = model(features)
                loss = loss_fn(outputs, labels)
                loss.backward()

                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

                optimizer.step()

                # 更新EMA
                if ema is not None:
                    ema.update()

            if scheduler_step_mode == 'batch':
                main_scheduler.step()

            epoch_train_loss += loss.item()
            train_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = epoch_train_loss / train_batches
        train_losses.append(avg_train_loss)

        # 验证（使用EMA权重）
        model.eval()
        if ema is not None:
            ema.apply_shadow()  # 应用EMA权重

        epoch_val_losses = {}

        with torch.no_grad():
            for snr_group, val_loader in val_loaders.items():
                val_loss = 0.0
                val_batches = 0

                for batch in val_loader:
                    features, labels, _ = batch

                    # 输入维度处理（同训练）
                    if features.dim() == 3:
                        features = features.unsqueeze(1)

                    features = features.to(device, dtype=torch.float32)
                    labels = labels.to(device, dtype=torch.float32)

                    if use_amp:
                        with amp.autocast(enabled=True):
                            outputs = model(features)
                            # 验证时使用原始BCE损失（不用label smoothing）
                            if label_smoothing > 0:
                                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                                    outputs, labels, reduction='mean'
                                )
                            else:
                                loss = loss_fn(outputs, labels)
                    else:
                        outputs = model(features)
                        if label_smoothing > 0:
                            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                                outputs, labels, reduction='mean'
                            )
                        else:
                            loss = loss_fn(outputs, labels)

                    val_loss += loss.item()
                    val_batches += 1

                avg_val_loss = val_loss / val_batches
                val_losses[snr_group].append(avg_val_loss)
                epoch_val_losses[snr_group] = avg_val_loss

        if ema is not None:
            ema.restore()  # 恢复训练权重

        # 更新学习率
        avg_val_loss = np.mean(list(epoch_val_losses.values()))

        if scheduler_step_mode == 'epoch':
            if scheduler_type == 'plateau':
                # Warmup期间使用warmup_scheduler，之后使用plateau scheduler
                if use_warmup and warmup_scheduler is not None and epoch < warmup_epochs:
                    warmup_scheduler.step()
                else:
                    main_scheduler.step(avg_val_loss)
            else:
                main_scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        # 记录epoch时间
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)

        # 打印训练进度
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  学习率: {current_lr:.6f}")
        print(f"  训练损失: {avg_train_loss:.4f}")
        print(f"  验证损失 (平均): {avg_val_loss:.4f}")
        for snr_group, loss_val in epoch_val_losses.items():
            print(f"    - {snr_group}: {loss_val:.4f}")
        print(f"  Epoch耗时: {epoch_time:.2f}秒")

        # 早停检查
        if early_stop_patience > 0:
            if avg_val_loss < best_val_loss - 1e-5:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                patience_counter = 0
                # 保存最佳模型（EMA权重）
                model.eval()
                if ema is not None:
                    ema.apply_shadow()
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                if ema is not None:
                    ema.restore()
                print(f"  *** 新的最佳验证损失: {best_val_loss:.4f} ***")
            else:
                patience_counter += 1
                print(f"  早停计数: {patience_counter}/{early_stop_patience}")

            if patience_counter >= early_stop_patience:
                print(f"\n[早停] 在epoch {epoch+1}触发，最佳epoch为{best_epoch+1}")
                break

    # 记录总训练时间
    total_training_time = time.time() - total_training_start

    # 恢复最佳模型
    if early_stop_patience > 0 and best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        print(f"\n已恢复最佳模型 (Epoch {best_epoch+1}, 验证损失: {best_val_loss:.4f})")

    print(f"\n总训练时间: {total_training_time/3600:.2f} 小时 ({total_training_time/60:.2f} 分钟)")

    # 5. 绘制训练曲线
    print("\n" + "="*80)
    print("步骤 5: 绘制训练曲线")
    print("="*80)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    for snr_group, losses in val_losses.items():
        plt.plot(losses, label=snr_group, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss by SNR Group')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(exp_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[OK] 训练曲线已保存至: {exp_dir / 'training_curves.png'}")

    # 6. 评估模型
    print("\n" + "="*80)
    print("步骤 6: 在测试集上评估模型")
    print("="*80)

    model.eval()
    test_results = {}

    for snr_group, test_loader in test_loaders.items():
        print(f"\n评估 {snr_group}...")

        # 准备评估器
        snr_suffix = snr_group.replace('snr_', '')
        metadata_file = cfg.DCASE_PATH_CONFIG['snr_data_path'] / snr_group / 'test' / f'test_{snr_suffix}.tsv'

        if not metadata_file.exists():
            print(f"  警告: 未找到metadata文件 {metadata_file}")
            continue

        # 加载ground truth
        ground_truth_df, ground_truth_dict, audio_durations = load_ground_truth_dcase(metadata_file)

        # 收集预测
        all_preds = []
        all_filenames = []

        with amp.autocast(enabled=use_amp), torch.no_grad():
            for batch in tqdm(test_loader, desc=f"  预测 {snr_group}", leave=False):
                features, labels, filenames = batch

                # 输入维度处理（同训练）
                if features.dim() == 3:
                    features = features.unsqueeze(1)

                features = features.to(device, dtype=torch.float32)
                outputs = model(features)

                all_preds.append(outputs.cpu())
                all_filenames.extend([Path(f).stem for f in filenames])

        all_preds = torch.cat(all_preds, dim=0)

        # 评估
        evaluator = DCASEEventDetectionEvaluator()
        metrics = evaluator.compute_all_metrics(
            all_preds,
            all_filenames,
            ground_truth_dict,
            ground_truth_df,
            audio_durations
        )

        test_results[snr_group] = metrics

        # 打印主要指标
        print(f"\n{snr_group} 测试结果:")
        print(f"  PSDS1: {metrics['psds1_score']:.4f}")
        print(f"  PSDS2: {metrics['psds2_score']:.4f}")
        print(f"  Macro pAUC: {metrics['macro_pauc']:.4f}")
        print(f"  Optimal F1: {metrics['optimal_macro_f1']:.4f}")
        print(f"  Segment F1 (micro): {metrics['segment_based_f1_micro']:.4f}")
        print(f"  Segment F1 (macro): {metrics['segment_based_f1_macro']:.4f}")
        print(f"  Event F1: {metrics['event_based_f1']:.4f}")

    # 7. 保存结果
    print("\n" + "="*80)
    print("步骤 7: 保存结果")
    print("="*80)

    # 保存模型
    model_path = exp_dir / 'best_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"\n[OK] 模型已保存至: {model_path}")

    # 保存完整实验信息
    experiment_info = {
        'model_type': model_type,
        'seed': seed,
        'timestamp': datetime.now().isoformat(),

        # 数据集信息
        'dataset_info': {
            'train_samples': len(train_loader.dataset),
            'val_samples': {snr: len(loader.dataset) for snr, loader in val_loaders.items()},
            'test_samples': {snr: len(loader.dataset) for snr, loader in test_loaders.items()},
            'num_classes': len(cfg.DCASE_MODEL_CONFIG['class_names']),
            'class_names': cfg.DCASE_MODEL_CONFIG['class_names']
        },

        # 模型信息
        'model_info': {
            **model_info,
            'flops': flops_info
        },

        # 训练配置
        'training_config': {
            'epochs': epochs,
            'actual_epochs': len(train_losses),
            'batch_size': batch_size,
            'learning_rate': lr,
            'weight_decay': weight_decay,
            'scheduler_type': scheduler_type,
            'use_warmup': use_warmup,
            'warmup_epochs': warmup_epochs if use_warmup else 0,
            'ema_decay': ema_decay,
            'label_smoothing': label_smoothing,
            'gradient_clip': gradient_clip,
            'early_stop_patience': early_stop_patience,
            'scheduler_patience': scheduler_patience if scheduler_type == 'plateau' else 'N/A',
            'scheduler_factor': scheduler_factor if scheduler_type == 'plateau' else 'N/A',
            'use_mixed_precision': use_amp,
            'optimization_strategy': 'Fusion Config (Config3 + Config4)'
        },

        # 损失函数信息
        'loss_info': loss_info,

        # 训练历史
        'training_history': {
            'train_losses': [float(x) for x in train_losses],
            'val_losses': {k: [float(x) for x in v] for k, v in val_losses.items()},
            'learning_rates': [float(x) for x in learning_rates],
            'epoch_times': [float(x) for x in epoch_times],
            'total_training_time': float(total_training_time),
            'avg_epoch_time': float(np.mean(epoch_times)),
            'best_epoch': int(best_epoch) if early_stop_patience > 0 else int(np.argmin([np.mean([val_losses[snr][i] for snr in val_losses.keys()]) for i in range(len(train_losses))])),
            'best_val_loss': float(best_val_loss) if early_stop_patience > 0 else float(min([np.mean([val_losses[snr][i] for snr in val_losses.keys()]) for i in range(len(train_losses))])),
            'final_train_loss': float(train_losses[-1]),
            'initial_train_loss': float(train_losses[0])
        },

        # 测试结果
        'test_results': {}
    }

    # 转换测试结果为可JSON序列化的格式
    for snr_group, metrics in test_results.items():
        experiment_info['test_results'][snr_group] = {
            'psds1_score': float(metrics['psds1_score']),
            'psds2_score': float(metrics['psds2_score']),
            'macro_pauc': float(metrics['macro_pauc']),
            'optimal_macro_f1': float(metrics['optimal_macro_f1']),
            'segment_based_f1_micro': float(metrics['segment_based_f1_micro']),
            'segment_based_er_micro': float(metrics['segment_based_er_micro']),
            'segment_based_f1_macro': float(metrics['segment_based_f1_macro']),
            'event_based_f1': float(metrics['event_based_f1']),

            # 每个类别的详细指标
            'class_wise_metrics': {
                class_name: {
                    k: float(v) if not np.isnan(v) else None
                    for k, v in class_metrics.items()
                }
                for class_name, class_metrics in metrics['class_wise_metrics'].items()
            }
        }

    # 保存为JSON
    info_file = exp_dir / 'experiment_info.json'
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(experiment_info, f, ensure_ascii=False, indent=2)
    print(f"[OK] 实验信息已保存至: {info_file}")

    # 保存为可读文本
    results_file = exp_dir / 'results.txt'
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write(f"实验结果 - {model_type.upper()} (Seed {seed})\n")
        f.write("="*80 + "\n\n")

        f.write("模型信息:\n")
        f.write(f"  类型: {model_type}\n")
        f.write(f"  总参数: {model_info.get('total_params', 'N/A'):,}\n")
        f.write(f"  可训练参数: {model_info.get('trainable_params', 'N/A'):,}\n")
        f.write(f"  FLOPs: {flops_info['flops_readable']}\n")
        f.write(f"  参数量: {flops_info['params_readable']}\n\n")

        f.write("训练配置:\n")
        for key, value in experiment_info['training_config'].items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")

        f.write("损失函数:\n")
        for key, value in loss_info.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")

        f.write("数据集信息:\n")
        f.write(f"  训练样本: {experiment_info['dataset_info']['train_samples']}\n")
        for snr, count in experiment_info['dataset_info']['val_samples'].items():
            f.write(f"  验证样本 ({snr}): {count}\n")
        for snr, count in experiment_info['dataset_info']['test_samples'].items():
            f.write(f"  测试样本 ({snr}): {count}\n")
        f.write(f"  类别数: {experiment_info['dataset_info']['num_classes']}\n\n")

        f.write("训练历史摘要:\n")
        f.write(f"  初始训练损失: {train_losses[0]:.4f}\n")
        f.write(f"  最终训练损失: {train_losses[-1]:.4f}\n")
        f.write(f"  损失下降: {train_losses[0] - train_losses[-1]:.4f}\n")
        f.write(f"  最佳Epoch: {experiment_info['training_history']['best_epoch']+1}\n")
        f.write(f"  最佳验证损失: {experiment_info['training_history']['best_val_loss']:.4f}\n")
        f.write(f"  总训练时间: {total_training_time/3600:.2f} 小时 ({total_training_time/60:.1f} 分钟)\n")
        f.write(f"  平均Epoch时间: {np.mean(epoch_times):.2f} 秒\n\n")

        # 添加详细的epoch-by-epoch训练历史
        f.write("详细训练历史 (Epoch-by-Epoch):\n")
        f.write("-"*120 + "\n")
        f.write(f"{'Epoch':>6} | {'LR':>10} | {'Train Loss':>12} | {'Val Loss':>12} | ")
        f.write(f"{'snr_low':>10} | {'snr_medium':>12} | {'snr_high':>10} | {'Time(s)':>8}\n")
        f.write("-"*120 + "\n")

        for epoch in range(len(train_losses)):
            avg_val_loss = np.mean([val_losses[snr][epoch] for snr in val_losses.keys()])
            val_low = val_losses.get('snr_low', [0])[epoch] if epoch < len(val_losses.get('snr_low', [])) else 0
            val_medium = val_losses.get('snr_medium', [0])[epoch] if epoch < len(val_losses.get('snr_medium', [])) else 0
            val_high = val_losses.get('snr_high', [0])[epoch] if epoch < len(val_losses.get('snr_high', [])) else 0

            marker = " *" if (epoch == experiment_info['training_history']['best_epoch']) else ""

            f.write(f"{epoch+1:6d} | {learning_rates[epoch]:10.2e} | {train_losses[epoch]:12.4f} | {avg_val_loss:12.4f} | ")
            f.write(f"{val_low:10.4f} | {val_medium:12.4f} | {val_high:10.4f} | {epoch_times[epoch]:8.2f}{marker}\n")

        f.write("-"*120 + "\n")
        f.write("* 表示最佳epoch\n\n")

        f.write("测试集评估结果:\n")
        f.write("-"*80 + "\n\n")

        # 7个主要指标
        metrics_to_report = [
            'psds1_score',
            'psds2_score',
            'macro_pauc',
            'optimal_macro_f1',
            'segment_based_f1_micro',
            'segment_based_er_micro',
            'segment_based_f1_macro',
            'event_based_f1'
        ]

        for snr_group in ['snr_low', 'snr_medium', 'snr_high']:
            if snr_group in test_results:
                metrics = test_results[snr_group]
                f.write(f"{snr_group}:\n")
                for metric_name in metrics_to_report:
                    if metric_name in metrics:
                        f.write(f"  {metric_name}: {metrics[metric_name]:.4f}\n")

                # 每个类别的详细指标
                f.write(f"\n  Per-class metrics:\n")
                for class_name, class_metrics in metrics['class_wise_metrics'].items():
                    f.write(f"    {class_name}:\n")
                    f.write(f"      Segment F1:        {class_metrics.get('segment_f_measure', 0):.4f}\n")
                    f.write(f"      Segment Precision: {class_metrics.get('segment_precision', 0):.4f}\n")
                    f.write(f"      Segment Recall:    {class_metrics.get('segment_recall', 0):.4f}\n")
                    f.write(f"      Event F1:          {class_metrics.get('event_f_measure', 0):.4f}\n")
                    f.write(f"      Event Precision:   {class_metrics.get('event_precision', 0):.4f}\n")
                    f.write(f"      Event Recall:      {class_metrics.get('event_recall', 0):.4f}\n")
                    if 'error_rate' in class_metrics:
                        f.write(f"      Error Rate:        {class_metrics['error_rate']:.4f}\n")
                f.write("\n")

        # 平均指标
        f.write("\n平均测试指标（3个SNR组）:\n")
        f.write("-"*80 + "\n")
        for metric_name in metrics_to_report:
            values = [test_results[snr][metric_name] for snr in ['snr_low', 'snr_medium', 'snr_high'] if snr in test_results]
            if values:
                f.write(f"  平均 {metric_name}: {np.mean(values):.4f} ± {np.std(values):.4f}\n")
                f.write(f"    - 范围: [{np.min(values):.4f}, {np.max(values):.4f}]\n")

        # 添加实验总结
        f.write("\n" + "="*80 + "\n")
        f.write("实验总结\n")
        f.write("="*80 + "\n\n")

        # 训练收敛性分析
        loss_decrease = train_losses[0] - train_losses[-1]
        loss_decrease_pct = (loss_decrease / train_losses[0]) * 100
        f.write("训练收敛性:\n")
        f.write(f"  损失下降: {loss_decrease:.4f} ({loss_decrease_pct:.1f}%)\n")
        if loss_decrease > 0:
            f.write(f"  ✓ 训练损失正常下降\n")
        else:
            f.write(f"  ✗ 训练损失未下降，可能需要调整学习率\n")

        # 验证损失分析
        val_decreased_count = sum(
            1 for snr in val_losses.keys()
            if val_losses[snr][-1] < val_losses[snr][0]
        )
        f.write(f"\n验证损失:\n")
        for snr in ['snr_low', 'snr_medium', 'snr_high']:
            if snr in val_losses:
                val_dec = val_losses[snr][0] - val_losses[snr][-1]
                f.write(f"  {snr}: {val_losses[snr][0]:.4f} -> {val_losses[snr][-1]:.4f} ({val_dec:+.4f})\n")
        f.write(f"  {val_decreased_count}/{len(val_losses)} 组验证损失下降\n")

        # 性能评估
        avg_psds1 = np.mean([test_results[snr]['psds1_score'] for snr in ['snr_low', 'snr_medium', 'snr_high'] if snr in test_results])
        avg_f1 = np.mean([test_results[snr]['optimal_macro_f1'] for snr in ['snr_low', 'snr_medium', 'snr_high'] if snr in test_results])

        f.write(f"\n性能评估:\n")
        f.write(f"  平均 PSDS1: {avg_psds1:.4f}\n")
        f.write(f"  平均 Optimal F1: {avg_f1:.4f}\n")

        if avg_psds1 > 0.3 and avg_f1 > 0.3:
            f.write(f"  ✓ 性能良好\n")
        elif avg_psds1 > 0.1 and avg_f1 > 0.1:
            f.write(f"  ⚠ 性能中等，可能需要更多训练或调优\n")
        else:
            f.write(f"  ✗ 性能较低，建议检查模型配置和训练设置\n")

        # 模型保存说明
        f.write(f"\n模型保存:\n")
        if early_stop_patience > 0 and best_model_state is not None:
            f.write(f"  保存模型: 最佳验证损失模型 (Epoch {best_epoch+1})\n")
        else:
            f.write(f"  保存模型: 最后一个epoch的模型\n")
        f.write(f"  路径: {model_path}\n")

    print(f"[OK] 详细结果已保存至: {results_file}")

    print("\n" + "="*80)
    print(f"[OK] Seed {seed} 训练完成！")
    print("="*80)

    return {
        'experiment_info': experiment_info,
        'model_path': str(model_path),
        'exp_dir': str(exp_dir)
    }


def run_multi_seed_experiment(
    model_type,
    seeds=[42, 123, 456],
    **kwargs
):
    """运行多种子实验

    Args:
        model_type: 模型类型
        seeds: 随机种子列表
        **kwargs: 传递给train_single_seed的参数

    Returns:
        dict: 所有种子的实验结果汇总
    """
    print("\n" + "="*80)
    print(f"开始多种子实验: {model_type.upper()}")
    print(f"种子数量: {len(seeds)}")
    print(f"种子列表: {seeds}")
    print("="*80)

    all_results = []

    for seed in seeds:
        result = train_single_seed(
            model_type=model_type,
            seed=seed,
            **kwargs
        )
        all_results.append(result)

    # 汇总结果
    print("\n" + "="*80)
    print("汇总多种子实验结果")
    print("="*80)

    # 创建汇总目录
    exp_base_dir = kwargs.get('exp_base_dir', 'experiments_dcase/multi_seed')
    summary_dir = Path(exp_base_dir) / model_type / 'summary'
    summary_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有种子的测试结果
    all_test_results = {}
    for result in all_results:
        seed = result['experiment_info']['seed']
        test_results = result['experiment_info']['test_results']
        all_test_results[f'seed_{seed}'] = test_results

    # 计算统计信息
    metrics_to_report = [
        'psds1_score',
        'psds2_score',
        'macro_pauc',
        'optimal_macro_f1',
        'segment_based_f1_micro',
        'segment_based_er_micro',
        'segment_based_f1_macro',
        'event_based_f1'
    ]

    summary_stats = {}
    for snr_group in ['snr_low', 'snr_medium', 'snr_high']:
        summary_stats[snr_group] = {}

        for metric_name in metrics_to_report:
            values = []
            for seed_result in all_test_results.values():
                if snr_group in seed_result and metric_name in seed_result[snr_group]:
                    values.append(seed_result[snr_group][metric_name])

            if values:
                summary_stats[snr_group][metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'values': values
                }

    # 保存汇总信息
    summary_info = {
        'model_type': model_type,
        'seeds': seeds,
        'num_seeds': len(seeds),
        'timestamp': datetime.now().isoformat(),
        'statistics': summary_stats,
        'all_results': all_test_results
    }

    summary_file = summary_dir / 'summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_info, f, ensure_ascii=False, indent=2)

    # 保存可读文本
    summary_txt = summary_dir / 'summary.txt'
    with open(summary_txt, 'w', encoding='utf-8') as f:
        f.write(f"多种子实验汇总 - {model_type.upper()}\n")
        f.write("="*80 + "\n\n")

        f.write(f"种子数量: {len(seeds)}\n")
        f.write(f"种子列表: {seeds}\n\n")

        f.write("统计结果 (Mean ± Std):\n")
        f.write("-"*80 + "\n\n")

        for snr_group in ['snr_low', 'snr_medium', 'snr_high']:
            f.write(f"{snr_group}:\n")
            for metric_name in metrics_to_report:
                if metric_name in summary_stats[snr_group]:
                    stats = summary_stats[snr_group][metric_name]
                    f.write(f"  {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f} ")
                    f.write(f"[{stats['min']:.4f}, {stats['max']:.4f}]\n")
            f.write("\n")

        # 总平均
        f.write("\n总平均（所有SNR组）:\n")
        for metric_name in metrics_to_report:
            all_values = []
            for snr_group in ['snr_low', 'snr_medium', 'snr_high']:
                if metric_name in summary_stats[snr_group]:
                    all_values.extend(summary_stats[snr_group][metric_name]['values'])

            if all_values:
                f.write(f"  {metric_name}: {np.mean(all_values):.4f} ± {np.std(all_values):.4f}\n")

    # 保存CSV格式（便于Excel分析）
    csv_data = []
    for snr_group in ['snr_low', 'snr_medium', 'snr_high']:
        for metric_name in metrics_to_report:
            if metric_name in summary_stats[snr_group]:
                stats = summary_stats[snr_group][metric_name]
                csv_data.append({
                    'model': model_type,
                    'snr_group': snr_group,
                    'metric': metric_name,
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'min': stats['min'],
                    'max': stats['max'],
                    'num_seeds': len(seeds)
                })

    csv_df = pd.DataFrame(csv_data)
    csv_file = summary_dir / 'summary_metrics.csv'
    csv_df.to_csv(csv_file, index=False)

    print(f"\n[OK] 汇总结果已保存至: {summary_dir}")
    print(f"  - JSON: {summary_file}")
    print(f"  - TXT: {summary_txt}")
    print(f"  - CSV: {csv_file}")

    return summary_info


def main():
    parser = argparse.ArgumentParser(
        description="多种子实验运行脚本 - 融合版配置（配置3+4优化）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
融合版配置（基于配置3和配置4的最佳特性）：
  - 学习率: 2.3e-4（介于配置3的2.5e-4和配置4的2e-4之间）
  - Warmup: 8 epochs（快速warmup）
  - EMA decay: 0.995（轻量EMA提升稳定性）
  - Label smoothing: 0.005（轻正则化）
  - ReduceLROnPlateau: patience=6, factor=0.6（快速响应）
  - 梯度裁剪: 0.5
  - 早停: patience=18
        """
    )

    parser.add_argument('--model', type=str, required=True,
                       choices=['conformer_optimized', 'faf_heavy', 'daapnet', 'daapnet_ulf'],
                       help='模型类型: conformer_optimized, faf_heavy, daapnet, daapnet_ulf')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42],
                       help='随机种子列表（默认: 42, 123, 456）')
    parser.add_argument('--epochs', type=int, default=80,
                       help='训练轮数（默认: 80）')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批次大小（默认: 64）')
    parser.add_argument('--lr', type=float, default=2.3e-4,
                       help='学习率（默认: 2.3e-4 融合版）')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='权重衰减（默认: 0.01）')
    parser.add_argument('--scheduler', type=str, default='plateau',
                       choices=['plateau', 'cosine', 'onecycle'],
                       help='学习率调度器（默认: plateau）')
    parser.add_argument('--warmup', action='store_true', default=True,
                       help='使用warmup（默认: True）')
    parser.add_argument('--warmup_epochs', type=int, default=8,
                       help='Warmup轮数（默认: 8）')
    parser.add_argument('--ema_decay', type=float, default=0.995,
                       help='EMA衰减率（默认: 0.995，0表示禁用）')
    parser.add_argument('--label_smoothing', type=float, default=0.005,
                       help='Label smoothing强度（默认: 0.005）')
    parser.add_argument('--gradient_clip', type=float, default=0.5,
                       help='梯度裁剪阈值（默认: 0.5）')
    parser.add_argument('--early_stop', type=int, default=10,
                       help='早停容忍度（默认: 18）')
    parser.add_argument('--scheduler_patience', type=int, default=6,
                       help='ReduceLROnPlateau patience（默认: 6）')
    parser.add_argument('--scheduler_factor', type=float, default=0.6,
                       help='ReduceLROnPlateau factor（默认: 0.6）')
    parser.add_argument('--exp_base_dir', type=str,
                       default='experiments_dcase_new/multi_seed_optimized',
                       help='实验基础目录（默认: multi_seed_optimized）')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU编号（默认: 0）')

    args = parser.parse_args()

    # 设置GPU
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # 模型配置（针对FAF_Heavy）
    model_config = None
    if args.model == 'faf_heavy':
        model_config = {
            'use_faf_filt': True,
            'faf_config': 'heavy',
            'use_projection': True,
            'projection_method': 'conv1d'
        }

    # 处理EMA禁用（0表示None）
    ema_decay = args.ema_decay if args.ema_decay > 0 else None

    print("\n" + "="*80)
    print("融合版配置参数:")
    print("="*80)
    print(f"  学习率: {args.lr}")
    print(f"  Warmup epochs: {args.warmup_epochs}")
    print(f"  EMA decay: {ema_decay if ema_decay else '禁用'}")
    print(f"  Label smoothing: {args.label_smoothing}")
    print(f"  Scheduler: {args.scheduler} (patience={args.scheduler_patience}, factor={args.scheduler_factor})")
    print(f"  梯度裁剪: {args.gradient_clip}")
    print(f"  早停patience: {args.early_stop}")
    print("="*80)

    # 运行多种子实验
    summary = run_multi_seed_experiment(
        model_type=args.model,
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler_type=args.scheduler,
        use_warmup=args.warmup,
        warmup_epochs=args.warmup_epochs,
        ema_decay=ema_decay,
        label_smoothing=args.label_smoothing,
        gradient_clip=args.gradient_clip,
        early_stop_patience=args.early_stop,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        model_config=model_config,
        exp_base_dir=args.exp_base_dir
    )

    print("\n" + "="*80)
    print("所有实验完成！")
    print("="*80)


if __name__ == "__main__":
    main()
