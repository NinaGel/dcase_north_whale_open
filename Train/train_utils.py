import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time
import logging
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import json
from pathlib import Path
import inspect  # 用于双输入模型检测
import random
import config as cfg

# 配置日志
logger = logging.getLogger(__name__)


def apply_batch_mixup(features, labels, mix_cfg):
    """批次级Mixup增强 / Batch-level Mixup augmentation

    Args:
        features: 特征张量 / feature tensor [B, C, F, T]
        labels: 标签张量 / label tensor [B, T, num_classes]
        mix_cfg: 配置字典 / config dict {enabled, prob, alpha}

    Returns:
        混合后的特征和标签 / mixed features and labels
    """
    if not mix_cfg or not mix_cfg.get('enabled', False):
        return features, labels

    prob = mix_cfg.get('prob', 0.0)
    alpha = mix_cfg.get('alpha', 0.0)

    if features.size(0) < 2 or random.random() > prob or alpha <= 0:
        return features, labels

    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(features.size(0), device=features.device)

    mixed_features = lam * features + (1 - lam) * features[index]
    mixed_labels = lam * labels + (1 - lam) * labels[index]
    return mixed_features, mixed_labels

class SNRBalancer:
    """SNR组权重平衡器 / SNR Group Weight Balancer

    根据不同SNR组的性能动态调整损失权重
    Dynamically adjusts loss weights based on performance across SNR groups
    """

    def __init__(self):
        # SNR范围定义 / SNR range definitions
        self.snr_ranges = {
            'very_low': (-10, -5),  # 极低SNR：-10 ~ -5 dB
            'low': (-5, 0),         # 低SNR：-5 ~ 0 dB
            'medium': (0, 5),       # 中SNR：0 ~ 5 dB
            'high': (5, 10)         # 高SNR：5 ~ 10 dB
        }

        # SNR组名映射
        self.group_mapping = {
            'snr_very_low': 'very_low',
            'snr_low': 'low',
            'snr_medium': 'medium',
            'snr_high': 'high',
            'very_low': 'very_low',
            'low': 'low',
            'medium': 'medium',
            'high': 'high'
        }

        # 初始化权重
        self.group_weights = {
            'very_low': 2.0,  # 极低SNR最高权重
            'low': 1.5,       # 低SNR较高权重
            'medium': 1.0,    # 中等SNR标准权重
            'high': 0.8       # 高SNR较低权重
        }

        # 初始化性能历史记录
        self.performance_history = {
            'very_low': [],
            'low': [],
            'medium': [],
            'high': []
        }

    def _normalize_group_name(self, group):
        """标准化组名
        
        Args:
            group: 原始组名
            
        Returns:
            str: 标准化后的组名
        """
        return self.group_mapping.get(group, 'medium')

    def get_snr_group(self, snr):
        """根据SNR值确定所属组"""
        for group, (min_snr, max_snr) in self.snr_ranges.items():
            if min_snr <= snr < max_snr:
                return group
        return 'medium'

    def get_group_weight(self, snr):
        """获取SNR对应的组权重"""
        group = self.get_snr_group(snr)
        return self.group_weights.get(group, 1.0)

    def update_weights(self, group_performances):
        """动态更新权重"""
        if not group_performances:
            return

        # 标准化组名并收集性能值
        normalized_performances = {}
        for group, perf in group_performances.items():
            norm_group = self._normalize_group_name(group)
            if isinstance(perf, (list, np.ndarray)):
                normalized_performances[norm_group] = float(np.mean(perf))
            else:
                normalized_performances[norm_group] = float(perf)

        # 计算平均性能
        mean_perf = np.mean(list(normalized_performances.values()))
        if mean_perf == 0:
            return

        # 更新权重
        for group in self.group_weights.keys():
            if group in normalized_performances:
                perf_ratio = normalized_performances[group] / mean_perf
                self.group_weights[group] = 1.0 + 2.0 / (1.0 + np.exp(-2 * (1 - perf_ratio)))

        # 归一化权重
        total_weight = sum(self.group_weights.values())
        if total_weight > 0:
            for group in self.group_weights:
                self.group_weights[group] /= total_weight

    def calculate_weighted_loss(self, losses, snrs):
        """计算加权损失，如果snrs不是数值，则返回平均损失"""
        if not torch.is_tensor(snrs) or snrs.dtype == torch.string:
            return losses.mean()
        batch_weights = torch.tensor([self.get_group_weight(snr) for snr in snrs],
                                     device=losses.device)
        weighted_losses = losses * batch_weights.view(-1, 1)
        return weighted_losses.mean()

    def record_performance(self, group_metrics):
        """记录各组性能"""
        for group, metrics in group_metrics.items():
            norm_group = self._normalize_group_name(group)
            if isinstance(metrics, dict):
                value = metrics.get('mean_loss', 0.0)
            else:
                value = float(metrics)
            self.performance_history[norm_group].append(value)

    def get_performance_stats(self):
        """获取性能统计"""
        stats = {}
        for group, history in self.performance_history.items():
            if history:
                stats[group] = {
                    'mean_loss': np.mean(history),
                    'std_loss': np.std(history),
                    'min_loss': np.min(history),
                    'max_loss': np.max(history),
                    'current_weight': self.group_weights[group]
                }
        return stats


class BaseTrainer:
    """基础训练器类"""

    def __init__(self, model, optimizer, loss_fn, device, snr_balancer=None, scaler=None, config=None):
        """初始化训练器

        Args:
            model: 模型
            optimizer: 优化器
            loss_fn: 损失函数
            device: 设备
            snr_balancer: SNR平衡器（可选）
            scaler: 混合精度训练的梯度缩放器（可选）
            config: 训练配置字典（可选），如果不提供则使用默认的 cfg.TRAIN_CONFIG
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.snr_balancer = snr_balancer or SNRBalancer()

        # 存储配置，如果没有提供则使用默认配置
        self.config = config if config is not None else cfg.TRAIN_CONFIG

        # 如果提供了scaler就使用提供的,否则根据配置创建新的
        self.scaler = scaler if scaler is not None else GradScaler(
            enabled=self.config['mixed_precision']
        )

        self.metric_tracker = MetricTracker()

        # 初始化学习率调度器
        self.scheduler = self._initialize_scheduler()

        # 训练状态
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.best_model_state = None
        self.training_state = {}

        # 双输入模型检测（运行时检测一次并缓存）
        self._is_dual_input = self._detect_dual_input_model()

    def _detect_dual_input_model(self):
        """
        检测模型是否为双输入模型

        通过检查 forward 方法的签名来判断：
        - 单输入模型: forward(self, x)
        - 双输入模型: forward(self, x_act, x_beats) 或 forward(self, x1, x2)

        Returns:
            bool: True 如果模型需要两个输入参数
        """
        try:
            sig = inspect.signature(self.model.forward)
            params = list(sig.parameters.keys())

            # 排除 'self' 参数，检查剩余位置参数数量
            # 双输入模型至少有 2 个参数（x_act, x_beats）
            # 单输入模型只有 1 个参数（x）
            num_params = len([p for p in params if p != 'self'])
            return num_params >= 2
        except Exception as e:
            logging.warning(f"无法检测模型输入类型，默认为单输入模型: {e}")
            return False

    def _unpack_batch_dual(self, batch):
        """
        解包双输入批次数据

        Args:
            batch: ((feat_act, feat_beats), labels, filenames)

        Returns:
            feat_act, feat_beats, labels, snrs (snrs 可能为 None)
        """
        if len(batch) >= 3:
            (feat_act, feat_beats), labels, filenames = batch[:3]
            snrs = batch[3] if len(batch) > 3 else None
        elif len(batch) == 2:
            (feat_act, feat_beats), labels = batch
            snrs = None
        else:
            raise ValueError(f"双输入批次格式错误: 期望 ((feat_act, feat_beats), labels, ...), 得到长度 {len(batch)}")

        return feat_act, feat_beats, labels, snrs

    def _unpack_batch_single(self, batch):
        """
        解包单输入批次数据

        Args:
            batch: (features, labels, filenames) 或 (features, labels)

        Returns:
            features, labels, snrs (snrs 可能为 None)
        """
        if isinstance(batch, (tuple, list)):
            if len(batch) == 3:
                x, y, snrs_or_filenames = batch
                # 判断第三个元素是 SNR 值还是文件名
                if isinstance(snrs_or_filenames, (list, str, np.ndarray)):
                    snrs = None  # 文件名
                else:
                    snrs = snrs_or_filenames
            elif len(batch) == 2:
                x, y = batch
                snrs = None
            else:
                raise ValueError(f"批次格式错误: 期望 2-3 个元素, 得到 {len(batch)}")
        else:
            if hasattr(batch, 'x') and hasattr(batch, 'y'):
                x, y = batch.x, batch.y
                snrs = batch.snrs if hasattr(batch, 'snrs') else None
            else:
                raise ValueError("批次格式无法识别")

        return x, y, snrs

    def _initialize_scheduler(self):
        """初始化学习率调度器"""
        scheduler_config = self.config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'cosine')

        if scheduler_type == 'cosine':
            # 普通余弦退火
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['epochs'],
                eta_min=scheduler_config.get('eta_min', 1e-6)
            )
        elif scheduler_type == 'cosine_warm_restarts':
            # 带热重启的余弦退火
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=scheduler_config.get('T_0', 10),  # 第一次重启的周期
                T_mult=scheduler_config.get('T_mult', 2),  # 每次重启后周期的倍数
                eta_min=scheduler_config.get('eta_min', 1e-6)
            )
        else:
            return None
    
    def save_checkpoint(self, epoch, save_path):
        """保存检查点
        
        Args:
            epoch: 当前轮次
            save_path: 保存路径
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict(),
            'best_loss': self.best_loss,
            'best_model_state': self.best_model_state,
            'training_state': self.training_state,
            'snr_balancer_state': {
                'group_weights': self.snr_balancer.group_weights,
                'performance_history': self.snr_balancer.performance_history
            }
        }
        torch.save(checkpoint, save_path)
        logging.info(f"检查点已保存到: {save_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点
        
        Args:
            checkpoint_path: 检查点路径
        """
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 恢复模型和优化器状态
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 恢复学习率调度器状态
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 恢复其他训练状态
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.best_model_state = checkpoint['best_model_state']
        self.training_state = checkpoint['training_state']
        
        # 恢复SNR平衡器状态
        snr_state = checkpoint['snr_balancer_state']
        self.snr_balancer.group_weights = snr_state['group_weights']
        self.snr_balancer.performance_history = snr_state['performance_history']
        
        logging.info(f"从轮次 {self.current_epoch} 恢复训练")
        
    def train_epoch(self, train_loader):
        """训练一个epoch（支持单/双输入模型）"""
        self.model.train()
        total_loss = 0
        batch_times = []

        grad_clip = self.config.get('grad_clip', False)
        grad_clip_value = self.config.get('grad_clip_value', 1.0)

        current_lr = self.optimizer.param_groups[0]['lr']
        with tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1} (lr={current_lr:.2e})") as pbar:
            for i, batch in enumerate(pbar):
                batch_start = time.time()

                # 根据模型类型解包批次
                if self._is_dual_input:
                    # 双输入模型: ((feat_act, feat_beats), labels, ...)
                    feat_act, feat_beats, y, snrs = self._unpack_batch_dual(batch)
                    feat_act = feat_act.to(self.device, non_blocking=True)
                    feat_beats = feat_beats.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)

                    # 混合精度前向传播（双输入）
                    with autocast(enabled=self.scaler.is_enabled()):
                        y_pred = self.model(feat_act, feat_beats)
                        loss = self.loss_fn(y_pred, y)
                else:
                    # 单输入模型: (features, labels, ...)
                    x, y, snrs = self._unpack_batch_single(batch)
                    x = x.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)

                    # Batch级 Mixup（使用配置中的mixing.mixup）
                    mix_cfg = cfg.AUGMENTATION_CONFIG.get('mixing', {}).get('mixup', {})
                    x, y = apply_batch_mixup(x, y, mix_cfg)

                    # 混合精度前向传播（单输入）
                    with autocast(enabled=self.scaler.is_enabled()):
                        y_pred = self.model(x)
                        loss = self.loss_fn(y_pred, y)

                # 使用梯度缩放器进行反向传播
                self.scaler.scale(loss).backward()

                if grad_clip:
                    # 在更新权重之前取消缩放梯度
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = clip_grad_norm_(self.model.parameters(), grad_clip_value)
                    self.metric_tracker.update('grad_norms', grad_norm.item())

                # 使用梯度缩放器更新权重
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零

                total_loss += loss.item()
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)
                
                # 更新进度条信息
                pbar.set_postfix({
                    'train_loss': f'{loss.item():.3f}',
                    'time': f'{batch_time:.2f}s'
                })

        return total_loss / len(train_loader), np.mean(batch_times)

    def validate(self, val_loader):
        """验证模型性能（支持单/双输入模型）

        Args:
            val_loader: 可以是单个DataLoader或包含多个SNR组的字典

        Returns:
            tuple: (平均验证损失, SNR组损失字典)
        """
        self.model.eval()
        snr_group_metrics = {}

        # 如果val_loader是字典，则需要遍历所有SNR组
        if isinstance(val_loader, dict):
            total_loss = 0
            group_count = 0

            for snr_group, loader in val_loader.items():
                group_losses = []

                with torch.no_grad():
                    with autocast(enabled=self.scaler.is_enabled()):
                        for batch in loader:
                            # 根据模型类型解包批次
                            if self._is_dual_input:
                                # 双输入模型
                                feat_act, feat_beats, y, snrs = self._unpack_batch_dual(batch)
                                feat_act = feat_act.to(self.device, non_blocking=True)
                                feat_beats = feat_beats.to(self.device, non_blocking=True)
                                y = y.to(self.device, non_blocking=True)

                                y_pred = self.model(feat_act, feat_beats)
                            else:
                                # 单输入模型
                                x, y, snrs = self._unpack_batch_single(batch)

                                # 确保输入维度正确
                                if x.dim() == 3:  # [batch_size, freq, time]
                                    x = x.unsqueeze(1)  # 添加通道维度

                                x = x.to(self.device, non_blocking=True)
                                y = y.to(self.device, non_blocking=True)

                                y_pred = self.model(x)

                            loss = self.loss_fn(y_pred, y)
                            group_losses.append(loss.item())

                if group_losses:
                    avg_group_loss = sum(group_losses) / len(group_losses)
                    snr_group_metrics[snr_group] = avg_group_loss
                    total_loss += avg_group_loss
                    group_count += 1

            # 计算所有组的平均损失
            avg_loss = total_loss / group_count if group_count > 0 else float('inf')

        else:
            # 处理单个验证加载器的情况
            losses = []
            with torch.no_grad():
                with autocast(enabled=self.scaler.is_enabled()):
                    for batch in val_loader:
                        # 根据模型类型解包批次
                        if self._is_dual_input:
                            # 双输入模型
                            feat_act, feat_beats, y, snrs = self._unpack_batch_dual(batch)
                            feat_act = feat_act.to(self.device, non_blocking=True)
                            feat_beats = feat_beats.to(self.device, non_blocking=True)
                            y = y.to(self.device, non_blocking=True)

                            y_pred = self.model(feat_act, feat_beats)
                        else:
                            # 单输入模型
                            x, y, snrs = self._unpack_batch_single(batch)

                            # 确保输入维度正确
                            if x.dim() == 3:  # [batch_size, freq, time]
                                x = x.unsqueeze(1)  # 添加通道维度

                            x = x.to(self.device, non_blocking=True)
                            y = y.to(self.device, non_blocking=True)

                            y_pred = self.model(x)

                        loss = self.loss_fn(y_pred, y)
                        losses.append(loss.item())

            avg_loss = sum(losses) / len(losses) if losses else float('inf')
            snr_group_metrics = {'default': avg_loss}

        return avg_loss, snr_group_metrics

    def train_model(self, train_loader, val_loader, epochs, model_name="model", 
                   checkpoint_dir=None, resume_from=None):
        """完整的模型训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器（可以是单个加载器或SNR组字典）
            epochs: 训练轮数
            model_name: 模型名称
            checkpoint_dir: 检查点保存目录
            resume_from: 恢复训练的检查点路径
            
        Returns:
            dict: 训练结果
        """
        # 设置检查点目录
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 如果指定了恢复训练的检查点，则加载它
        if resume_from:
            self.load_checkpoint(resume_from)
            start_epoch = self.current_epoch
        else:
            start_epoch = 0
        
        train_losses = self.training_state.get('train_losses', [])
        val_losses = self.training_state.get('val_losses', {})
        patience_counter = self.training_state.get('patience_counter', 0)
        best_loss = float('inf')
        best_model_state = None
        
        try:
            epoch_start_time = time.time()
            for epoch in range(start_epoch, epochs):
                self.current_epoch = epoch
                
                # 训练阶段
                train_loss, batch_time = self.train_epoch(train_loader)
                train_losses.append(train_loss)
                
                # 验证阶段
                val_loss, group_metrics = self.validate(val_loader)
                
                # 更新验证损失历史
                for group, loss in group_metrics.items():
                    if group not in val_losses:
                        val_losses[group] = []
                    val_losses[group].append(loss)
                
                # 更新SNR权重
                self.snr_balancer.update_weights(group_metrics)
                self.snr_balancer.record_performance(group_metrics)
                
                # 更新最佳模型
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
                    patience_counter = 0
                    best_epoch = epoch
                    
                    # 保存最佳模型检查点
                    if checkpoint_dir:
                        self.save_checkpoint(
                            epoch,
                            checkpoint_dir / f'{model_name}_best.pth'
                        )
                else:
                    patience_counter += 1
                
                # 定期保存检查点
                checkpoint_freq = self.config.get('checkpoint_frequency', 10)
                if checkpoint_dir and (epoch + 1) % checkpoint_freq == 0:
                    self.save_checkpoint(
                        epoch,
                        checkpoint_dir / f'{model_name}_epoch_{epoch + 1}.pth'
                    )
                
                # 更新训练状态
                self.training_state.update({
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'patience_counter': patience_counter,
                    'last_epoch': epoch
                })
                
                # 打印训练信息
                print(f"Epoch {epoch + 1}/{epochs}")
                print(f"Train Loss: {train_loss:.4f}")
                print("验证损失:")
                # 计算所有SNR组的平均验证损失
                total_val_loss = 0
                valid_group_count = 0
                for group, loss in group_metrics.items():
                    print(f"- {group}: {loss:.4f}")
                    if group in ['snr_very_low', 'snr_low', 'snr_medium', 'snr_high']:
                        total_val_loss += loss
                        valid_group_count += 1
                
                if valid_group_count > 0:
                    avg_val_loss = total_val_loss / valid_group_count
                    print(f"平均验证损失: {avg_val_loss:.4f}")
                
                print(f"时间: {time.time() - epoch_start_time:.2f}s")
                
                # 早停检查
                if patience_counter >= self.config['patience']:
                    print(f"\n早停触发于轮次 {epoch + 1}")
                    break
                
                # 更新学习率
                if self.scheduler:
                    self.scheduler.step()
                
                # 更新epoch开始时间
                epoch_start_time = time.time()
            
        except KeyboardInterrupt:
            print("\n检测到训练中断，保存检查点...")
            if checkpoint_dir:
                self.save_checkpoint(
                    epoch,
                    checkpoint_dir / f'{model_name}_interrupted.pth'
                )
        
        # 计算SNR组性能统计
        snr_metrics = {}
        for group, losses in val_losses.items():
            snr_metrics[group] = {
                'mean_loss': sum(losses) / len(losses),
                'best_loss': min(losses),
                'current_weight': self.snr_balancer.group_weights.get(group, 1.0)
            }
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_epoch': best_epoch,
            'best_loss': best_loss,
            'best_model_state': best_model_state,
            'last_model_state': {
                k: v.cpu().clone() for k, v in self.model.state_dict().items()
            },
            'snr_metrics': snr_metrics,
            'start_time': epoch_start_time
        }


class MetricTracker:
    """训练指标跟踪器"""

    def __init__(self):
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'grad_norms': [],
            'batch_times': [],
            'epoch_times': []
        }

    def update(self, metric_name, value):
        """更新指标"""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)

    def get_latest(self, metric_name):
        """获取最新值"""
        return self.metrics[metric_name][-1] if self.metrics[metric_name] else None

    def get_mean(self, metric_name, window_size=None):
        """计算平均值"""
        values = self.metrics[metric_name]
        if not values:
            return None
        if window_size:
            values = values[-window_size:]
        return np.mean(values)


def plot_losses(train_losses, val_losses, epochs, save_path):
    """绘制损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'bo-', label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}_losses.png")
    plt.close()