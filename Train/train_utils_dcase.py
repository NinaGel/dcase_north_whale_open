import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time
import logging
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import json
from pathlib import Path
import config_dcase as cfg

# 配置日志
logger = logging.getLogger(__name__)

class DCASEEventBalancer:
    """DCASE事件权重平衡器 - 核心功能类"""

    def __init__(self):
        """初始化DCASE事件平衡器"""
        # DCASE事件类别定义
        self.event_classes = {
            'Speech': 0,
            'Dishes': 1, 
            'Frying': 2,
            'Electric_shaver_toothbrush': 3,
            'Vacuum_cleaner': 4,
            'Running_water': 5,
            'Cat': 6,
            'Dog': 7,
            'Alarm_bell_ringing': 8,
            'Blender': 9
        }

        # 事件难度权重（基于经验调整）
        self.class_weights = {
            'Speech': 1.0,                        # 中等难度
            'Dishes': 1.2,                        # 较难区分
            'Frying': 1.3,                        # 背景噪声相似
            'Electric_shaver_toothbrush': 1.5,    # 高频，易混淆
            'Vacuum_cleaner': 1.1,                # 相对容易
            'Running_water': 1.2,                 # 连续性强
            'Cat': 1.4,                           # 短暂，易漏检
            'Dog': 1.3,                           # 变化大
            'Alarm_bell_ringing': 1.0,            # 特征明显
            'Blender': 1.1                        # 相对容易
        }

        # 初始化性能历史记录
        self.performance_history = {
            class_name: [] for class_name in self.event_classes.keys()
        }

    def get_class_weight(self, class_name):
        """获取事件类别对应的权重"""
        return self.class_weights.get(class_name, 1.0)

    def update_weights(self, class_performances):
        """动态更新权重"""
        if not class_performances:
            return

        # 收集性能值
        normalized_performances = {}
        for class_name, perf in class_performances.items():
            if isinstance(perf, dict):
                # 如果是字典，提取F1分数作为性能指标
                normalized_performances[class_name] = float(perf.get('f1_score', 0.0))
            elif isinstance(perf, (list, np.ndarray)):
                normalized_performances[class_name] = float(np.mean(perf))
            else:
                normalized_performances[class_name] = float(perf)

        # 计算平均性能
        mean_perf = np.mean(list(normalized_performances.values()))
        if mean_perf == 0:
            return

        # 更新权重：性能差的类别权重更高
        for class_name in self.class_weights.keys():
            if class_name in normalized_performances:
                perf_ratio = normalized_performances[class_name] / mean_perf
                # 使用sigmoid函数调整权重
                self.class_weights[class_name] = 1.0 + 2.0 / (1.0 + np.exp(-2 * (1 - perf_ratio)))

        # 归一化权重
        total_weight = sum(self.class_weights.values())
        if total_weight > 0:
            for class_name in self.class_weights:
                self.class_weights[class_name] /= total_weight

    def calculate_weighted_loss(self, losses, labels):
        """计算加权损失
        
        Args:
            losses: [batch_size, num_classes] 每个样本每个类别的损失
            labels: [batch_size, num_classes] 全局标签，或者
                    [batch_size, frames, num_classes] 帧级标签
        
        Returns:
            标量损失值
        """
        if not torch.is_tensor(labels):
            return losses.mean()
        
        # 如果是帧级标签 [batch_size, frames, num_classes]，转换为全局标签
        if labels.dim() == 3:
            # 如果任意帧中存在某个类别，则全局标签为1
            labels = (labels.sum(dim=1) > 0).float()  # [batch_size, num_classes]
        
        # 现在labels形状是 [batch_size, num_classes]
        # losses形状是 [batch_size, num_classes]
        
        # 对于多标签分类，计算每个样本的平均权重
        batch_weights = torch.ones(losses.shape[0], device=losses.device)
        
        for i in range(labels.shape[0]):
            sample_weights = []
            for j, class_name in enumerate(self.event_classes.keys()):
                # 现在labels[i, j]是一个标量张量
                if labels[i, j].item() > 0.5:  # 如果该类别存在
                    sample_weights.append(self.class_weights[class_name])
            
            if sample_weights:
                batch_weights[i] = np.mean(sample_weights)
        
        # losses的形状是 [batch_size, num_classes]
        # 应用样本级权重到所有类别
        weighted_losses = losses * batch_weights.view(-1, 1)
        return weighted_losses.mean()

    def record_performance(self, class_metrics):
        """记录各类别性能"""
        for class_name, metrics in class_metrics.items():
            if class_name in self.performance_history:
                if isinstance(metrics, dict):
                    value = metrics.get('f1_score', metrics.get('mean_loss', 0.0))
                else:
                    value = float(metrics)
                self.performance_history[class_name].append(value)

    def get_performance_stats(self):
        """获取性能统计"""
        stats = {}
        for class_name, history in self.performance_history.items():
            if history:
                stats[class_name] = {
                    'mean_performance': np.mean(history),
                    'std_performance': np.std(history),
                    'min_performance': np.min(history),
                    'max_performance': np.max(history),
                    'current_weight': self.class_weights[class_name]
                }
        return stats


class DCASETrainer:
    """DCASE专用训练器类"""

    def __init__(self, model, optimizer, loss_fn, device, event_balancer=None, scaler=None, data_adapter=None):
        """初始化DCASE训练器
        
        Args:
            model: 模型
            optimizer: 优化器
            loss_fn: 损失函数
            device: 设备
            event_balancer: 事件平衡器（可选）
            scaler: 混合精度训练的梯度缩放器（可选）
            data_adapter: 数据适配器（可选），用于不同模型的数据格式转换
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.event_balancer = event_balancer or DCASEEventBalancer()
        self.data_adapter = data_adapter  # 添加数据适配器
        
        # 如果提供了scaler就使用提供的，否则根据配置创建新的
        self.scaler = scaler if scaler is not None else torch.amp.GradScaler(
            'cuda',
            enabled=cfg.DCASE_TRAIN_CONFIG['mixed_precision']
        )
        
        self.metric_tracker = MetricTracker()
        
        # 初始化学习率调度器
        self.scheduler = self._initialize_scheduler()
        
        # 训练状态
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.best_model_state = None
        self.training_state = {}
    
    def _initialize_scheduler(self):
        """初始化学习率调度器"""
        scheduler_config = cfg.DCASE_TRAIN_CONFIG.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'cosine')
        
        if scheduler_type == 'cosine':
            # 普通余弦退火
            return CosineAnnealingLR(
                self.optimizer,
                T_max=cfg.DCASE_TRAIN_CONFIG['epochs'],
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
            'event_balancer_state': {
                'class_weights': self.event_balancer.class_weights,
                'performance_history': self.event_balancer.performance_history
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
        
        # 恢复事件平衡器状态
        balancer_state = checkpoint['event_balancer_state']
        self.event_balancer.class_weights = balancer_state['class_weights']
        self.event_balancer.performance_history = balancer_state['performance_history']
        
        logging.info(f"从轮次 {self.current_epoch} 恢复训练")
        
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        batch_times = []

        grad_clip = cfg.DCASE_TRAIN_CONFIG.get('grad_clip', False)
        grad_clip_value = cfg.DCASE_TRAIN_CONFIG.get('grad_clip_value', 1.0)

        current_lr = self.optimizer.param_groups[0]['lr']
        with tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1} (lr={current_lr:.2e})") as pbar:
            for i, batch in enumerate(pbar):
                batch_start = time.time()

                # 处理批次数据 - DCASE格式
                if isinstance(batch, (tuple, list)):
                    if len(batch) == 3:
                        x, y, metadata = batch  # metadata可能包含文件名、事件时间等
                    elif len(batch) == 2:
                        x, y = batch
                        metadata = None
                    else:
                        raise ValueError(f"Unexpected batch size: {len(batch)}")
                else:
                    if hasattr(batch, 'x') and hasattr(batch, 'y'):
                        x, y = batch.x, batch.y
                        metadata = batch.metadata if hasattr(batch, 'metadata') else None
                    else:
                        raise ValueError("Batch format not recognized")

                # 如果提供了数据适配器，使用它转换数据格式
                if self.data_adapter is not None:
                    x, y, _ = self.data_adapter.convert_batch((x, y, metadata))

                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                # 使用混合精度训练 - 使用新版PyTorch API
                with torch.amp.autocast('cuda', enabled=self.scaler.is_enabled()):
                    y_pred = self.model(x)
                    
                    # 处理帧级标签：转换为全局标签用于损失计算
                    # y可能是 [batch_size, num_classes] 或 [batch_size, frames, num_classes]
                    if y.dim() == 3:
                        # 帧级标签：如果任意帧存在该类别，则全局标签为1
                        y_global = (y.sum(dim=1) > 0).float()
                    else:
                        y_global = y
                    
                    # 将帧级预测聚合成全局预测
                    # y_pred: [batch, time_steps, num_classes]
                    # 使用max pooling获取每个类别在所有时间步的最大激活
                    if y_pred.dim() == 3:
                        y_pred_global = y_pred.max(dim=1)[0]  # [batch, num_classes]
                    else:
                        y_pred_global = y_pred
                    
                    # loss_fn返回 [batch_size, num_classes] 的损失
                    loss_per_sample = self.loss_fn(y_pred_global, y_global)
                    
                    # 使用事件权重平衡损失
                    if hasattr(self.event_balancer, 'calculate_weighted_loss'):
                        loss = self.event_balancer.calculate_weighted_loss(loss_per_sample, y_global)
                    else:
                        # 如果没有事件平衡器，直接取平均
                        loss = loss_per_sample.mean()

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

    def validate(self, val_loaders):
        """验证模型性能
        
        Args:
            val_loaders: 验证数据加载器（可以是单个loader或loader字典）
            
        Returns:
            tuple: (平均验证损失, 事件类别损失字典)
        """
        self.model.eval()
        event_class_metrics = {}
        
        # 初始化各类别的损失记录
        for class_name in self.event_balancer.event_classes.keys():
            event_class_metrics[class_name] = []

        # 处理字典形式的loaders（多个SNR组）
        if isinstance(val_loaders, dict):
            all_losses = []
            all_predictions = []
            all_targets = []
            
            # 遍历每个SNR组的loader
            for snr_group, val_loader in val_loaders.items():
                with torch.no_grad():
                    with torch.amp.autocast('cuda', enabled=self.scaler.is_enabled()):
                        for batch in val_loader:
                            # 处理批次数据
                            if isinstance(batch, (tuple, list)):
                                if len(batch) >= 2:
                                    x, y = batch[:2]
                                    metadata = batch[2] if len(batch) > 2 else None
                                else:
                                    raise ValueError(f"Batch size too small: {len(batch)}")
                            else:
                                if hasattr(batch, 'x') and hasattr(batch, 'y'):
                                    x, y = batch.x, batch.y
                                    metadata = batch.metadata if hasattr(batch, 'metadata') else None
                                else:
                                    raise ValueError("Unexpected batch format")

                            # 如果提供了数据适配器，使用它转换数据格式
                            if self.data_adapter is not None:
                                x, y, _ = self.data_adapter.convert_batch((x, y, metadata))
                            else:
                                # 默认行为：确保输入维度正确
                                if x.dim() == 3:  # [batch_size, freq, time]
                                    x = x.unsqueeze(1)  # 添加通道维度 [batch_size, channel, freq, time]

                            x = x.to(self.device, non_blocking=True)
                            y = y.to(self.device, non_blocking=True)

                            y_pred = self.model(x)
                            
                            # 处理帧级标签：转换为全局标签用于损失计算
                            if y.dim() == 3:
                                y_global = (y.sum(dim=1) > 0).float()
                            else:
                                y_global = y
                            
                            # 将帧级预测聚合成全局预测
                            if y_pred.dim() == 3:
                                y_pred_global = y_pred.max(dim=1)[0]  # [batch, num_classes]
                            else:
                                y_pred_global = y_pred
                            
                            # loss_fn返回 [batch_size, num_classes] 的损失
                            loss_per_sample = self.loss_fn(y_pred_global, y_global)
                            
                            # 计算平均损失
                            loss = loss_per_sample.mean()
                            all_losses.append(loss.item())
                            all_predictions.append(y_pred_global.cpu())
                            all_targets.append(y_global.cpu())
        else:
            # 处理单个loader的情况
            all_losses = []
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=self.scaler.is_enabled()):
                    for batch in val_loaders:
                        # 处理批次数据
                        if isinstance(batch, (tuple, list)):
                            if len(batch) >= 2:
                                x, y = batch[:2]
                                metadata = batch[2] if len(batch) > 2 else None
                            else:
                                raise ValueError(f"Batch size too small: {len(batch)}")
                        else:
                            if hasattr(batch, 'x') and hasattr(batch, 'y'):
                                x, y = batch.x, batch.y
                                metadata = batch.metadata if hasattr(batch, 'metadata') else None
                            else:
                                raise ValueError("Unexpected batch format")

                        # 如果提供了数据适配器，使用它转换数据格式
                        if self.data_adapter is not None:
                            x, y, _ = self.data_adapter.convert_batch((x, y, metadata))
                        else:
                            # 默认行为：确保输入维度正确
                            if x.dim() == 3:  # [batch_size, freq, time]
                                x = x.unsqueeze(1)  # 添加通道维度 [batch_size, channel, freq, time]

                        x = x.to(self.device, non_blocking=True)
                        y = y.to(self.device, non_blocking=True)

                        y_pred = self.model(x)
                        
                        # 处理帧级标签：转换为全局标签用于损失计算
                        if y.dim() == 3:
                            y_global = (y.sum(dim=1) > 0).float()
                        else:
                            y_global = y
                        
                        # 将帧级预测聚合成全局预测
                        if y_pred.dim() == 3:
                            y_pred_global = y_pred.max(dim=1)[0]  # [batch, num_classes]
                        else:
                            y_pred_global = y_pred
                        
                        # loss_fn返回 [batch_size, num_classes] 的损失
                        loss_per_sample = self.loss_fn(y_pred_global, y_global)
                        
                        # 计算平均损失
                        loss = loss_per_sample.mean()
                        all_losses.append(loss.item())
                        all_predictions.append(y_pred_global.cpu())
                        all_targets.append(y_global.cpu())
        
        # 计算各类别的性能指标
        if all_predictions and all_targets:
            all_predictions = torch.cat(all_predictions, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            
            # 计算每个类别的F1分数
            for i, class_name in enumerate(self.event_balancer.event_classes.keys()):
                pred_class = (torch.sigmoid(all_predictions[:, i]) > 0.5).float()
                target_class = all_targets[:, i]
                
                # 计算F1分数
                tp = (pred_class * target_class).sum().item()
                fp = (pred_class * (1 - target_class)).sum().item()
                fn = ((1 - pred_class) * target_class).sum().item()
                
                if tp + fp > 0:
                    precision = tp / (tp + fp)
                else:
                    precision = 0.0
                    
                if tp + fn > 0:
                    recall = tp / (tp + fn)
                else:
                    recall = 0.0
                    
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                else:
                    f1 = 0.0
                
                event_class_metrics[class_name] = {
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall
                }

        avg_loss = sum(all_losses) / len(all_losses) if all_losses else float('inf')
        
        return avg_loss, event_class_metrics

    def train_model(self, train_loader, val_loader, epochs, model_name="dcase_model", 
                   checkpoint_dir=None, resume_from=None):
        """完整的模型训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
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
        val_losses = self.training_state.get('val_losses', [])
        class_metrics_history = self.training_state.get('class_metrics_history', {})
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
                val_loss, class_metrics = self.validate(val_loader)
                val_losses.append(val_loss)
                
                # 记录类别指标历史
                for class_name, metrics in class_metrics.items():
                    if class_name not in class_metrics_history:
                        class_metrics_history[class_name] = []
                    class_metrics_history[class_name].append(metrics)
                
                # 更新事件权重
                self.event_balancer.update_weights(class_metrics)
                self.event_balancer.record_performance(class_metrics)
                
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
                checkpoint_freq = cfg.DCASE_TRAIN_CONFIG.get('checkpoint_frequency', 10)
                if checkpoint_dir and (epoch + 1) % checkpoint_freq == 0:
                    self.save_checkpoint(
                        epoch,
                        checkpoint_dir / f'{model_name}_epoch_{epoch + 1}.pth'
                    )
                
                # 更新训练状态
                self.training_state.update({
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'class_metrics_history': class_metrics_history,
                    'patience_counter': patience_counter,
                    'last_epoch': epoch
                })
                
                # 打印训练信息
                print(f"Epoch {epoch + 1}/{epochs}")
                print(f"Train Loss: {train_loss:.4f}")
                print(f"Val Loss: {val_loss:.4f}")
                print("各类别性能:")
                for class_name, metrics in class_metrics.items():
                    if isinstance(metrics, dict) and 'f1_score' in metrics:
                        print(f"- {class_name}: F1={metrics['f1_score']:.3f}, "
                              f"P={metrics['precision']:.3f}, R={metrics['recall']:.3f}")
                
                print(f"时间: {time.time() - epoch_start_time:.2f}s")
                
                # 早停检查
                patience = cfg.DCASE_TRAIN_CONFIG.get('patience', 20)
                if patience_counter >= patience:
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
        
        # 计算类别性能统计
        final_class_metrics = {}
        for class_name, history in class_metrics_history.items():
            if history:
                f1_scores = [h.get('f1_score', 0) for h in history if isinstance(h, dict)]
                if f1_scores:
                    final_class_metrics[class_name] = {
                        'mean_f1': np.mean(f1_scores),
                        'best_f1': max(f1_scores),
                        'current_weight': self.event_balancer.class_weights.get(class_name, 1.0)
                    }
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'class_metrics_history': class_metrics_history,
            'best_epoch': best_epoch,
            'best_loss': best_loss,
            'best_model_state': best_model_state,
            'last_model_state': {
                k: v.cpu().clone() for k, v in self.model.state_dict().items()
            },
            'final_class_metrics': final_class_metrics,
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
            'epoch_times': [],
            'class_f1_scores': {},
            'class_precision': {},
            'class_recall': {}
        }

    def update(self, metric_name, value):
        """更新指标"""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)

    def update_class_metric(self, metric_type, class_name, value):
        """更新类别指标"""
        if metric_type not in self.metrics:
            self.metrics[metric_type] = {}
        if class_name not in self.metrics[metric_type]:
            self.metrics[metric_type][class_name] = []
        self.metrics[metric_type][class_name].append(value)

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


def plot_dcase_losses(train_losses, val_losses, save_path):
    """绘制DCASE损失曲线"""
    plt.figure(figsize=(12, 8))

    # 损失曲线
    plt.subplot(2, 1, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'bo-', label='Training Loss', alpha=0.7)

    # 处理验证损失：可能是字典(按SNR组)或列表
    if isinstance(val_losses, dict):
        # 计算每个epoch的平均验证损失
        num_epochs = max(len(losses) for losses in val_losses.values()) if val_losses else 0
        avg_val_losses = []
        for epoch in range(num_epochs):
            epoch_losses = [losses[epoch] for losses in val_losses.values() if epoch < len(losses)]
            if epoch_losses:
                avg_val_losses.append(sum(epoch_losses) / len(epoch_losses))

        # 绘制平均验证损失
        if avg_val_losses:
            plt.plot(range(1, len(avg_val_losses) + 1), avg_val_losses, 'ro-',
                    label='Validation Loss (Avg)', alpha=0.7, linewidth=2)

        # 绘制各SNR组的验证损失
        colors = ['orange', 'green', 'purple', 'brown']
        for idx, (snr_group, losses) in enumerate(val_losses.items()):
            if losses:
                plt.plot(range(1, len(losses) + 1), losses,
                        linestyle='--', alpha=0.5, color=colors[idx % len(colors)],
                        label=f'{snr_group}')
    else:
        # 原有逻辑：直接绘制列表形式的验证损失
        plt.plot(range(1, len(val_losses) + 1), val_losses, 'ro-', label='Validation Loss', alpha=0.7)

    plt.title('DCASE Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_path}_dcase_losses.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_class_metrics(class_metrics_history, save_path):
    """绘制各类别性能指标"""
    if not class_metrics_history:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # F1分数
    ax1 = axes[0, 0]
    for class_name, history in class_metrics_history.items():
        f1_scores = [h.get('f1_score', 0) for h in history if isinstance(h, dict)]
        if f1_scores:
            ax1.plot(range(1, len(f1_scores) + 1), f1_scores, 'o-', 
                    label=class_name, alpha=0.7, linewidth=2)
    ax1.set_title('F1 Score by Class')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('F1 Score')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 精确率
    ax2 = axes[0, 1]
    for class_name, history in class_metrics_history.items():
        precision_scores = [h.get('precision', 0) for h in history if isinstance(h, dict)]
        if precision_scores:
            ax2.plot(range(1, len(precision_scores) + 1), precision_scores, 'o-', 
                    label=class_name, alpha=0.7, linewidth=2)
    ax2.set_title('Precision by Class')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Precision')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 召回率
    ax3 = axes[1, 0]
    for class_name, history in class_metrics_history.items():
        recall_scores = [h.get('recall', 0) for h in history if isinstance(h, dict)]
        if recall_scores:
            ax3.plot(range(1, len(recall_scores) + 1), recall_scores, 'o-', 
                    label=class_name, alpha=0.7, linewidth=2)
    ax3.set_title('Recall by Class')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Recall')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # 平均性能
    ax4 = axes[1, 1]
    avg_f1_by_epoch = []
    epochs = max(len(history) for history in class_metrics_history.values())
    
    for epoch in range(epochs):
        epoch_f1_scores = []
        for class_name, history in class_metrics_history.items():
            if epoch < len(history) and isinstance(history[epoch], dict):
                epoch_f1_scores.append(history[epoch].get('f1_score', 0))
        if epoch_f1_scores:
            avg_f1_by_epoch.append(np.mean(epoch_f1_scores))
    
    if avg_f1_by_epoch:
        ax4.plot(range(1, len(avg_f1_by_epoch) + 1), avg_f1_by_epoch, 
                'ko-', label='Average F1', linewidth=3)
        ax4.set_title('Average F1 Score Across All Classes')
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('Average F1 Score')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}_dcase_class_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()
