#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DCASE训练器 / DCASE Trainer
用于DCASE2020数据集的帧级声音事件检测
Frame-level sound event detection for DCASE2020 dataset

特点 / Features:
- 帧级标签训练 / Frame-level label training
- 10类事件检测 / 10-class event detection
- DCASE专用评估指标 / DCASE-specific evaluation metrics
"""

import torch
import torch.nn as nn
import numpy as np
from torch.cuda import amp
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

# 导入基础训练器
from Train.train_utils import BaseTrainer, MetricTracker
import config_dcase as cfg


class DCASEMetricTracker(MetricTracker):
    """DCASE指标追踪器 / DCASE Metric Tracker

    追踪帧级、事件级、段级F1等DCASE特有指标
    Tracks frame-level, event-level, segment-level F1 and other DCASE metrics
    """

    def __init__(self):
        super().__init__()
        # DCASE特有指标
        self.add_metric('frame_accuracy')     # 帧级准确率
        self.add_metric('event_f1')          # 事件级F1
        self.add_metric('segment_f1')        # 段级F1
        self.add_metric('class_wise_f1')     # 每类F1分数
        
    def compute_frame_metrics(self, predictions, targets, threshold=0.5):
        """计算帧级指标
        
        Args:
            predictions: [batch_size, frames, num_classes]
            targets: [batch_size, frames, num_classes]
            threshold: 阈值
        """
        # 转换为二进制预测
        pred_binary = (predictions > threshold).float()
        
        # 帧级准确率 (考虑多标签的情况)
        correct_frames = (pred_binary == targets).all(dim=-1)  # [batch_size, frames]
        frame_acc = correct_frames.float().mean().item()
        
        self.update('frame_accuracy', frame_acc)
        
        return {
            'frame_accuracy': frame_acc,
            'pred_binary': pred_binary
        }
    
    def compute_class_wise_metrics(self, predictions, targets, threshold=0.5):
        """计算每类别的指标"""
        pred_binary = (predictions > threshold).float()
        
        # 计算每个类别的F1分数
        class_f1_scores = []
        num_classes = predictions.shape[-1]
        
        for class_idx in range(num_classes):
            pred_class = pred_binary[..., class_idx].flatten()
            target_class = targets[..., class_idx].flatten()
            
            # 计算TP, FP, FN
            tp = ((pred_class == 1) & (target_class == 1)).sum().float()
            fp = ((pred_class == 1) & (target_class == 0)).sum().float()
            fn = ((pred_class == 0) & (target_class == 1)).sum().float()
            
            # 计算F1分数
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            
            class_f1_scores.append(f1.item())
        
        avg_class_f1 = np.mean(class_f1_scores)
        self.update('class_wise_f1', avg_class_f1)
        
        return {
            'class_f1_scores': class_f1_scores,
            'avg_class_f1': avg_class_f1
        }


class DCASETrainer(BaseTrainer):
    """DCASE专用训练器"""
    
    def __init__(self, model, optimizer, loss_fn, device, scaler=None):
        """初始化DCASE训练器
        
        Args:
            model: 模型
            optimizer: 优化器  
            loss_fn: 损失函数
            device: 设备
            scaler: 混合精度训练的梯度缩放器
        """
        # 不需要SNR平衡器，因为DCASE数据集没有SNR分组
        super().__init__(model, optimizer, loss_fn, device, snr_balancer=None, scaler=scaler)
        
        # 替换为DCASE专用指标追踪器
        self.metric_tracker = DCASEMetricTracker()
        
        # DCASE特定配置
        self.num_classes = cfg.DCASE_MODEL_CONFIG['num_classes']
        self.class_names = cfg.DCASE_MODEL_CONFIG['class_names']
        
        logging.info(f"DCASE训练器初始化完成，类别数: {self.num_classes}")
        
    def _initialize_scheduler(self):
        """初始化学习率调度器 - 使用DCASE配置"""
        scheduler_config = cfg.DCASE_TRAIN_CONFIG.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'cosine')
        
        if scheduler_type == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            return CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.get('T_max', cfg.DCASE_TRAIN_CONFIG['epochs']),
                eta_min=scheduler_config.get('eta_min', 5e-7)
            )
        else:
            return None
    
    def train_epoch(self, train_loader):
        """训练一个epoch - 适配DCASE数据格式"""
        self.model.train()
        total_loss = 0
        batch_times = []
        
        # 获取梯度裁剪配置
        grad_clip = cfg.DCASE_TRAIN_CONFIG.get('grad_clip', False)
        grad_clip_value = cfg.DCASE_TRAIN_CONFIG.get('grad_clip_value', 1.0)
        
        current_lr = self.optimizer.param_groups[0]['lr']
        
        with tqdm(train_loader, desc=f"DCASE Epoch {self.current_epoch + 1} (lr={current_lr:.2e})") as pbar:
            for i, batch in enumerate(pbar):
                batch_start = time.time()
                
                # DCASE数据格式: (features, labels, filenames)
                if len(batch) >= 3:
                    features, labels, filenames = batch[:3]
                elif len(batch) == 2:
                    features, labels = batch[:2]
                else:
                    raise ValueError(f"DCASE batch格式错误: {len(batch)}")
                
                # 移动到设备
                features = features.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # 确保输入维度正确 [batch_size, channel, n_mels, frames]
                if features.dim() == 3:  # [batch_size, n_mels, frames]
                    features = features.unsqueeze(1)  # [batch_size, 1, n_mels, frames]
                
                # 使用混合精度训练
                with amp.autocast(enabled=self.scaler.is_enabled()):
                    predictions = self.model(features)  # [batch_size, frames, num_classes]
                    
                    # 确保预测和标签维度匹配
                    if predictions.shape != labels.shape:
                        logging.warning(f"维度不匹配: pred {predictions.shape}, label {labels.shape}")
                        # 可能需要转置或调整维度
                        if len(predictions.shape) == 3 and len(labels.shape) == 3:
                            if predictions.shape[1] != labels.shape[1]:
                                predictions = predictions.transpose(1, 2)
                    
                    loss = self.loss_fn(predictions, labels)
                
                # 反向传播
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪
                if grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = clip_grad_norm_(self.model.parameters(), grad_clip_value)
                    self.metric_tracker.update('grad_norms', grad_norm.item())
                
                # 更新参数
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                
                # 统计
                total_loss += loss.item()
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)
                
                # 计算训练指标
                with torch.no_grad():
                    frame_metrics = self.metric_tracker.compute_frame_metrics(
                        torch.sigmoid(predictions), labels
                    )
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'frame_acc': f'{frame_metrics["frame_accuracy"]:.3f}',
                    'time': f'{batch_time:.2f}s'
                })
        
        avg_loss = total_loss / len(train_loader)
        avg_time = np.mean(batch_times)
        
        return avg_loss, avg_time
    
    def validate(self, val_loader):
        """验证模型 - 适配DCASE数据格式"""
        self.model.eval()
        total_loss = 0
        batch_count = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            with amp.autocast(enabled=self.scaler.is_enabled()):
                for batch in tqdm(val_loader, desc="Validating", leave=False):
                    # DCASE数据格式
                    if len(batch) >= 3:
                        features, labels, filenames = batch[:3]
                    elif len(batch) == 2:
                        features, labels = batch[:2]
                    else:
                        raise ValueError(f"DCASE验证batch格式错误: {len(batch)}")
                    
                    features = features.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    
                    # 确保输入维度正确
                    if features.dim() == 3:
                        features = features.unsqueeze(1)
                    
                    predictions = self.model(features)
                    
                    # 调整维度匹配
                    if predictions.shape != labels.shape:
                        if len(predictions.shape) == 3 and len(labels.shape) == 3:
                            if predictions.shape[1] != labels.shape[1]:
                                predictions = predictions.transpose(1, 2)
                    
                    loss = self.loss_fn(predictions, labels)
                    
                    total_loss += loss.item()
                    batch_count += 1
                    
                    # 收集预测和标签用于指标计算
                    all_predictions.append(torch.sigmoid(predictions).cpu())
                    all_targets.append(labels.cpu())
        
        # 计算验证指标
        if all_predictions:
            all_predictions = torch.cat(all_predictions, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            
            # 计算帧级指标
            frame_metrics = self.metric_tracker.compute_frame_metrics(
                all_predictions, all_targets
            )
            
            # 计算类别级指标
            class_metrics = self.metric_tracker.compute_class_wise_metrics(
                all_predictions, all_targets
            )
            
            val_metrics = {
                'frame_accuracy': frame_metrics['frame_accuracy'],
                'class_f1_scores': class_metrics['class_f1_scores'],
                'avg_class_f1': class_metrics['avg_class_f1']
            }
        else:
            val_metrics = {}
        
        avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
        
        return avg_loss, val_metrics
    
    def get_dcase_training_summary(self):
        """获取DCASE训练摘要"""
        summary = {
            'model_info': {
                'num_classes': self.num_classes,
                'class_names': self.class_names,
                'total_params': sum(p.numel() for p in self.model.parameters()),
                'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            },
            'training_state': {
                'current_epoch': self.current_epoch,
                'best_loss': self.best_loss,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            },
            'metrics': self.metric_tracker.get_summary()
        }
        
        return summary
    
    def save_dcase_checkpoint(self, epoch, save_path, val_metrics=None):
        """保存DCASE检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict(),
            'best_loss': self.best_loss,
            'best_model_state': self.best_model_state,
            'training_state': self.training_state,
            'val_metrics': val_metrics,
            'dcase_config': {
                'num_classes': self.num_classes,
                'class_names': self.class_names
            },
            'metric_history': self.metric_tracker.get_summary()
        }
        
        torch.save(checkpoint, save_path)
        logging.info(f"DCASE检查点已保存到: {save_path}")
        
        return checkpoint


def create_dcase_trainer(model, device):
    """创建DCASE训练器的工厂函数"""
    
    # 初始化优化器
    optimizer = cfg.DCASE_TRAIN_CONFIG['optimizer'](
        model.parameters(),
        **cfg.DCASE_TRAIN_CONFIG['optimizer_params']
    )
    
    # 损失函数
    loss_fn = cfg.loss_fn
    
    # 混合精度缩放器
    scaler = amp.GradScaler(enabled=cfg.DCASE_TRAIN_CONFIG['mixed_precision'])
    
    # 创建训练器
    trainer = DCASETrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        scaler=scaler
    )
    
    return trainer


if __name__ == "__main__":
    # 测试DCASE训练器
    print("🧪 测试DCASE训练器...")
    
    # 这里可以添加简单的单元测试
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建dummy模型进行测试
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 32, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((313, 1))
            self.fc = nn.Linear(32, 10)
            
        def forward(self, x):
            # x: [batch, 1, 128, 313]
            x = self.conv(x)  # [batch, 32, 128, 313]
            x = self.pool(x)  # [batch, 32, 313, 1]
            x = x.squeeze(-1).transpose(1, 2)  # [batch, 313, 32]
            x = self.fc(x)  # [batch, 313, 10]
            return x
    
    model = DummyModel().to(device)
    trainer = create_dcase_trainer(model, device)
    
    print(f"✅ DCASE训练器创建成功")
    print(f"   类别数: {trainer.num_classes}")
    print(f"   类别名: {trainer.class_names}")
    print("🎉 测试完成！")

