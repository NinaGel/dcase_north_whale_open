#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DCASE专用音频数据增强器
基于原始AudioAugmentor，但使用config_dcase配置
"""

import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import config_dcase as cfg


class DCASEAudioAugmentor:
    """DCASE专用音频数据增强器
    
    实现多种数据增强策略，专门适配DCASE2020数据集:
    1. 时域增强:
        - 时间拉伸
        - 音高偏移
        - 音量增益
        - 高斯噪声
        - 粉红噪声
    2. 频域增强:
        - SpecAugment (频率掩码和时间掩码)
    3. 混合增强:
        - Mixup
        - Cutmix
        - 随机擦除
    """
    
    def __init__(self, snr_group: Optional[str] = None):
        """初始化DCASE音频增强器
        
        Args:
            snr_group: SNR组名称，用于调整增强参数
        """
        config = cfg.DCASE_AUGMENTATION_CONFIG
        self.enabled = config['enabled']
        self.augment_prob = config['augment_prob']
        
        # 时域增强配置
        time_config = config['time_domain']
        self.time_domain_enabled = time_config['enabled']
        self.time_domain_prob = time_config['prob']
        
        self.time_stretch_range = time_config['time_stretch']['range']
        self.pitch_shift_range = time_config['pitch_shift']['range']
        self.gain_range = time_config['gain']['range']
        self.noise_snr_range = time_config['noise']['snr_range']
        
        # 频域增强配置
        spec_config = config['spec_augment']
        self.spec_augment_enabled = spec_config['enabled']
        self.spec_augment_prob = spec_config['prob']
        self.freq_mask_param = spec_config['freq_mask']['param']
        self.time_mask_param = spec_config['time_mask']['param']
        self.num_freq_mask = spec_config['freq_mask']['num_masks']
        self.num_time_mask = spec_config['time_mask']['num_masks']
        
        # 混合增强配置
        mixing_config = config['mixing']
        self.mixing_enabled = mixing_config['enabled']
        self.mixup_enabled = mixing_config['mixup']['enabled']
        self.mixup_prob = mixing_config['mixup']['prob']
        self.mixup_alpha = mixing_config['mixup']['alpha']
        self.cutmix_enabled = mixing_config['cutmix']['enabled']
        self.cutmix_prob = mixing_config['cutmix']['prob']
        self.cutmix_alpha = mixing_config['cutmix']['alpha']
        
        # 噪声概率
        self.noise_prob = 0.25
        self.mixing_prob = 0.35
        
    def _add_noise(self, x: torch.Tensor, noise_type: str = 'gaussian') -> torch.Tensor:
        """添加噪声"""
        if noise_type == 'gaussian':
            noise = torch.randn_like(x) * 0.005
        elif noise_type == 'pink':
            # 简化的粉红噪声
            noise = torch.randn_like(x) * 0.005
            # 应用简单的低通滤波来模拟粉红噪声特性
            if x.dim() >= 3:
                kernel = torch.tensor([0.2, 0.6, 0.2]).view(1, 1, 3).to(x.device)
                noise = F.conv1d(noise.view(-1, 1, x.size(-1)), kernel, padding=1).view_as(noise)
        else:
            noise = torch.zeros_like(x)
        
        return x + noise
    
    def _apply_time_stretch(self, x: torch.Tensor) -> torch.Tensor:
        """应用时间拉伸（简化版本）"""
        if not self.time_stretch_range:
            return x
        
        stretch_factor = random.uniform(*self.time_stretch_range)
        if abs(stretch_factor - 1.0) < 0.05:  # 如果拉伸因子接近1，跳过
            return x
        
        # 简化的时间拉伸：通过插值实现
        original_length = x.size(-1)
        new_length = int(original_length * stretch_factor)
        
        if new_length > 0:
            # 对于4D张量[batch, channel, freq, time]，需要使用nearest模式或者重新排列维度
            if x.dim() == 4:
                # 重塑为3D: [batch*channel*freq, 1, time] 然后使用linear模式
                batch_size, channels, freq_bins, time_bins = x.shape
                x_reshaped = x.view(batch_size * channels * freq_bins, 1, time_bins)
                x_stretched = F.interpolate(x_reshaped, size=new_length, mode='linear', align_corners=False)
                x_stretched = x_stretched.view(batch_size, channels, freq_bins, new_length)
            else:
                # 对于其他维度，使用nearest模式
                x_stretched = F.interpolate(x, size=new_length, mode='nearest')
            
            # 裁剪或填充到原始长度
            if new_length > original_length:
                x_stretched = x_stretched[..., :original_length]
            elif new_length < original_length:
                padding = original_length - new_length
                x_stretched = F.pad(x_stretched, (0, padding), mode='constant', value=0)
            return x_stretched
        
        return x
    
    def _apply_pitch_shift(self, x: torch.Tensor) -> torch.Tensor:
        """应用音高偏移（简化版本）"""
        if not self.pitch_shift_range:
            return x
        
        # 简化的音高偏移：通过频率域移位实现
        # 将浮点数范围转换为整数bin偏移
        min_shift, max_shift = self.pitch_shift_range
        shift_bins = random.randint(int(min_shift), int(max_shift))
        if shift_bins == 0:
            return x
        
        if shift_bins > 0:
            # 向上移位
            x_shifted = torch.zeros_like(x)
            x_shifted[..., shift_bins:, :] = x[..., :-shift_bins, :]
        else:
            # 向下移位
            x_shifted = torch.zeros_like(x)
            x_shifted[..., :shift_bins, :] = x[..., -shift_bins:, :]
        
        return x_shifted
    
    def _apply_gain(self, x: torch.Tensor) -> torch.Tensor:
        """应用音量增益"""
        if not self.gain_range:
            return x
        
        gain_db = random.uniform(*self.gain_range)
        gain_linear = 10 ** (gain_db / 20.0)
        return x * gain_linear
    
    def _apply_spec_augment(self, x: torch.Tensor) -> torch.Tensor:
        """应用SpecAugment"""
        if not self.spec_augment_enabled:
            return x
        
        batch_size, channels, freq_bins, time_bins = x.shape
        
        # 频率掩码
        for _ in range(self.num_freq_mask):
            freq_mask_size = random.randint(0, min(self.freq_mask_param, freq_bins // 4))
            if freq_mask_size > 0:
                freq_mask_start = random.randint(0, freq_bins - freq_mask_size)
                x[:, :, freq_mask_start:freq_mask_start + freq_mask_size, :] = 0
        
        # 时间掩码
        for _ in range(self.num_time_mask):
            time_mask_size = random.randint(0, min(self.time_mask_param, time_bins // 4))
            if time_mask_size > 0:
                time_mask_start = random.randint(0, time_bins - time_mask_size)
                x[:, :, :, time_mask_start:time_mask_start + time_mask_size] = 0
        
        return x
    
    def _apply_mixup(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用Mixup增强"""
        if x.size(0) < 2:
            return x, y
        
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        mixed_y = lam * y + (1 - lam) * y[index, :]
        
        return mixed_x, mixed_y
    
    def _apply_cutmix(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用CutMix增强"""
        if x.size(0) < 2:
            return x, y
        
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        _, _, H, W = x.shape
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # 随机选择切割区域
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
        
        # 调整标签权重
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        mixed_y = lam * y + (1 - lam) * y[index, :]
        
        return x, mixed_y
    
    def _apply_random_erase(self, x: torch.Tensor) -> torch.Tensor:
        """应用随机擦除"""
        if random.random() > 0.3:  # 30%概率应用随机擦除
            return x
        
        _, _, H, W = x.shape
        area = H * W
        
        for attempt in range(100):
            target_area = random.uniform(0.02, 0.2) * area
            aspect_ratio = random.uniform(0.3, 3.3)
            
            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))
            
            if w < W and h < H:
                x1 = random.randint(0, H - h)
                y1 = random.randint(0, W - w)
                x[:, :, x1:x1 + h, y1:y1 + w] = torch.randn_like(x[:, :, x1:x1 + h, y1:y1 + w]) * 0.1
                break
        
        return x
    
    def augment(self, 
                x: torch.Tensor,
                y: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """应用数据增强
        
        Args:
            x: 输入频谱图，形状为[channel, freq, time]或[batch, channel, freq, time]
            y: 标签，可选，形状为[frames, num_classes]或[batch, frames, num_classes]
            
        Returns:
            增强后的频谱图和标签
        """
        if not self.enabled or random.random() >= self.augment_prob:
            return x, y
        
        # 记录原始维度
        orig_shape = x.shape
        orig_dim = x.dim()
        
        # 确保输入维度正确：[batch, channel, freq, time]
        if orig_dim == 3:  # [channel, freq, time]
            x = x.unsqueeze(0)  # [1, channel, freq, time]
            single_sample = True
        else:
            single_sample = False
        
        # 时域增强（在频谱域进行）
        if random.random() < self.noise_prob:
            if random.random() < 0.5:
                x = self._add_noise(x, noise_type='gaussian')
            else:
                x = self._add_noise(x, noise_type='pink')
                
        if random.random() < self.time_domain_prob:
            x = self._apply_time_stretch(x)
        if random.random() < self.time_domain_prob:
            x = self._apply_pitch_shift(x)
        if random.random() < self.time_domain_prob:
            x = self._apply_gain(x)
        
        # 频域增强
        if random.random() < self.spec_augment_prob:
            x = self._apply_spec_augment(x)
        
        # 混合增强（只对批次数据有效）
        if y is not None and self.mixing_enabled and x.size(0) > 1:
            if random.random() < self.mixing_prob:
                if random.random() < 0.5 and self.mixup_enabled:
                    x, y = self._apply_mixup(x, y)
                elif self.cutmix_enabled:
                    x, y = self._apply_cutmix(x, y)
        
        # 随机擦除
        x = self._apply_random_erase(x)
        
        # 恢复原始维度
        if single_sample:
            x = x.squeeze(0)  # 移除batch维度
        
        return x, y
