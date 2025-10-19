import torch
import torch.nn as nn
import numpy as np
import random
import torchaudio
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import config as cfg


class AudioAugmentor:
    """音频数据增强器
    
    实现多种数据增强策略，包括:
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
        """初始化音频增强器
        
        Args:
            snr_group: SNR组名称，用于调整增强参数
        """
        config = cfg.AUGMENTATION_CONFIG
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
        
        self.random_erase_enabled = mixing_config['random_erase']['enabled']
        self.random_erase_prob = mixing_config['random_erase']['prob']
        self.random_erase_area_ratio = mixing_config['random_erase']['area_ratio']
        self.random_erase_aspect_ratio = mixing_config['random_erase']['aspect_ratio']
        self.random_erase_noise_std = mixing_config['random_erase']['noise_std']
        
        # SNR相关配置
        if snr_group and config['snr_based']['enabled']:
            snr_config = config['snr_based'][snr_group]
            self.noise_prob = snr_config['noise_prob']
            self.spec_augment_prob = snr_config['spec_augment_prob']
            self.mixing_prob = snr_config['mixing_prob']
        else:
            self.noise_prob = self.time_domain_prob
            self.spec_augment_prob = self.spec_augment_prob
            self.mixing_prob = self.mixup_prob
        
        # 初始化频谱增强
        self.freq_masking = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=self.freq_mask_param
        )
        self.time_masking = torchaudio.transforms.TimeMasking(
            time_mask_param=self.time_mask_param
        )
        
    def _apply_time_stretch(self, spec: torch.Tensor) -> torch.Tensor:
        """应用时间拉伸（在频谱域）"""
        if not self.time_domain_enabled:
            return spec
            
        rate = random.uniform(*self.time_stretch_range)
        if abs(rate - 1.0) < 1e-3:  # 如果拉伸率接近1，则不进行拉伸
            return spec
            
        # 使用插值进行时间拉伸
        orig_time = spec.size(-1)
        target_time = int(orig_time * rate)
        
        # 使用双线性插值进行拉伸
        stretched = F.interpolate(
            spec.unsqueeze(1) if spec.dim() == 3 else spec,
            size=(spec.size(-2), target_time),
            mode='bilinear',
            align_corners=False
        )
        
        # 如果输入是3维的，则移除添加的维度
        if spec.dim() == 3:
            stretched = stretched.squeeze(1)
        
        # 确保输出维度与输入相同
        if target_time > orig_time:
            # 如果拉伸后更长，则随机裁剪
            start = random.randint(0, target_time - orig_time)
            stretched = stretched[..., start:start + orig_time]
        elif target_time < orig_time:
            # 如果拉伸后更短，则填充
            pad_left = random.randint(0, orig_time - target_time)
            pad_right = orig_time - target_time - pad_left
            stretched = F.pad(stretched, (pad_left, pad_right))
            
        return stretched
    
    def _apply_pitch_shift(self, spec: torch.Tensor) -> torch.Tensor:
        """应用音高偏移（在频谱域）"""
        if not self.time_domain_enabled:
            return spec
            
        shift = random.randint(*self.pitch_shift_range)
        if shift == 0:
            return spec
            
        # 在频率维度上移动
        freq_bins = spec.size(-2)
        shift_bins = int(abs(shift) * freq_bins / 12)  # 将半音转换为频率bin数
        
        if shift > 0:
            # 向上移动音高
            shifted = F.pad(spec, (0, 0, shift_bins, 0))[..., :-shift_bins, :]
        else:
            # 向下移动音高
            shifted = F.pad(spec, (0, 0, 0, shift_bins))[..., shift_bins:, :]
            
        return shifted
    
    def _apply_gain(self, spec: torch.Tensor) -> torch.Tensor:
        """应用音量增益（在频谱域）"""
        if not self.time_domain_enabled:
            return spec
            
        gain = random.uniform(*self.gain_range)
        return spec * (10 ** (gain / 20.0))
    
    def _add_noise(self, spec: torch.Tensor, noise_type: str = 'gaussian') -> torch.Tensor:
        """添加噪声（在频谱域）"""
        if not self.time_domain_enabled:
            return spec
            
        snr = random.uniform(*self.noise_snr_range)
        signal_power = spec.pow(2).mean()
        noise_power = signal_power / (10 ** (snr / 10.0))
        
        if noise_type == 'gaussian':
            noise = torch.randn_like(spec) * torch.sqrt(noise_power)
        else:  # pink noise
            noise = torch.randn_like(spec)
            # 应用1/f滤波
            freq_dim = spec.size(-2)
            pink_filter = 1 / torch.sqrt(torch.arange(1, freq_dim + 1, device=spec.device).float())
            pink_filter = pink_filter.view(-1, 1)
            noise = noise * pink_filter
            noise *= torch.sqrt(noise_power / noise.pow(2).mean())
            
        return spec + noise
    
    def _apply_spec_augment(self, spec: torch.Tensor) -> torch.Tensor:
        """应用SpecAugment"""
        if not self.spec_augment_enabled:
            return spec
            
        aug_spec = spec.clone()
        
        for _ in range(self.num_freq_mask):
            aug_spec = self.freq_masking(aug_spec)
            
        for _ in range(self.num_time_mask):
            aug_spec = self.time_masking(aug_spec)
            
        return aug_spec
    
    def _apply_mixup(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用Mixup"""
        if not self.mixup_enabled or x.size(0) <= 1:
            return x, y
            
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:
            lam = 1
            
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        mixed_y = lam * y + (1 - lam) * y[index]
        
        return mixed_x, mixed_y
    
    def _apply_cutmix(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用Cutmix增强
        
        Args:
            x: 输入张量，支持3D [batch, freq, time] 或 4D [batch, channel, freq, time]
            y: 标签张量
            
        Returns:
            tuple: (增强后的特征张量, 混合后的标签)
        """
        if not self.cutmix_enabled or x.size(0) <= 1:
            return x, y
            
        # 确保输入维度正确
        if x.dim() not in [3, 4]:
            raise ValueError(f"输入维度必须是3或4，但得到{x.dim()}")
            
        is_3d = x.dim() == 3
        if is_3d:
            x = x.unsqueeze(1)  # [B, 1, F, T]
            
        # 获取批次索引
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        
        # 获取混合比例
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        
        # 获取特征维度
        _, n_channels, freq_size, time_size = x.size()
        
        # 计算裁剪区域大小
        freq_cut = int(freq_size * np.sqrt(1.0 - lam))
        time_cut = int(time_size * np.sqrt(1.0 - lam))
        
        # 确保裁剪区域不为0
        freq_cut = max(freq_cut, 1)
        time_cut = max(time_cut, 1)
        
        # 随机选择裁剪位置
        freq_start = random.randint(0, freq_size - freq_cut)
        time_start = random.randint(0, time_size - time_cut)
        
        # 应用cutmix
        x_cutmix = x.clone()
        x_cutmix[:, :, freq_start:freq_start+freq_cut, time_start:time_start+time_cut] = \
            x[index, :, freq_start:freq_start+freq_cut, time_start:time_start+time_cut]
            
        # 计算混合比例
        lam = 1 - (freq_cut * time_cut) / (freq_size * time_size)
        mixed_y = lam * y + (1 - lam) * y[index]
        
        # 如果输入是3D，则恢复原始维度
        if is_3d:
            x_cutmix = x_cutmix.squeeze(1)
        
        return x_cutmix, mixed_y
    
    def _apply_random_erase(self, x: torch.Tensor) -> torch.Tensor:
        """应用随机擦除
        
        Args:
            x: 输入张量，支持3D [batch, freq, time] 或 4D [batch, channel, freq, time]
            
        Returns:
            增强后的张量，与输入维度相同
        """
        if not self.random_erase_enabled or random.random() > self.random_erase_prob:
            return x
            
        # 确保输入维度正确
        if x.dim() not in [3, 4]:
            raise ValueError(f"输入维度必须是3或4，但得到{x.dim()}")
            
        is_3d = x.dim() == 3
        if is_3d:
            x = x.unsqueeze(1)  # [B, 1, F, T]
            
        # 获取正确的维度
        batch_size, n_channels, freq_size, time_size = x.size()
        x_erased = x.clone()
        
        for i in range(batch_size):
            area = freq_size * time_size
            target_area = random.uniform(*self.random_erase_area_ratio) * area
            aspect_ratio = random.uniform(*self.random_erase_aspect_ratio)
            
            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))
            
            if h < freq_size and w < time_size:
                top = random.randint(0, freq_size - h)
                left = random.randint(0, time_size - w)
                
                # 为每个通道生成噪声
                noise = torch.randn(
                    (n_channels, h, w),
                    device=x.device
                ) * self.random_erase_noise_std
                
                x_erased[i, :, top:top+h, left:left+w] = noise
                
        # 如果输入是3D，则恢复原始维度
        if is_3d:
            x_erased = x_erased.squeeze(1)
                
        return x_erased
    
    def augment(self, 
                x: torch.Tensor,
                y: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """应用数据增强
        
        Args:
            x: 输入频谱图，形状为[batch, freq, time]或[batch, channel, freq, time]
            y: 标签，可选
            
        Returns:
            增强后的频谱图和标签
        """
        if not self.enabled or random.random() >= self.augment_prob:
            return x, y
        
        # 记录原始维度
        orig_shape = x.shape
        orig_dim = x.dim()
        
        # 确保输入维度正确
        if orig_dim == 3:  # [batch, freq, time]
            x = x.unsqueeze(1)  # [batch, channel, freq, time]
        
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
        
        # 混合增强
        if y is not None and self.mixing_enabled and x.size(0) > 1:
            if random.random() < self.mixing_prob:
                if random.random() < 0.5:
                    x, y = self._apply_mixup(x, y)
                else:
                    x, y = self._apply_cutmix(x, y)
        
        x = self._apply_random_erase(x)
        
        # 确保输出维度与输入相同
        if x.shape != orig_shape:
            x = F.interpolate(x, size=(orig_shape[-2], orig_shape[-1]), mode='bilinear', align_corners=False)
        
        # 如果输入是3维的，则移除channel维度
        if orig_dim == 3:
            x = x.squeeze(1)
        
        return x, y 