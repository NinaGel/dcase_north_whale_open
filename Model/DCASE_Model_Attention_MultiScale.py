#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DCASE2020专用的多尺度注意力模型
适配DCASE2020数据集的10类声音事件检测任务

基于原始Whale_Model_Attention_MultiScale，但使用DCASE配置
"""

import torch
from torch import nn
import sys
from pathlib import Path

# 添加模型路径
sys.path.append(str(Path(__file__).parent))

from BA_Conv import BSA_Conv
from Spatial_shift_modified import Spatial_shift
from MultiScale_Ldsa import MultiScaleLDSA
import config_dcase as cfg


class DCASE_Model_Attention_MultiScale(nn.Module):
    """使用多尺度LDSA的DCASE声音事件检测模型
    
    专门适配DCASE2020数据集：
    - 输入：128个Mel滤波器 × 313帧
    - 输出：10类声音事件的帧级预测
    """

    def __init__(self):
        super().__init__()
        
        # 使用DCASE配置
        self.config = cfg.DCASE_MODEL_CONFIG
        self.audio_config = cfg.DCASE_AUDIO_CONFIG

        # 验证输入维度 - DCASE使用ACT特征（512 freq bins）
        input_freq = self.audio_config['freq']  # 512 (ACT特征)
        if input_freq & (input_freq - 1) != 0:
            # 如果不是2的幂，我们需要调整卷积参数
            print(f"[Warning] Input frequency dimension {input_freq} is not a power of 2")

        # 验证注意力头数
        n_feat = self.config['feature_dims']['n_feat']  # 256 (32*8)
        n_head = self.config['attention']['n_head']     # 8
        if n_feat % n_head != 0:
            raise ValueError(f"Feature dimension {n_feat} must be divisible by attention heads {n_head}")

        print(f"[Info] Creating DCASE model:")
        print(f"   Input dimensions: {input_freq} x {self.audio_config['frame']}")
        print(f"   Feature dimensions: {n_feat}")
        print(f"   Attention heads: {n_head}")
        print(f"   Output classes: {self.config['num_classes']}")
        
        # 特征提取层 - 使用DCASE配置
        self.bsa_conv1 = BSA_Conv(**self.config['bsa_conv1'])
        self.spatial_shift = Spatial_shift(
            in_channel=self.config['bsa_conv1']['c3_out']
        )
        self.bsa_conv2 = BSA_Conv(**self.config['bsa_conv2'])
        
        # 使用多尺度LDSA - 使用DCASE配置
        self.ldsa = MultiScaleLDSA(
            n_head=self.config['attention']['n_head'],
            n_feat=self.config['feature_dims']['n_feat'],
            dropout_rate=self.config['attention']['dropout'],
            context_sizes=cfg.DCASE_DDSA_CONFIG['context_sizes'],  # [21, 41, 61, 81]
            use_bias=cfg.DCASE_LDSA_CONFIG['use_bias']
        )
        
        # GRU层 - 使用DCASE配置
        self.gru = nn.GRU(
            input_size=self.config['attention']['n_feat'],
            hidden_size=self.config['gru']['hidden_size'],
            num_layers=self.config['gru']['num_layers'],
            batch_first=True,
            bidirectional=True,
            dropout=self.config['gru']['dropout'] 
            if self.config['gru']['num_layers'] > 1 else 0
        )
        
        # 分类层 - 输出DCASE的10个类别
        self.fc = nn.Linear(
            self.config['gru']['hidden_size'] * 2,  # 双向GRU
            self.config['num_classes']  # 10类
        )
        
        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        """前向传播

        Args:
            x: 输入特征 [B, F, T] 或 [B, C, F, T]
               - B: batch size
               - C: 通道数 (通常为1,可选)
               - F: 频率维度 (DCASE: 512 for ACT features)
               - T: 时间维度 (DCASE: 311)

        Returns:
            output: [B, T, num_classes] 帧级预测
        """
        # 确保输入维度正确
        if x.dim() == 3:  # [B, F, T]
            x = x.unsqueeze(1)  # 添加通道维度 -> [B, 1, F, T]
        elif x.dim() != 4:
            raise ValueError(f"Expected 3D or 4D input, got {x.dim()}D: {x.shape}")

        # 记录输入形状用于调试
        input_shape = x.shape

        # 特征提取
        x = self.bsa_conv1(x)  # [B, 8, F/4, T]
        x = self.spatial_shift(x)  # [B, 8, F/4, T]
        x = self.bsa_conv2(x)  # [B, 32, F/16, T]

        # 获取特征维度
        B, C, F, T = x.shape

        # 调整维度为序列格式
        # [B, C, F, T] -> [B, F, C, T] -> [B, F*C, T] -> [B, T, F*C]
        x = x.transpose(1, 2)  # [B, F, C, T]
        x = x.reshape(B, F * C, T)  # [B, F*C, T] -> [B, n_feat, T]
        x = x.transpose(1, 2)  # [B, T, n_feat]

        # 多尺度LDSA注意力
        x = self.ldsa(x, x, x)  # [B, T, F*C] -> [B, 313, 512]

        # GRU处理
        gru_out, _ = self.gru(x)  # [B, T, H*2] -> [B, 313, 512]

        # 分类 - 输出帧级预测
        output = self.fc(gru_out)  # [B, T, num_classes] -> [B, 313, 10]

        return output

    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_name': 'DCASE_Model_Attention_MultiScale',
            'total_params': total_params,
            'trainable_params': trainable_params,
            'input_shape': (1, self.audio_config['freq'], self.audio_config['frame']),
            'output_shape': (self.audio_config['frame'], self.config['num_classes']),
            'num_classes': self.config['num_classes'],
            'class_names': self.config['class_names']
        }


def create_dcase_model():
    """创建DCASE模型的工厂函数"""
    return DCASE_Model_Attention_MultiScale()


# 测试函数
def test_dcase_model():
    """测试DCASE模型"""
    print("🧪 测试DCASE模型...")
    
    model = create_dcase_model()
    model_info = model.get_model_info()
    
    print(f"📋 模型信息:")
    for key, value in model_info.items():
        print(f"   {key}: {value}")
    
    # 创建测试输入
    batch_size = 2
    freq_bins = cfg.DCASE_AUDIO_CONFIG['freq']  # 使用ACT特征维度（512）
    frames = cfg.DCASE_AUDIO_CONFIG['frame']

    test_input = torch.randn(batch_size, 1, freq_bins, frames)
    print(f"\n📊 测试输入形状: {test_input.shape}")
    
    # 前向传播测试
    model.eval()
    with torch.no_grad():
        try:
            output = model(test_input)
            print(f"✅ 前向传播成功")
            print(f"📤 输出形状: {output.shape}")
            print(f"   预期: [batch_size, frames, num_classes] = [{batch_size}, {frames}, {cfg.DCASE_MODEL_CONFIG['num_classes']}]")
            
            # 检查输出范围
            print(f"📈 输出统计:")
            print(f"   最小值: {output.min().item():.4f}")
            print(f"   最大值: {output.max().item():.4f}")
            print(f"   均值: {output.mean().item():.4f}")
            print(f"   标准差: {output.std().item():.4f}")
            
        except Exception as e:
            print(f"❌ 前向传播失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("🎉 测试完成")


if __name__ == "__main__":
    test_dcase_model()
