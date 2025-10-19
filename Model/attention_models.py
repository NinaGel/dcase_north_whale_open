import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


from Model.BA_Conv import BSA_Conv
from Model.Ldsa import LocalDenseSynthesizerAttention
from Model.Spatial_shift_modified import Spatial_shift
from Model.MultiScale_Ldsa import MultiScaleLDSA
from Model.Synthesizer import DenseSynthesizerAttention
import config as cfg
from typing import Dict, Any, Optional, Tuple


class StandardSelfAttention(nn.Module):
    """标准自注意力机制"""
    
    def __init__(self, n_head: int, n_feat: int, use_bias: bool = True):
        """
        Args:
            n_head: 注意力头数
            n_feat: 特征维度 (512)
            use_bias: 是否使用偏置
        """
        super().__init__()
        self.n_head = n_head
        self.n_feat = n_feat
        self.d_k = n_feat // n_head
        self.scaling = self.d_k ** -0.5
        
        # 初始化Q、K、V变换矩阵
        self.w_q = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.w_k = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.w_v = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.w_out = nn.Linear(n_feat, n_feat, bias=use_bias)
        
        # 用于存储最后一次的注意力权重
        self.last_attention_weights = None
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: (batch_size, seq_len, n_feat) 例如 (1, 309, 512)
            key: 同上
            value: 同上
            mask: 可选，(batch_size, seq_len, seq_len)
        Returns:
            output: (batch_size, seq_len, n_feat) 例如 (1, 309, 512)
        """
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # 检查输入维度
        assert query.size() == (batch_size, seq_len, self.n_feat), \
            f"Query shape should be ({batch_size}, {seq_len}, {self.n_feat}), got {query.size()}"
        
        # 线性变换
        q = self.w_q(query).view(batch_size, seq_len, self.n_head, self.d_k)
        k = self.w_k(key).view(batch_size, seq_len, self.n_head, self.d_k)
        v = self.w_v(value).view(batch_size, seq_len, self.n_head, self.d_k)
        
        # 调整维度顺序
        q = q.transpose(1, 2)  # [batch_size, n_head, seq_len, d_k]
        k = k.transpose(1, 2)  # [batch_size, n_head, seq_len, d_k]
        v = v.transpose(1, 2)  # [batch_size, n_head, seq_len, d_k]
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.n_head, -1, -1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attn = F.softmax(scores, dim=-1)
        
        # 保存注意力权重
        self.last_attention_weights = attn.detach()
        
        # 应用注意力
        x = torch.matmul(attn, v)  # [batch_size, n_head, seq_len, d_k]
        
        # 重塑张量
        x = x.transpose(1, 2).contiguous()  # [batch_size, seq_len, n_head, d_k]
        x = x.view(batch_size, seq_len, -1)  # [batch_size, seq_len, n_feat]
        
        # 输出投影
        x = self.w_out(x)
        
        return x


class DenseSynthesizerAttention(nn.Module):
    """密集合成器注意力机制"""
    
    def __init__(self, n_head: int, n_feat: int, use_bias: bool = True):
        """
        Args:
            n_head: 注意力头数
            n_feat: 特征维度
            use_bias: 是否使用偏置
        """
        super().__init__()
        self.n_head = n_head
        self.n_feat = n_feat
        self.d_k = n_feat // n_head
        
        # 初始化变换矩阵
        self.linear_v = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.linear_out = nn.Linear(n_feat, n_feat, bias=use_bias)
        
        # 合成器网络
        hidden_dim = n_feat * 2
        self.synthesizer = nn.Sequential(
            nn.Linear(n_feat, hidden_dim, bias=use_bias),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_head * self.d_k, bias=use_bias)  # 修改输出维度
        )
        
        # 用于存储最后一次的注意力权重
        self.last_attention_weights = None
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: [batch_size, seq_len, n_feat]
            key: [batch_size, seq_len, n_feat]
            value: [batch_size, seq_len, n_feat]
            mask: Optional[batch_size, seq_len, seq_len]
            
        Returns:
            output: [batch_size, seq_len, n_feat]
        """
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # 生成注意力权重
        attn_weights = self.synthesizer(query)  # [batch_size, seq_len, n_head * d_k]
        attn_weights = attn_weights.view(batch_size, seq_len, self.n_head, self.d_k)
        attn_weights = attn_weights.transpose(1, 2)  # [batch_size, n_head, seq_len, d_k]
        
        # 生成注意力分数矩阵
        scores = torch.matmul(attn_weights, attn_weights.transpose(-2, -1))  # [batch_size, n_head, seq_len, seq_len]
        scores = scores / math.sqrt(self.d_k)  # 缩放
        
        # 应用mask（如果有）
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.n_head, -1, -1)  # [batch_size, n_head, seq_len, seq_len]
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 应用softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # 保存注意力权重
        self.last_attention_weights = attn_weights.detach()
        
        # 处理value
        v = self.linear_v(value).view(batch_size, seq_len, self.n_head, self.d_k)
        v = v.transpose(1, 2)  # [batch_size, n_head, seq_len, d_k]
        
        # 应用注意力
        x = torch.matmul(attn_weights, v)  # [batch_size, n_head, seq_len, d_k]
        
        # 重塑并投影
        x = x.transpose(1, 2).contiguous()  # [batch_size, seq_len, n_head, d_k]
        x = x.view(batch_size, seq_len, -1)  # [batch_size, seq_len, n_feat]
        x = self.linear_out(x)
        
        return x


class WhaleModelBase(nn.Module):
    """基础鲸鱼声音检测模型"""
    
    def __init__(self, attention_module: nn.Module):
        super().__init__()
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # BSA-Conv层和空间位移
        self.bsa_conv1 = BSA_Conv(**cfg.MODEL_CONFIG['bsa_conv1'])
        self.spatial_shift = Spatial_shift(
            in_channel=cfg.MODEL_CONFIG['bsa_conv1']['c3_out']
        )
        self.bsa_conv2 = BSA_Conv(**cfg.MODEL_CONFIG['bsa_conv2'])
        
        # 注意力模块
        self.attention = attention_module
        
        # GRU层
        self.gru = nn.GRU(
            input_size=cfg.MODEL_CONFIG['feature_dims']['n_feat'],
            hidden_size=cfg.MODEL_CONFIG['gru']['hidden_size'],
            num_layers=cfg.MODEL_CONFIG['gru']['num_layers'],
            batch_first=True,
            bidirectional=True,
            dropout=cfg.MODEL_CONFIG['gru']['dropout'] 
            if cfg.MODEL_CONFIG['gru']['num_layers'] > 1 else 0
        )
        
        # 分类层
        self.fc = nn.Linear(
            cfg.MODEL_CONFIG['gru']['hidden_size'] * 2,
            cfg.MODEL_CONFIG['num_classes']
        )
        
        # 存储注意力权重
        self.attention_weights = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, channel, freq, time] 或 [batch_size, freq, time]
        
        Returns:
            output: 分类输出，形状为 [batch_size, time, num_classes]
        """
        # 确保输入数据类型为 float32
        x = x.to(torch.float32)
        
        # 如果输入是3维的，添加通道维度
        if x.dim() == 3:
            x = x.unsqueeze(1)  # [B, 1, H, W]
        
        # 检查输入维度
        B, C, F, T = x.shape
        assert C == 1, f"Expected 1 channel, got {C}"
        assert F == cfg.AUDIO_CONFIG['freq'], f"Expected frequency dimension {cfg.AUDIO_CONFIG['freq']}, got {F}"
        assert T == cfg.AUDIO_CONFIG['frame'], f"Expected time dimension {cfg.AUDIO_CONFIG['frame']}, got {T}"
        
        # BSA-Conv1
        x = self.bsa_conv1(x)  # [B, 8, H/3, W]
        
        # 空间位移
        x = self.spatial_shift(x)  # [B, 8, H/3, W]
        
        # BSA-Conv2
        x = self.bsa_conv2(x)  # [B, 32, H/6, W]
        
        # 调整维度
        B, C, F, T = x.shape
        x = x.transpose(1, 2)  # [B, F, C, T]
        x = x.reshape(B, F * C, T)  # [B, F*C, T]
        x = x.transpose(1, 2)  # [B, T, F*C]
        
        # 注意力机制
        x = self.attention(x, x, x)  # [B, T, F*C]
        
        # 保存注意力权重
        if hasattr(self.attention, 'last_attention_weights'):
            self.attention_weights = self.attention.last_attention_weights
        
        # GRU层
        x, _ = self.gru(x)  # [B, T, H*2]
        
        # 分类层
        x = self.fc(x)  # [B, T, num_classes]
        
        return x


def initialize_model(model_variant: str) -> nn.Module:
    """初始化注意力变体模型
    
    Args:
        model_variant: 模型变体名称
            - StandardSA: 标准自注意力
            - DSA: 密集合成器注意力
            - LDSA: 局部密集合成器注意力
            - MultiScale: 多尺度局部密集合成器注意力
            
    Returns:
        nn.Module: 初始化的模型
    """
    if model_variant == 'StandardSA':
        attention_module = StandardSelfAttention(
            n_head=cfg.MODEL_CONFIG['attention']['n_head'],
            n_feat=cfg.MODEL_CONFIG['feature_dims']['n_feat'],
            use_bias=cfg.LDSA_OPTIMIZED_CONFIG['use_bias']
        )
    elif model_variant == 'DSA':
        attention_module = DenseSynthesizerAttention(
            n_head=cfg.MODEL_CONFIG['attention']['n_head'],
            n_feat=cfg.MODEL_CONFIG['feature_dims']['n_feat'],
            use_bias=cfg.LDSA_OPTIMIZED_CONFIG['use_bias']
        )
    elif model_variant == 'LDSA':
        attention_module = LocalDenseSynthesizerAttention(
            n_head=cfg.MODEL_CONFIG['attention']['n_head'],
            n_feat=cfg.MODEL_CONFIG['feature_dims']['n_feat'],
            dropout_rate=cfg.MODEL_CONFIG['attention']['dropout'],
            context_size=cfg.MODEL_CONFIG['attention']['context_size'],
            use_bias=cfg.LDSA_OPTIMIZED_CONFIG['use_bias']
        )
    elif model_variant == 'MultiScale':
        attention_module = MultiScaleLDSA(
            n_head=cfg.MODEL_CONFIG['attention']['n_head'],
            n_feat=cfg.MODEL_CONFIG['feature_dims']['n_feat'],
            dropout_rate=cfg.MODEL_CONFIG['attention']['dropout'],
            context_sizes=cfg.MULTISCALE_CONFIG['context_sizes'],
            use_bias=cfg.LDSA_OPTIMIZED_CONFIG['use_bias']
        )
    else:
        raise ValueError(f"未知的模型变体: {model_variant}")
    
    return WhaleModelBase(attention_module) 