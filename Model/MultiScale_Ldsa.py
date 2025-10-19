import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def chunkwise(xs, N_l, N_c, N_r):
    """分块函数 / Chunking function

    将输入按滑动窗口分块
    Split input into overlapping chunks with sliding window

    Args:
        xs: 输入张量 / input tensor [B, T, D]
        N_l: 左上下文 / left context
        N_c: 中心块 / center chunk
        N_r: 右上下文 / right context

    Returns:
        分块张量 / chunked tensor [B, T, N_l+N_c+N_r, D]
    """
    bs, xmax, idim = xs.size()
    n_chunks = xmax
    c = N_l + N_c + N_r
    
    # 填充输入
    xs_pad = torch.cat([
        xs.new_zeros(bs, N_l, idim),
        xs,
        xs.new_zeros(bs, N_r, idim)
    ], dim=1)
    
    # 创建滑动窗口
    chunks = []
    for i in range(xmax):
        start_idx = i
        chunk = xs_pad[:, start_idx:start_idx + c, :]
        chunks.append(chunk)
    
    # 堆叠所有块
    xs_chunk = torch.stack(chunks, dim=1)  # [B, T, N_l+N_c+N_r, D]
    return xs_chunk


class MultiScaleLDSA(nn.Module):
    """多尺度局部密集合成注意力 / Multi-Scale Local Dense Synthesizer Attention

    使用多个上下文窗口尺度，捕获不同范围的时序依赖
    Uses multiple context window sizes to capture temporal dependencies at different scales
    """

    def __init__(self, n_head, n_feat, dropout_rate, context_sizes=[31, 61], use_bias=False):
        super().__init__()
        assert n_feat % n_head == 0
        # 头数需能被尺度数整除 / heads must be divisible by number of scales
        assert n_head % len(context_sizes) == 0
        
        self.d_k = n_feat // n_head
        self.h = n_head
        self.context_sizes = context_sizes
        self.heads_per_scale = n_head // len(context_sizes)
        
        # 各尺度线性变换 / Linear transforms per scale
        self.w1 = nn.ModuleList([
            nn.Linear(n_feat, n_feat, bias=use_bias)  # 保持输入维度不变
            for _ in range(len(context_sizes))
        ])
        
        self.w2 = nn.ModuleList([
            nn.Linear(n_feat, self.heads_per_scale * context_size, bias=use_bias)
            for context_size in context_sizes
        ])
        
        self.w3 = nn.ModuleList([
            nn.Linear(n_feat, n_feat, bias=use_bias)  # 保持输入维度不变
            for _ in range(len(context_sizes))
        ])
        
        # 尺度融合权重 / Scale fusion weights
        self.scale_weights = nn.Parameter(
            torch.ones(len(context_sizes)) / len(context_sizes)
        )
        
        self.w_out = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.attn = None
        
    def forward(self, query, key, value):
        bs, time = query.size()[: 2]
        outputs = []
        attns = []

        # 分尺度处理 / Process each scale
        for i, context_size in enumerate(self.context_sizes):
            # 投影到对应尺度的特征空间
            q = self.w1[i](query)  # [B, T, n_feat]
            v = self.w3[i](value)  # [B, T, n_feat]
            
            # 生成注意力权重
            weight = self.w2[i](torch.relu(q))  # [B, T, heads_per_scale * context_size]
            weight = weight.view(bs * time, self.heads_per_scale, 1, context_size)

            # 分块处理value
            value_cw = chunkwise(v, (context_size - 1) // 2, 1, (context_size - 1) // 2)  # [B, T, context_size, D]

            # 重新组织维度
            B, T, C, D = value_cw.shape
            # Bug修复：不应该再次调用w3，v已经是w3变换后的结果了
            # 原代码错误地调用了两次w3，导致FLOPs虚高
            # value_cw = value_cw.reshape(B * T * C, D)  # [B*T*C, D]
            # value_cw = self.w3[i](value_cw)  # [B*T*C, n_feat] ← BUG: 重复调用w3

            # 计算每个头的维度
            head_dim = D // self.heads_per_scale
            value_cw = value_cw.view(B * T, C, self.heads_per_scale, head_dim)  # [B*T, C, H, D/H]
            value_cw = value_cw.transpose(1, 2)  # [B*T, H, C, D/H]
            
            # 计算注意力
            attn = torch.softmax(weight, dim=-1)
            attns.append(attn)
            p_attn = self.dropout(attn)
            
            # 注意力加权求和
            x = torch.matmul(p_attn, value_cw)  # [bs*time, heads_per_scale, 1, d_k]
            x = x.squeeze(2).view(bs, time, -1)  # [B, T, heads_per_scale*d_k]
            outputs.append(x)
        
        # 存储注意力权重
        self.attn = attns
        
        # 加权融合不同尺度的输出
        scale_weights = F.softmax(self.scale_weights, dim=0)
        x = torch.zeros_like(query)  # [B, T, n_feat]
        for i, output in enumerate(outputs):
            x += scale_weights[i] * output
            
        x = self.w_out(x)  # [B, T, n_feat]
        return x 



