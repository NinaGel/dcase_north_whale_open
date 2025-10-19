import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseSynthesizerAttention(nn.Module):
    """Dense Synthesizer attention 模块
    
    Args:
        n_head (int): 注意力头数
        n_feat (int): 特征维度
        dropout_rate (float): dropout率
        use_bias (bool): 是否使用偏置
    """
    def __init__(self, n_head, n_feat, dropout_rate, use_bias=False):
        super().__init__()
        assert n_feat % n_head == 0
        # 每个头的维度
        self.d_k = n_feat // n_head
        self.h = n_head
        
        # 三个线性变换
        self.w1 = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.w2 = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.w_out = nn.Linear(n_feat, n_feat, bias=use_bias)
        
        self.dropout = nn.Dropout(p=dropout_rate)
        self.attn = None

    def forward(self, query, key, value, mask=None):
        """前向传播
        
        Args:
            query (torch.Tensor): [B, T, d]
            key (torch.Tensor): [B, T, d] 
            value (torch.Tensor): [B, T, d]
            mask (torch.Tensor): [B, T, T]
            
        Returns:
            torch.Tensor: [B, T, d]
        """
        batch_size, time = query.size()[:2]
        
        # 特征变换
        query = self.w1(query)  # [B, T, d]
        
        # 直接合成注意力权重
        attn_weights = self.w2(F.relu(query))  # [B, T, d]
        attn_weights = attn_weights.view(batch_size, time, self.h, self.d_k)
        attn_weights = attn_weights.transpose(1, 2)  # [B, h, T, d_k]
        
        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, T, T] 
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(attn_weights, dim=-1)
        self.attn = attn_weights
        
        # dropout
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        value = value.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)  # [B, h, T, d_k]
        x = torch.matmul(attn_weights, value)  # [B, h, T, d_k]
        
        # 拼接多头的结果
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)  # [B, T, d]
        
        # 输出变换
        return self.w_out(x)  # [B, T, d] 