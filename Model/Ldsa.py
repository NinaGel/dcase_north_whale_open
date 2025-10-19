import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def chunkwise(xs, N_l, N_c, N_r):
    """Slice input frames chunk by chunk.

    Args:
        xs (FloatTensor): `[B, T, input_dim]`
        N_l (int): number of frames for left context
        N_c (int): number of frames for current context
        N_r (int): number of frames for right context
    Returns:
        xs (FloatTensor): `[B * n_chunks, N_l + N_c + N_r, input_dim]`
            where n_chunks = ceil(T / N_c)
    """
    bs, xmax, idim = xs.size()
    n_chunks = math.ceil(xmax / N_c)
    c = N_l + N_c + N_r
    s_index = torch.arange(0, xmax, N_c).unsqueeze(-1)  #
    c_index = torch.arange(0, c)
    index = s_index + c_index
    xs_pad = torch.cat([xs.new_zeros(bs, N_l, idim),
                        xs,
                        xs.new_zeros(bs, N_c * n_chunks - xmax + N_r, idim)], dim=1)
    xs_chunk = xs_pad[:, index].contiguous().view(bs * n_chunks, N_l + N_c + N_r, idim)
    return xs_chunk


class LocalDenseSynthesizerAttention(nn.Module):
    """局部密集合成注意力 / Local Dense Synthesizer Attention

    不使用点积计算注意力，而是通过前馈网络直接生成注意力权重
    Generates attention weights via feedforward network instead of dot-product

    Args:
        n_head: 注意力头数 / number of attention heads
        n_feat: 特征维度 / feature dimension
        dropout_rate: dropout比率 / dropout rate
        context_size: 上下文窗口大小 / context window size
        use_bias: 是否使用偏置 / whether to use bias
    """

    def __init__(self, n_head, n_feat, dropout_rate, context_size=45, use_bias=False):
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.c = context_size
        self.w1 = nn.Linear(n_feat, n_feat, bias=use_bias)   # w1: (in_features = n_feat = 频率×通道 = 64×64 =4096, out_features = n_feat =4096)
        self.w2 = nn.Linear(n_feat, n_head * context_size, bias=use_bias) # w2: (in_features = n_feat = 4096, out_features = n_head * context_size = 8 * 15 = 120)
        self.w3 = nn.Linear(n_feat, n_feat, bias=use_bias) # w3: (in_features = n_feat = 4096, out_features = n_feat = 4096)
        self.w_out = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, query, key, value):
        """Forward pass.

        :param torch.Tensor query: (batch, time, size)
        :param torch.Tensor key: (batch, time, size) dummy
        :param torch.Tensor value: (batch, time, size)
        :return torch.Tensor: attentioned and transformed `value` (batch, time, d_model)
        """
        # 输入 query/key/value shape: [64, 309, 512]
        bs, time = query.size()[: 2]  # bs=64, time=309
        
        # 1. Query变换
        query = self.w1(query)  # [64, 309, 512]
        
        # 2. Weight计算
        # self.w2: 输入[64, 309, 512] -> 输出[64, 309, 8*45]
        weight = self.w2(torch.relu(query))
        # 重塑维度: [64, 309, 360] -> [19776, 8, 1, 45]
        c = int(self.c)
        weight = weight.view(bs * time, self.h, 1, c)
        
        # 3. Value变换
        value = self.w3(value)  # [64, 309, 512]
        
        # 4. Value分块处理
        # chunkwise参数: N_l=22, N_c=1, N_r=22 (因为(45-1)/2=22)
        # value_cw shape演变:
        # -> chunkwise输出: [19776, 45, 512]  (19776 = 64*309)
        # -> view: [19776, 45, 8, 64]
        # -> transpose: [19776, 8, 45, 64]
        value_cw = chunkwise(value, (c - 1) // 2, 1, (c - 1) // 2) \
            .view(bs * time, c, self.h, self.d_k).transpose(1, 2)
        
        # 5. 注意力权重
        self.attn = torch.softmax(weight, dim=-1)  # [19776, 8, 1, 45]
        p_attn = self.dropout(self.attn)
        
        # 6. 注意力计算
        # matmul: [19776, 8, 1, 45] × [19776, 8, 45, 64] -> [19776, 8, 1, 64]
        x = torch.matmul(p_attn, value_cw)
        
        # 7. 输出处理
        # view: [19776, 8, 1, 64] -> [64, 309, 512]
        x = x.contiguous().view(bs, -1, self.h * self.d_k)
        
        # 8. 最终输出投影
        x = self.w_out(x)  # [64, 309, 512]
        
        return x
