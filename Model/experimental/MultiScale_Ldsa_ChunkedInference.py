"""
Chunked Inference MultiScale LDSA - 分块推理版本

通过时间维度分块处理来大幅降低激活内存:
- 将长序列分成小块处理
- 每块处理完立即释放内存
- 参数量完全不变

理论上可以将激活内存降低到 O(chunk_size) 而不是 O(sequence_length)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleLDSA_ChunkedInference(nn.Module):
    """分块推理的多尺度LDSA

    通过时间维度分块来降低峰值内存:
    - 推理时将序列分成多个chunk处理
    - 每个chunk独立计算注意力
    - 使用边界padding确保输出正确

    参数量与原始MultiScaleLDSA完全相同
    """

    def __init__(self, n_head, n_feat, dropout_rate,
                 context_sizes=[31, 61], use_bias=False,
                 inference_chunk_size=64):
        super().__init__()
        assert n_feat % n_head == 0
        assert n_head % len(context_sizes) == 0

        self.d_k = n_feat // n_head
        self.h = n_head
        self.context_sizes = context_sizes
        self.heads_per_scale = n_head // len(context_sizes)
        self.n_feat = n_feat
        self.inference_chunk_size = inference_chunk_size
        self.max_context = max(context_sizes)

        # 与原始完全相同的参数
        self.w1 = nn.ModuleList([
            nn.Linear(n_feat, n_feat, bias=use_bias)
            for _ in range(len(context_sizes))
        ])

        self.w2 = nn.ModuleList([
            nn.Linear(n_feat, self.heads_per_scale * context_size, bias=use_bias)
            for context_size in context_sizes
        ])

        self.w3 = nn.ModuleList([
            nn.Linear(n_feat, n_feat, bias=use_bias)
            for _ in range(len(context_sizes))
        ])

        self.scale_weights = nn.Parameter(
            torch.ones(len(context_sizes)) / len(context_sizes)
        )

        self.w_out = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.dropout = nn.Dropout(p=dropout_rate)

    def _process_chunk(self, query_chunk, value_padded, chunk_start, chunk_end, scale_idx):
        """处理单个时间块"""
        context_size = self.context_sizes[scale_idx]
        bs = query_chunk.size(0)
        chunk_len = chunk_end - chunk_start

        # 投影query
        q = self.w1[scale_idx](query_chunk)
        weight = self.w2[scale_idx](torch.relu(q))
        del q

        weight = weight.view(bs * chunk_len, self.heads_per_scale, 1, context_size)
        attn = F.softmax(weight, dim=-1)
        del weight
        attn = self.dropout(attn)

        # 提取需要的value区域 (包含context)
        half_ctx = (context_size - 1) // 2
        v_start = chunk_start  # value_padded已经添加了padding
        v_end = chunk_start + chunk_len + context_size - 1

        v_slice = value_padded[:, v_start:v_end, :]
        v = self.w3[scale_idx](v_slice)
        del v_slice

        # 使用unfold创建滑动窗口
        v_unfolded = v.unfold(1, context_size, 1)  # [B, chunk_len, D, context_size]
        v_unfolded = v_unfolded.transpose(2, 3)    # [B, chunk_len, context_size, D]
        del v

        # 重组维度
        B, T, C, D = v_unfolded.shape
        head_dim = D // self.heads_per_scale
        v_unfolded = v_unfolded.reshape(B * T, C, self.heads_per_scale, head_dim)
        v_unfolded = v_unfolded.permute(0, 2, 1, 3)  # [B*T, H, C, D/H]

        # 注意力计算
        out = torch.matmul(attn, v_unfolded)
        del attn, v_unfolded

        out = out.squeeze(2).reshape(bs, chunk_len, -1)
        return out

    def forward(self, query, key, value):
        bs, time, feat = query.size()

        if not self.training and time > self.inference_chunk_size:
            # 推理时使用分块处理
            return self._forward_chunked(query, key, value)
        else:
            # 训练时使用标准处理（保持梯度计算正确）
            return self._forward_standard(query, key, value)

    def _forward_standard(self, query, key, value):
        """标准前向传播（用于训练）"""
        bs, time, _ = query.size()

        scale_weights = F.softmax(self.scale_weights, dim=0)
        output = torch.zeros(bs, time, self.n_feat, device=query.device, dtype=query.dtype)

        for i, context_size in enumerate(self.context_sizes):
            # 投影
            q = self.w1[i](query)
            v = self.w3[i](value)

            # 生成注意力权重
            weight = self.w2[i](torch.relu(q))
            weight = weight.view(bs * time, self.heads_per_scale, 1, context_size)
            attn = F.softmax(weight, dim=-1)
            attn = self.dropout(attn)

            # 分块value
            half_ctx = (context_size - 1) // 2
            v_pad = F.pad(v, (0, 0, half_ctx, half_ctx), mode='constant', value=0)
            v_unfolded = v_pad.unfold(1, context_size, 1).transpose(2, 3)

            B, T, C, D = v_unfolded.shape
            head_dim = D // self.heads_per_scale
            v_unfolded = v_unfolded.reshape(B * T, C, self.heads_per_scale, head_dim)
            v_unfolded = v_unfolded.permute(0, 2, 1, 3)

            out = torch.matmul(attn, v_unfolded)
            out = out.squeeze(2).reshape(bs, time, -1)

            output = output + scale_weights[i] * out

        return self.w_out(output)

    def _forward_chunked(self, query, key, value):
        """分块前向传播（用于推理，低内存）"""
        bs, time, _ = query.size()
        chunk_size = self.inference_chunk_size

        scale_weights = F.softmax(self.scale_weights, dim=0)

        # 预分配输出
        output = torch.zeros(bs, time, self.n_feat, device=query.device, dtype=query.dtype)

        # 对每个尺度分别处理
        for scale_idx, context_size in enumerate(self.context_sizes):
            half_ctx = (context_size - 1) // 2

            # 预先padding value（只做一次）
            v_padded = F.pad(value, (0, 0, half_ctx, half_ctx), mode='constant', value=0)

            # 分块处理
            for chunk_start in range(0, time, chunk_size):
                chunk_end = min(chunk_start + chunk_size, time)

                # 提取当前chunk的query
                query_chunk = query[:, chunk_start:chunk_end, :]

                # 处理当前chunk
                chunk_out = self._process_chunk(
                    query_chunk, v_padded, chunk_start, chunk_end, scale_idx
                )

                # 累加到输出
                output[:, chunk_start:chunk_end, :] += scale_weights[scale_idx] * chunk_out

                # 释放中间结果
                del chunk_out, query_chunk

            del v_padded

        return self.w_out(output)


def create_chunked_ldsa(original_ldsa, inference_chunk_size=64):
    """从原始LDSA创建分块推理版本

    用法:
        from Model.MultiScale_Ldsa import MultiScaleLDSA
        original = MultiScaleLDSA(...)
        chunked = create_chunked_ldsa(original, inference_chunk_size=64)
    """
    chunked = MultiScaleLDSA_ChunkedInference(
        n_head=original_ldsa.h,
        n_feat=original_ldsa.w_out.in_features,
        dropout_rate=original_ldsa.dropout.p,
        context_sizes=original_ldsa.context_sizes,
        use_bias=original_ldsa.w1[0].bias is not None,
        inference_chunk_size=inference_chunk_size
    )

    # 复制权重
    chunked.load_state_dict(original_ldsa.state_dict())

    return chunked


if __name__ == '__main__':
    import gc

    print("="*70)
    print("分块推理 MultiScaleLDSA 测试")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 8
    time_steps = 309
    n_feat = 256
    n_head = 8
    context_sizes = [31, 61]

    x = torch.randn(batch_size, time_steps, n_feat, device=device)

    # 测试原始版本
    print("\n1. 原始 MultiScaleLDSA:")
    from MultiScale_Ldsa import MultiScaleLDSA

    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model_orig = MultiScaleLDSA(n_head, n_feat, 0.1, context_sizes).to(device)
    model_orig.eval()

    with torch.no_grad():
        out_orig = model_orig(x, x, x)

    if device.type == 'cuda':
        peak_orig = torch.cuda.max_memory_allocated() / 1024**2
        print(f"   峰值GPU内存: {peak_orig:.1f} MiB")
    print(f"   参数量: {sum(p.numel() for p in model_orig.parameters()):,}")

    # 测试分块推理版本 (chunk_size=64)
    for chunk_size in [128, 64, 32]:
        print(f"\n2. 分块推理版本 (chunk_size={chunk_size}):")

        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        model_chunked = MultiScaleLDSA_ChunkedInference(
            n_head, n_feat, 0.1, context_sizes,
            inference_chunk_size=chunk_size
        ).to(device)
        model_chunked.eval()

        # 复制权重
        model_chunked.load_state_dict(model_orig.state_dict())

        with torch.no_grad():
            out_chunked = model_chunked(x, x, x)

        if device.type == 'cuda':
            peak_chunked = torch.cuda.max_memory_allocated() / 1024**2
            print(f"   峰值GPU内存: {peak_chunked:.1f} MiB")
            print(f"   内存减少: {(1 - peak_chunked/peak_orig)*100:.1f}%")
        print(f"   参数量: {sum(p.numel() for p in model_chunked.parameters()):,}")

        # 验证输出一致性
        diff = (out_orig - out_chunked).abs().max().item()
        print(f"   输出差异: {diff:.2e}")

    print("\n" + "="*70)
