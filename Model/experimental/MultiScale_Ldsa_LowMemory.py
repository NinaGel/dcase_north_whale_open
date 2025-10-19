"""
Low Memory MultiScale LDSA - 低内存多尺度LDSA

在不改变模型参数的情况下优化激活内存:
1. 顺序处理各尺度，立即释放中间张量
2. 避免不必要的tensor复制
3. 使用原地操作
4. 复用缓冲区

与原始MultiScaleLDSA完全相同的参数量和输出，但激活内存更低
"""

import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleLDSA_LowMemory(nn.Module):
    """低内存版本的多尺度局部密集合成注意力

    与原始MultiScaleLDSA完全相同的:
    - 参数量
    - 模型结构
    - 输出结果

    不同的是:
    - 激活内存更低 (通过顺序处理和及时释放)
    """

    def __init__(self, n_head, n_feat, dropout_rate, context_sizes=[21, 41, 61, 81], use_bias=False):
        super().__init__()
        assert n_feat % n_head == 0
        assert n_head % len(context_sizes) == 0

        self.d_k = n_feat // n_head
        self.h = n_head
        self.context_sizes = context_sizes
        self.heads_per_scale = n_head // len(context_sizes)
        self.n_feat = n_feat

        # 与原始MultiScaleLDSA完全相同的参数结构
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
        self.attn = None

    def _chunkwise_efficient(self, xs, N_l, N_r):
        """内存优化的分块函数 - 使用unfold避免循环"""
        bs, xmax, idim = xs.size()
        c = N_l + 1 + N_r  # context size

        # 填充
        xs_pad = F.pad(xs, (0, 0, N_l, N_r), mode='constant', value=0)

        # 使用unfold - 比循环更高效
        # xs_pad: [B, T+N_l+N_r, D]
        xs_chunk = xs_pad.unfold(1, c, 1)  # [B, T, D, c]
        xs_chunk = xs_chunk.transpose(2, 3)  # [B, T, c, D]

        return xs_chunk

    def _process_scale_lowmem(self, query, value, scale_idx):
        """低内存处理单个尺度"""
        context_size = self.context_sizes[scale_idx]
        bs, time, _ = query.size()

        # Step 1: 投影 query
        q = self.w1[scale_idx](query)  # [B, T, n_feat]

        # Step 2: 生成注意力权重
        weight = self.w2[scale_idx](torch.relu(q))  # [B, T, heads_per_scale * context_size]
        del q  # 立即释放

        weight = weight.view(bs * time, self.heads_per_scale, 1, context_size)
        attn = F.softmax(weight, dim=-1)
        del weight  # 立即释放

        attn = self.dropout(attn)

        # Step 3: 投影 value 并分块
        v = self.w3[scale_idx](value)  # [B, T, n_feat]

        # 分块
        N_l = (context_size - 1) // 2
        N_r = (context_size - 1) // 2
        value_cw = self._chunkwise_efficient(v, N_l, N_r)  # [B, T, context_size, n_feat]
        del v  # 立即释放

        # Step 4: 重组维度并计算注意力
        B, T, C, D = value_cw.shape
        head_dim = D // self.heads_per_scale

        # 使用reshape而不是view（处理非连续张量）
        value_cw = value_cw.reshape(B * T, C, self.heads_per_scale, head_dim)
        value_cw = value_cw.permute(0, 2, 1, 3)  # [B*T, H, C, D/H]

        # Step 5: 注意力加权求和
        x = torch.matmul(attn, value_cw)  # [bs*time, heads_per_scale, 1, d_k]
        del attn, value_cw  # 立即释放

        x = x.squeeze(2).reshape(bs, time, -1)  # [B, T, heads_per_scale*d_k]

        return x

    def forward(self, query, key, value):
        bs, time = query.size()[: 2]

        # 获取融合权重
        scale_weights = F.softmax(self.scale_weights, dim=0)

        # 累积输出 - 使用原地加法
        output = torch.zeros(bs, time, self.n_feat, device=query.device, dtype=query.dtype)

        # 顺序处理每个尺度，立即累加结果
        for i in range(len(self.context_sizes)):
            scale_output = self._process_scale_lowmem(query, value, i)

            # 原地加权累加
            output.add_(scale_weights[i] * scale_output)

            # 立即释放当前尺度的输出
            del scale_output

        # 输出投影
        output = self.w_out(output)

        return output


def replace_ldsa_with_lowmem(model):
    """
    将模型中的MultiScaleLDSA替换为低内存版本

    用法:
        model = Whale_Model_Attention_MultiScale()
        replace_ldsa_with_lowmem(model)
    """
    from Model.MultiScale_Ldsa import MultiScaleLDSA

    for name, module in model.named_modules():
        if isinstance(module, MultiScaleLDSA):
            # 获取原始参数
            n_head = module.h
            n_feat = module.w_out.in_features
            dropout_rate = module.dropout.p
            context_sizes = module.context_sizes
            use_bias = module.w1[0].bias is not None

            # 创建低内存版本
            new_module = MultiScaleLDSA_LowMemory(
                n_head=n_head,
                n_feat=n_feat,
                dropout_rate=dropout_rate,
                context_sizes=context_sizes,
                use_bias=use_bias
            )

            # 复制权重
            new_module.load_state_dict(module.state_dict())

            # 替换模块
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]

            if parent_name:
                parent = model.get_submodule(parent_name)
                setattr(parent, child_name, new_module)
            else:
                setattr(model, name, new_module)

            print(f"Replaced {name} with LowMemory version")

    return model


# 测试
if __name__ == '__main__':
    import gc

    print("="*70)
    print("MultiScaleLDSA 低内存版本测试")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 配置
    batch_size = 8
    time_steps = 309
    n_feat = 256
    n_head = 8
    context_sizes = [21, 41, 61, 81]  # 与原始相同

    x = torch.randn(batch_size, time_steps, n_feat, device=device)

    # 测试原始版本
    print("\n1. 原始 MultiScaleLDSA:")
    from MultiScale_Ldsa import MultiScaleLDSA

    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model_orig = MultiScaleLDSA(n_head, n_feat, 0.1, context_sizes).to(device)
    params_orig = sum(p.numel() for p in model_orig.parameters())

    with torch.no_grad():
        out_orig = model_orig(x, x, x)

    if device.type == 'cuda':
        peak_orig = torch.cuda.max_memory_allocated() / 1024**2
        print(f"   参数量: {params_orig:,}")
        print(f"   峰值GPU内存: {peak_orig:.1f} MiB")
    print(f"   输出形状: {out_orig.shape}")

    del model_orig

    # 测试低内存版本
    print("\n2. 低内存 MultiScaleLDSA_LowMemory:")

    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model_lowmem = MultiScaleLDSA_LowMemory(n_head, n_feat, 0.1, context_sizes).to(device)
    params_lowmem = sum(p.numel() for p in model_lowmem.parameters())

    with torch.no_grad():
        out_lowmem = model_lowmem(x, x, x)

    if device.type == 'cuda':
        peak_lowmem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"   参数量: {params_lowmem:,} (与原始相同)")
        print(f"   峰值GPU内存: {peak_lowmem:.1f} MiB")
        print(f"   内存减少: {(1 - peak_lowmem/peak_orig)*100:.1f}%")
    print(f"   输出形状: {out_lowmem.shape}")

    # 验证输出一致性
    print("\n3. 输出一致性验证:")

    # 重新创建两个模型并共享权重
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    model_orig = MultiScaleLDSA(n_head, n_feat, 0.1, context_sizes).to(device)
    model_lowmem = MultiScaleLDSA_LowMemory(n_head, n_feat, 0.1, context_sizes).to(device)

    # 复制权重
    model_lowmem.load_state_dict(model_orig.state_dict())

    model_orig.eval()
    model_lowmem.eval()

    with torch.no_grad():
        out1 = model_orig(x, x, x)
        out2 = model_lowmem(x, x, x)

    diff = (out1 - out2).abs().max().item()
    print(f"   最大输出差异: {diff:.2e}")
    print(f"   输出一致: {'是' if diff < 1e-5 else '否'}")

    print("\n" + "="*70)
    print("测试完成!")
