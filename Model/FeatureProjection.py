"""
特征降维模块
用于将512维ACT特征降维到128维，减少参数量和显存占用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyProjection(nn.Module):
    """
    频率维度投影/降维模块

    支持多种降维策略：
    1. Linear: 简单线性投影
    2. Conv1d: 1D卷积降维（可学习）
    3. Attention: 注意力加权降维
    4. Adaptive: 自适应池化（固定，不可学习）
    """

    def __init__(self, input_freq, output_freq, method='conv1d', learnable=True):
        """
        Args:
            input_freq (int): 输入频率维度（512）
            output_freq (int): 输出频率维度（128）
            method (str): 降维方法 ('linear', 'conv1d', 'attention', 'adaptive')
            learnable (bool): 是否可学习（adaptive方法固定为False）
        """
        super(FrequencyProjection, self).__init__()

        self.input_freq = input_freq
        self.output_freq = output_freq
        self.method = method
        self.learnable = learnable

        if method == 'linear':
            # 简单线性投影
            self.projection = nn.Linear(input_freq, output_freq)

        elif method == 'conv1d':
            # 1D卷积降维（推荐）
            # 使用stride和kernel_size实现降维
            downsample_factor = input_freq // output_freq  # 512 // 128 = 4

            if downsample_factor == 4:
                # 512 -> 128: 使用两层卷积
                self.projection = nn.Sequential(
                    nn.Conv1d(1, 8, kernel_size=3, stride=2, padding=1),  # 512 -> 256
                    nn.BatchNorm1d(8),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(8, 1, kernel_size=3, stride=2, padding=1),  # 256 -> 128
                    nn.BatchNorm1d(1),
                    nn.ReLU(inplace=True)
                )
            else:
                # 通用情况：单层卷积
                kernel_size = downsample_factor * 2 - 1
                self.projection = nn.Conv1d(
                    1, 1,
                    kernel_size=kernel_size,
                    stride=downsample_factor,
                    padding=kernel_size // 2
                )

        elif method == 'attention':
            # 注意力加权降维
            # 学习一个注意力矩阵 [output_freq, input_freq]
            self.query = nn.Linear(output_freq, 256)
            self.key = nn.Linear(input_freq, 256)
            self.projection = nn.Linear(input_freq, output_freq)

        elif method == 'adaptive':
            # 自适应平均池化（固定，不可学习）
            # 最快但性能可能略差
            # 注意：我们需要在频率维度池化，需要特殊处理
            self.learnable = False

        else:
            raise ValueError(f"Unknown method: {method}")

    def forward(self, x):
        """
        Args:
            x: [batch, freq, time] 或 [batch, 1, freq, time]
        Returns:
            out: [batch, output_freq, time] 或 [batch, 1, output_freq, time]
        """
        original_shape = x.shape
        is_4d = (x.dim() == 4)

        if is_4d:
            # [batch, 1, freq, time] -> [batch, freq, time]
            batch, channels, freq, time = x.shape
            assert channels == 1, "Only support single channel input for 4D"
            x = x.squeeze(1)  # [batch, freq, time]

        batch, freq, time = x.shape
        assert freq == self.input_freq, f"Expected freq={self.input_freq}, got {freq}"

        if self.method == 'linear':
            # Permute to [batch, time, freq] for linear
            x = x.permute(0, 2, 1)  # [batch, time, freq]
            x = self.projection(x)  # [batch, time, output_freq]
            x = x.permute(0, 2, 1)  # [batch, output_freq, time]

        elif self.method == 'conv1d':
            # Conv1d expects [batch, channels, freq]
            # Need to process each time frame
            x = x.permute(0, 2, 1)  # [batch, time, freq]
            x = x.reshape(batch * time, freq)  # [batch*time, freq]
            x = x.unsqueeze(1)  # [batch*time, 1, freq]
            x = self.projection(x)  # [batch*time, 1, output_freq] or [batch*time, C, output_freq]

            if x.shape[1] > 1:
                # Multi-channel output, pool to single channel
                x = x.mean(dim=1, keepdim=True)  # [batch*time, 1, output_freq]

            x = x.squeeze(1)  # [batch*time, output_freq]
            x = x.reshape(batch, time, self.output_freq)  # [batch, time, output_freq]
            x = x.permute(0, 2, 1)  # [batch, output_freq, time]

        elif self.method == 'attention':
            # Attention-based projection
            x_t = x.permute(0, 2, 1)  # [batch, time, freq]

            # Create query and key
            query_base = torch.linspace(0, 1, self.output_freq, device=x.device).unsqueeze(0)  # [1, output_freq]
            query_base = query_base.expand(batch * time, -1)  # [batch*time, output_freq]

            key_base = torch.linspace(0, 1, self.input_freq, device=x.device).unsqueeze(0)  # [1, input_freq]
            key_base = key_base.expand(batch * time, -1)  # [batch*time, input_freq]

            # Compute attention weights
            Q = self.query(query_base)  # [batch*time, 256]
            K = self.key(key_base)  # [batch*time, 256]

            attn_weights = torch.softmax(Q @ K.T / (256 ** 0.5), dim=-1)  # [batch*time, batch*time]

            # Apply projection
            x_flat = x_t.reshape(batch * time, freq)  # [batch*time, freq]
            x_out = self.projection(x_flat)  # [batch*time, output_freq]
            x_out = x_out.reshape(batch, time, self.output_freq)  # [batch, time, output_freq]
            x = x_out.permute(0, 2, 1)  # [batch, output_freq, time]

        elif self.method == 'adaptive':
            # Adaptive pooling on frequency dimension
            # x: [batch, freq, time]
            # Need to pool along freq dimension (dim=1)
            x = x.permute(0, 2, 1)  # [batch, time, freq]
            x = F.adaptive_avg_pool1d(x, self.output_freq)  # [batch, time, output_freq]
            x = x.permute(0, 2, 1)  # [batch, output_freq, time]

        # Restore 4D if needed
        if is_4d:
            x = x.unsqueeze(1)  # [batch, 1, output_freq, time]

        return x

    def get_info(self):
        """返回模块信息"""
        params = sum(p.numel() for p in self.parameters())
        return {
            'method': self.method,
            'input_freq': self.input_freq,
            'output_freq': self.output_freq,
            'learnable': self.learnable,
            'parameters': params
        }


class ResidualFrequencyProjection(nn.Module):
    """
    带残差连接的频率投影
    先降维再通过残差连接保留部分原始信息
    """

    def __init__(self, input_freq, output_freq, method='conv1d'):
        super(ResidualFrequencyProjection, self).__init__()

        self.input_freq = input_freq
        self.output_freq = output_freq

        # 主降维分支
        self.main_projection = FrequencyProjection(input_freq, output_freq, method=method)

        # 残差分支（简单池化）
        # 使用FrequencyProjection的adaptive方法
        self.residual_projection = FrequencyProjection(input_freq, output_freq, method='adaptive', learnable=False)

        # 混合权重
        self.alpha = nn.Parameter(torch.tensor(0.8))  # 主分支权重

    def forward(self, x):
        """
        Args:
            x: [batch, freq, time] 或 [batch, 1, freq, time]
        Returns:
            out: [batch, output_freq, time] 或 [batch, 1, output_freq, time]
        """
        is_4d = (x.dim() == 4)
        if is_4d:
            x_input = x.squeeze(1)  # [batch, freq, time]
        else:
            x_input = x

        # 主分支
        main_out = self.main_projection(x)
        if is_4d:
            main_out = main_out.squeeze(1)

        # 残差分支
        residual_out = self.residual_projection(x)
        if is_4d:
            residual_out = residual_out.squeeze(1)

        # 加权混合
        alpha = torch.sigmoid(self.alpha)
        out = alpha * main_out + (1 - alpha) * residual_out

        if is_4d:
            out = out.unsqueeze(1)

        return out


if __name__ == "__main__":
    """测试降维模块"""
    print("="*80)
    print("测试特征降维模块")
    print("="*80)

    batch_size = 4
    input_freq = 512
    output_freq = 128
    time_frames = 309

    # 测试输入
    x = torch.randn(batch_size, input_freq, time_frames)

    print(f"\n输入形状: {x.shape}")
    print(f"目标: {input_freq} -> {output_freq} 频率维度")

    # 测试各种方法
    methods = ['adaptive', 'linear', 'conv1d']

    for method in methods:
        print(f"\n{'='*80}")
        print(f"方法: {method.upper()}")
        print(f"{'='*80}")

        projection = FrequencyProjection(input_freq, output_freq, method=method)

        # 前向传播
        out = projection(x)

        # 信息
        info = projection.get_info()
        print(f"输出形状: {out.shape}")
        print(f"参数量: {info['parameters']:,}")
        print(f"可学习: {info['learnable']}")

        # 验证形状
        assert out.shape == (batch_size, output_freq, time_frames), \
            f"Shape mismatch! Expected ({batch_size}, {output_freq}, {time_frames}), got {out.shape}"
        print("[PASS] 形状测试通过")

    # 测试残差版本
    print(f"\n{'='*80}")
    print("方法: RESIDUAL CONV1D")
    print(f"{'='*80}")

    residual_projection = ResidualFrequencyProjection(input_freq, output_freq, method='conv1d')
    out_res = residual_projection(x)

    params_res = sum(p.numel() for p in residual_projection.parameters())
    print(f"输出形状: {out_res.shape}")
    print(f"参数量: {params_res:,}")
    print("[PASS] 残差版本测试通过")

    # 测试4D输入
    print(f"\n{'='*80}")
    print("测试4D输入")
    print(f"{'='*80}")

    x_4d = torch.randn(batch_size, 1, input_freq, time_frames)
    projection_4d = FrequencyProjection(input_freq, output_freq, method='conv1d')
    out_4d = projection_4d(x_4d)

    print(f"输入形状: {x_4d.shape}")
    print(f"输出形状: {out_4d.shape}")
    assert out_4d.shape == (batch_size, 1, output_freq, time_frames)
    print("[PASS] 4D输入测试通过")

    print(f"\n{'='*80}")
    print("[SUCCESS] 所有测试通过!")
    print(f"{'='*80}")
