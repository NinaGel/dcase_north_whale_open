import torch
import torch.nn as nn
import config_dcase as cfg


class ConformerBlock(nn.Module):
    """Conformer块实现

    包含:
    1. Feed Forward Module
    2. Multi-Head Self Attention Module
    3. Convolution Module
    4. Feed Forward Module
    """
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.0,
        ff_dropout = 0.0,
        conv_dropout = 0.0
    ):
        super().__init__()

        # 第一个前馈模块
        self.ff1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(dim * ff_mult, dim)
        )

        # 多头自注意力模块
        self.attn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=heads,
                dropout=attn_dropout,
                batch_first=True
            )
        )

        # 卷积模块
        self.conv = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * conv_expansion_factor),
            nn.GELU(),
            nn.Dropout(conv_dropout),

            # 深度可分离卷积
            nn.Conv1d(
                dim * conv_expansion_factor,
                dim * conv_expansion_factor,
                conv_kernel_size,
                padding = conv_kernel_size // 2,
                groups = dim * conv_expansion_factor
            ),
            nn.BatchNorm1d(dim * conv_expansion_factor),
            nn.GELU(),

            # 点卷积
            nn.Conv1d(dim * conv_expansion_factor, dim, 1),
            nn.Dropout(conv_dropout)
        )

        # 第二个前馈模块
        self.ff2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(dim * ff_mult, dim)
        )

        # 最终层归一化
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # 第一个前馈模块，使用缩放残差连接
        x = self.ff1(x) * 0.3 + x

        # 多头自注意力模块
        attn_norm = self.attn[0](x)  # LayerNorm
        attn_out, _ = self.attn[1](attn_norm, attn_norm, attn_norm)
        x = attn_out * 0.7 + x

        # 卷积模块
        conv_norm = self.conv[0](x)  # LayerNorm
        conv_lin = self.conv[1:4](conv_norm)  # Linear + GELU + Dropout
        B, T, C = conv_lin.shape
        conv_lin = conv_lin.transpose(1, 2)  # [B, C, T]
        conv_depth = self.conv[4:8](conv_lin)  # Depthwise Conv + BN + GELU
        conv_point = self.conv[8:](conv_depth)  # Pointwise Conv + Dropout
        conv_out = conv_point.transpose(1, 2)  # [B, T, C]
        x = conv_out * 0.7 + x

        # 第二个前馈模块，使用缩放残差连接
        x = self.ff2(x) * 0.3 + x

        # 最终归一化
        return self.norm(x)


class ConformerDCASE_Optimized(nn.Module):
    """优化的Conformer模型（论文配置）

    参考论文: Barahona et al., IEEE/ACM TASLP 2024
    关键配置:
    - d_model = 144 (论文配置)
    - 3 个 Conformer 块
    - 4 个 attention heads
    - 7 层 CNN 特征提取器
    - CNN层使用Dropout2d进行正则化
    - Conformer块使用较高的dropout率
    - 采用缩放残差连接

    预期参数量: ~4.2M (相比原来的101M减少96%)
    """
    def __init__(self, num_classes=10, input_freq=None, input_frame=None):
        super().__init__()

        # 获取特征维度
        self.freq_dim = input_freq if input_freq is not None else cfg.DCASE_AUDIO_CONFIG['freq']
        self.frame_dim = input_frame if input_frame is not None else cfg.DCASE_AUDIO_CONFIG['frame']

        # 目标d_model (论文配置)
        self.d_model = 144

        # 7层CNN特征提取器（参考论文Section III.A）
        # 目标：将 [B, 1, 512, 311] 压缩到 d_model=144
        # 设计原则：使用足够的通道数，使CNN参数量达到~2.5M
        self.features = nn.Sequential(
            # Layer 1: 初始特征提取
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d((2, 1)),  # 512 → 256 (freq)

            # Layer 2: 扩展通道
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d((2, 1)),  # 256 → 128 (freq)

            # Layer 3: 进一步扩展
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d((2, 1)),  # 128 → 64 (freq)

            # Layer 4: 保持256通道
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d((2, 1)),  # 64 → 32 (freq)

            # Layer 5: 扩展到320通道
            nn.Conv2d(256, 320, kernel_size=3, padding=1),
            nn.BatchNorm2d(320),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d((2, 1)),  # 32 → 16 (freq)

            # Layer 6: 降回256通道
            nn.Conv2d(320, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d((2, 1)),  # 16 → 8 (freq)

            # Layer 7: 降维到d_model=144
            nn.Conv2d(256, 144, kernel_size=1),  # Pointwise conv
            nn.BatchNorm2d(144),
            nn.ReLU(),
            nn.Dropout2d(0.3)
        )

        # CNN输出: [B, 144, 8, 311]
        # 在频率维度做平均池化: [B, 144, 1, 311] -> [B, 144, 311]
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))  # 保持时间维度，压缩频率维度

        # 位置编码（使用DCASE的frame数: 311）
        self.pos_embedding = nn.Parameter(torch.randn(1, self.frame_dim, self.d_model))

        # 3个Conformer块（论文配置）
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                dim=self.d_model,
                dim_head=36,  # 144 / 4 heads = 36
                heads=4,  # 论文使用4个heads
                ff_mult=4,
                conv_expansion_factor=2,
                conv_kernel_size=31,
                attn_dropout=0.5,
                ff_dropout=0.5,
                conv_dropout=0.5
            ) for _ in range(3)  # 3个块（论文配置）
        ])

        # 分类层 (10类DCASE事件)
        self.classifier = nn.Linear(self.d_model, num_classes)

    def forward(self, x):
        # x: [B, freq_dim, frame] = [B, 512, 311] 或 [B, 1, freq_dim, frame]
        # 混合精度训练由autocast自动处理

        # 如果是3D，添加通道维度
        if x.dim() == 3:
            x = x.unsqueeze(1)  # [B, 1, 512, 311]

        # CNN特征提取: [B, 1, 512, 311] -> [B, 144, 8, 311]
        x = self.features(x)

        # 频率维度池化: [B, 144, 8, 311] -> [B, 144, 1, 311]
        x = self.freq_pool(x)

        # 压缩频率维度并转置: [B, 144, 1, 311] -> [B, 144, 311] -> [B, 311, 144]
        x = x.squeeze(2).transpose(1, 2)

        # 添加位置编码: [B, 311, 144]
        x = x + self.pos_embedding

        # Conformer处理: 3个块
        for block in self.conformer_blocks:
            x = block(x)

        # 分类输出: [B, 311, 144] -> [B, 311, 10]
        output = self.classifier(x)

        return output


if __name__ == '__main__':
    """测试优化版Conformer_DCASE"""
    print("=" * 80)
    print("测试优化版 Conformer_DCASE (d_model=144, 3 blocks)")
    print("=" * 80)

    # 创建模型
    model = ConformerDCASE_Optimized(num_classes=10)
    print(f"\n模型配置:")
    print(f"  d_model: {model.d_model}")
    print(f"  Conformer块数: {len(model.conformer_blocks)}")
    print(f"  Attention heads: 4")
    print(f"  CNN通道数: 1→64→128→256→256→320→256→144")

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n参数统计:")
    print(f"  总参数量:     {total_params:>12,} ({total_params/1e6:.3f}M)")
    print(f"  可训练参数:   {trainable_params:>12,} ({trainable_params/1e6:.3f}M)")

    # 各模块参数量
    print(f"\n模块参数分布:")
    cnn_params = sum(p.numel() for p in model.features.parameters())
    freq_pool_params = sum(p.numel() for p in model.freq_pool.parameters()) if hasattr(model.freq_pool, 'parameters') else 0
    pos_params = model.pos_embedding.numel()
    conformer_params = sum(p.numel() for p in model.conformer_blocks.parameters())
    classifier_params = sum(p.numel() for p in model.classifier.parameters())

    print(f"  CNN特征提取:  {cnn_params:>12,} ({cnn_params/1e6:.3f}M) - {cnn_params/total_params*100:.1f}%")
    if freq_pool_params > 0:
        print(f"  频率池化:     {freq_pool_params:>12,} ({freq_pool_params/1e6:.3f}M) - {freq_pool_params/total_params*100:.1f}%")
    print(f"  位置编码:     {pos_params:>12,} ({pos_params/1e6:.3f}M) - {pos_params/total_params*100:.1f}%")
    print(f"  Conformer块:  {conformer_params:>12,} ({conformer_params/1e6:.3f}M) - {conformer_params/total_params*100:.1f}%")
    print(f"  分类层:       {classifier_params:>12,} ({classifier_params/1e6:.3f}M) - {classifier_params/total_params*100:.1f}%")

    # 测试前向传播
    print(f"\n前向传播测试:")
    batch_size = 4

    # 测试3D输入
    dummy_input_3d = torch.randn(batch_size, cfg.DCASE_AUDIO_CONFIG['freq'],
                                  cfg.DCASE_AUDIO_CONFIG['frame'])
    print(f"  输入形状 (3D):  {dummy_input_3d.shape}")

    model.eval()
    with torch.no_grad():
        output = model(dummy_input_3d)

    print(f"  输出形状:       {output.shape}")
    expected_shape = (batch_size, cfg.DCASE_AUDIO_CONFIG['frame'], 10)
    print(f"  期望形状:       {expected_shape}")

    # 验证形状
    assert output.shape == expected_shape, f"Shape mismatch! Got {output.shape}, expected {expected_shape}"
    print(f"  ✅ 形状验证通过!")

    # 测试4D输入
    dummy_input_4d = torch.randn(batch_size, 1, cfg.DCASE_AUDIO_CONFIG['freq'],
                                  cfg.DCASE_AUDIO_CONFIG['frame'])
    with torch.no_grad():
        output_4d = model(dummy_input_4d)
    assert output_4d.shape == expected_shape, "4D input shape mismatch!"
    print(f"  ✅ 4D输入兼容性验证通过!")

    # 与目标对比
    print(f"\n与论文对比:")
    print(f"  论文报告参数量: 4.2M")
    print(f"  当前实现参数量: {total_params/1e6:.2f}M")
    print(f"  差异:           {abs(total_params/1e6 - 4.2):.2f}M ({abs(total_params/1e6 - 4.2)/4.2*100:.1f}%)")

    if abs(total_params/1e6 - 4.2) < 1.0:
        print(f"  ✅ 参数量在合理范围内!")
    else:
        print(f"  ⚠️  参数量可能需要进一步调整")

    # 与原始Conformer对比
    print(f"\n与原始Conformer_DCASE对比:")
    print(f"  原始参数量:     101.386M")
    print(f"  优化参数量:     {total_params/1e6:.3f}M")
    print(f"  减少比例:       {(1 - total_params/101.386e6)*100:.1f}%")
    print(f"  压缩倍数:       {101.386e6/total_params:.1f}x")

    print("\n" + "=" * 80)
    print("✅ 优化版Conformer测试完成!")
    print("=" * 80)
