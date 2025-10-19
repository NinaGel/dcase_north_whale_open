import torch
import torch.nn as nn
import config_dcase as cfg


class ConformerBlock(nn.Module):
    """Conformer模块 / Conformer Block

    结合自注意力和卷积的混合架构
    Hybrid architecture combining self-attention and convolution

    结构 / Structure: FFN -> MHSA -> Conv -> FFN
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

        # 前馈模块1 / Feed-forward module 1
        self.ff1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(dim * ff_mult, dim)
        )

        # 多头自注意力 / Multi-head self-attention
        self.attn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=heads,
                dropout=attn_dropout,
                batch_first=True
            )
        )

        # 卷积模块 / Convolution module
        self.conv = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * conv_expansion_factor),
            nn.GELU(),
            nn.Dropout(conv_dropout),

            # 深度卷积 / Depthwise conv
            nn.Conv1d(
                dim * conv_expansion_factor,
                dim * conv_expansion_factor,
                conv_kernel_size,
                padding = conv_kernel_size // 2,
                groups = dim * conv_expansion_factor
            ),
            nn.BatchNorm1d(dim * conv_expansion_factor),
            nn.GELU(),

            # 点卷积 / Pointwise conv
            nn.Conv1d(dim * conv_expansion_factor, dim, 1),
            nn.Dropout(conv_dropout)
        )

        # 前馈模块2 / Feed-forward module 2
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
        # 第一个前馈模块
        x = self.ff1(x) * 0.5 + x

        # 多头自注意力
        attn_norm = self.attn[0](x)  # LayerNorm
        attn_out, _ = self.attn[1](attn_norm, attn_norm, attn_norm)
        x = attn_out + x

        # 卷积模块
        conv_norm = self.conv[0](x)  # LayerNorm
        conv_lin = self.conv[1:4](conv_norm)  # Linear + GELU + Dropout
        B, T, C = conv_lin.shape
        conv_lin = conv_lin.transpose(1, 2)  # [B, C, T]
        conv_depth = self.conv[4:8](conv_lin)  # Depthwise Conv + BN + GELU
        conv_point = self.conv[8:](conv_depth)  # Pointwise Conv + Dropout
        conv_out = conv_point.transpose(1, 2)  # [B, T, C]
        x = conv_out + x

        # 第二个前馈模块
        x = self.ff2(x) * 0.5 + x

        # 最终归一化
        return self.norm(x)


class Conformer(nn.Module):
    """Conformer模型实现 (DCASE版本)

    结合卷积和Transformer的架构，专门用于音频处理
    适配DCASE2020数据集 (10类, 16kHz采样率)
    """
    def __init__(self, num_classes):
        super().__init__()

        # 获取DCASE特征维度
        self.freq_dim = cfg.DCASE_AUDIO_CONFIG['freq']  # 512

        # 特征提取
        self.features = nn.Sequential(
            # 第一次降维 (512 -> 128)
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((4, 1)),

            # 第二次降维 (128 -> 32)
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((4, 1))
        )

        # 特征维度: 32 * (512 // 16) = 32 * 32 = 1024
        self.d_model = 32 * (self.freq_dim // 16)

        # 相对位置编码 (使用DCASE的frame数: 311)
        self.pos_embedding = nn.Parameter(torch.randn(1, cfg.DCASE_AUDIO_CONFIG['frame'], self.d_model))

        # Conformer块 (4层)
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                dim=self.d_model,
                dim_head=64,
                heads=cfg.DCASE_DDSA_CONFIG['n_head'],  # 8 heads
                ff_mult=4,
                conv_expansion_factor=2,
                conv_kernel_size=31,
                attn_dropout=cfg.DCASE_DDSA_CONFIG['dropout_rate'],  # 0.3
                ff_dropout=cfg.DCASE_DDSA_CONFIG['dropout_rate'],
                conv_dropout=cfg.DCASE_DDSA_CONFIG['dropout_rate']
            ) for _ in range(4)
        ])

        # 分类层 (10类DCASE事件)
        self.classifier = nn.Linear(self.d_model, num_classes)

    def forward(self, x):
        # x: [B, freq_dim, frame] = [B, 512, 311] 或 [B, 1, freq_dim, frame] = [B, 1, 512, 311]
        # 混合精度训练由autocast自动处理

        # 如果是3D，添加通道维度
        if x.dim() == 3:
            x = x.unsqueeze(1)  # [B, 1, 512, 311]
        # 如果已经是4D，直接使用

        x = self.features(x)  # [B, 32, 32, 311]
        x = x.permute(0, 3, 1, 2)  # [B, 311, 32, 32]
        x = x.reshape(x.size(0), x.size(1), -1)  # [B, 311, 1024]

        # 添加位置编码
        x = x + self.pos_embedding

        # Conformer处理
        for block in self.conformer_blocks:
            x = block(x)

        # 分类输出
        output = self.classifier(x)  # [B, 311, 10]
        return output


if __name__ == '__main__':
    """测试Conformer_DCASE形状"""
    print("=== Testing Conformer_DCASE ===")

    # 创建模型
    model = Conformer(num_classes=10)
    print(f"Model created with d_model={model.d_model}")

    # 测试输入
    batch_size = 4
    dummy_input = torch.randn(batch_size, cfg.DCASE_AUDIO_CONFIG['freq'], cfg.DCASE_AUDIO_CONFIG['frame'])
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"  - Batch: {batch_size}")
    print(f"  - Freq bins: {cfg.DCASE_AUDIO_CONFIG['freq']}")
    print(f"  - Time frames: {cfg.DCASE_AUDIO_CONFIG['frame']}")

    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    print(f"\nOutput shape: {output.shape}")
    print(f"  - Expected: [{batch_size}, {cfg.DCASE_AUDIO_CONFIG['frame']}, 10]")

    # 验证形状
    expected_shape = (batch_size, cfg.DCASE_AUDIO_CONFIG['frame'], 10)
    assert output.shape == expected_shape, f"Shape mismatch! Got {output.shape}, expected {expected_shape}"

    print("\n✅ Conformer DCASE test passed!")

    # 参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  - Total: {total_params:,}")
    print(f"  - Trainable: {trainable_params:,}")
