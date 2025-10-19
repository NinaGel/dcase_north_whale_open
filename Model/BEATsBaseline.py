"""
BEATs Baseline 模型

使用预提取的 BEATs 特征作为输入的下游分类网络。
BEATs encoder 已冻结，只训练下游网络。

输入: [batch, 768, 311] (预提取的 BEATs 特征，转置后)
输出: [batch, 311, num_classes] (帧级预测)

架构:
    BEATs 特征 -> Linear -> LayerNorm -> ReLU -> Dropout
                    -> BiGRU
                    -> Linear -> 输出

参考文献:
    BEATs: Audio Pre-Training with Acoustic Tokenizers
    https://arxiv.org/abs/2212.09058
"""

import torch
import torch.nn as nn


class BEATsBaseline(nn.Module):
    """
    BEATs Baseline 模型

    使用预提取的 BEATs 特征进行声音事件检测。
    这是一个纯下游网络，不包含 BEATs encoder。

    Args:
        num_classes (int): 输出类别数，默认 10 (DCASE)
        beats_dim (int): BEATs 特征维度，默认 768
        hidden_dim (int): 隐藏层维度，默认 256
        gru_layers (int): GRU 层数，默认 2
        dropout (float): Dropout 比例，默认 0.3
    """

    def __init__(
        self,
        num_classes: int = 10,
        beats_dim: int = 768,
        hidden_dim: int = 256,
        gru_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.beats_dim = beats_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # 特征投影层: 768 -> 256
        self.projection = nn.Sequential(
            nn.Linear(beats_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # BiGRU 时序建模
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        # 输出层: hidden_dim * 2 (双向) -> num_classes
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes),
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        nn.init.xavier_uniform_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: BEATs 特征张量
               支持多种输入格式:
               - [batch, beats_dim, time_frames]: 转置格式 (推荐)
               - [batch, time_frames, beats_dim]: 序列格式
               - [batch, 1, beats_dim, time_frames]: 4D 格式

        Returns:
            output: [batch, time_frames, num_classes]
        """
        # 处理不同的输入格式
        if x.dim() == 4:
            # [batch, 1, beats_dim, time_frames] -> [batch, beats_dim, time_frames]
            x = x.squeeze(1)

        if x.dim() == 3:
            # 判断哪个维度是 beats_dim (768)
            if x.shape[1] == self.beats_dim:
                # [batch, beats_dim, time_frames] -> [batch, time_frames, beats_dim]
                x = x.transpose(1, 2)
            # 否则假设已经是 [batch, time_frames, beats_dim]

        # x: [batch, time_frames, beats_dim]
        batch_size, time_frames, _ = x.shape

        # 特征投影: [batch, time_frames, hidden_dim]
        x = self.projection(x)

        # BiGRU: [batch, time_frames, hidden_dim * 2]
        x, _ = self.gru(x)

        # 分类: [batch, time_frames, num_classes]
        output = self.classifier(x)

        return output

    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_name": "BEATsBaseline",
            "total_params": total_params,
            "trainable_params": trainable_params,
            "input_shape": f"[batch, {self.beats_dim}, time_frames]",
            "output_shape": f"[batch, time_frames, {self.num_classes}]",
            "beats_dim": self.beats_dim,
            "hidden_dim": self.hidden_dim,
            "architecture": "Linear -> LayerNorm -> ReLU -> BiGRU -> Linear",
        }


class BEATsBaselineMLP(nn.Module):
    """
    BEATs Baseline 模型 (MLP 版本)

    更简单的 MLP 架构，不使用 RNN。
    适合快速实验或对比。

    Args:
        num_classes (int): 输出类别数
        beats_dim (int): BEATs 特征维度
        hidden_dims (list): 隐藏层维度列表
        dropout (float): Dropout 比例
    """

    def __init__(
        self,
        num_classes: int = 10,
        beats_dim: int = 768,
        hidden_dims: list = None,
        dropout: float = 0.3,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.beats_dim = beats_dim
        self.num_classes = num_classes

        # 构建 MLP 层
        layers = []
        in_dim = beats_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, num_classes))

        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: BEATs 特征 [batch, beats_dim, time_frames] 或 [batch, time_frames, beats_dim]

        Returns:
            output: [batch, time_frames, num_classes]
        """
        # 处理输入格式
        if x.dim() == 4:
            x = x.squeeze(1)

        if x.dim() == 3 and x.shape[1] == self.beats_dim:
            x = x.transpose(1, 2)

        # x: [batch, time_frames, beats_dim]
        output = self.mlp(x)

        return output

    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_name": "BEATsBaselineMLP",
            "total_params": total_params,
            "trainable_params": trainable_params,
            "input_shape": f"[batch, {self.beats_dim}, time_frames]",
            "output_shape": f"[batch, time_frames, {self.num_classes}]",
        }


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("BEATsBaseline 模型测试")
    print("=" * 70)

    # 创建模型
    model = BEATsBaseline(num_classes=10, beats_dim=768, hidden_dim=256)
    print(f"\n模型信息:")
    info = model.get_model_info()
    for k, v in info.items():
        print(f"  {k}: {v}")

    # 测试不同输入格式
    print("\n" + "-" * 70)
    print("测试不同输入格式:")

    # 格式 1: [batch, beats_dim, time_frames]
    x1 = torch.randn(4, 768, 311)
    out1 = model(x1)
    print(f"  输入 [4, 768, 311] -> 输出 {list(out1.shape)}")

    # 格式 2: [batch, time_frames, beats_dim]
    x2 = torch.randn(4, 311, 768)
    out2 = model(x2)
    print(f"  输入 [4, 311, 768] -> 输出 {list(out2.shape)}")

    # 格式 3: [batch, 1, beats_dim, time_frames]
    x3 = torch.randn(4, 1, 768, 311)
    out3 = model(x3)
    print(f"  输入 [4, 1, 768, 311] -> 输出 {list(out3.shape)}")

    # 测试 MLP 版本
    print("\n" + "-" * 70)
    print("BEATsBaselineMLP 模型测试:")

    model_mlp = BEATsBaselineMLP(num_classes=10)
    info_mlp = model_mlp.get_model_info()
    for k, v in info_mlp.items():
        print(f"  {k}: {v}")

    x = torch.randn(4, 768, 311)
    out = model_mlp(x)
    print(f"\n  输入 [4, 768, 311] -> 输出 {list(out.shape)}")

    print("\n" + "=" * 70)
    print("[OK] 测试通过!")
    print("=" * 70)
