import torch
from torch import nn
import torch.nn.functional as F
import config as cfg
from Model.T_FAC import DynamicConv2D
'''这是空间移位模块，经过这一模块，输入输出Shape不变'''


class Spatial_shift(nn.Module):
    """改进的空间移位注意力模块

    特点：
    1. 时域动态卷积(TDY)
    2. 频域动态卷积(FDY)
    3. 空间移位增强
    4. 特征融合

    Args:
        in_channel (int): 输入通道数
        n (int): 移位分组数
    """

    def __init__(self, in_channel, n=cfg.MODEL_CONFIG['spatial_shift']['n']):
        super().__init__()
        self.in_channel = in_channel
        self.n = n

        # 使用组卷积扩展通道数
        self.group_conv = nn.Conv2d(in_channel, in_channel * 3, kernel_size=1, groups=in_channel)

        # 时域动态卷积
        self.tdy_conv = DynamicConv2D(
            in_channel, in_channel,
            kernel_size=3, pool_dim='time'
        )

        # 频域动态卷积
        self.fdy_conv = DynamicConv2D(
            in_channel, in_channel,
            kernel_size=3, pool_dim='freq'
        )

        # 使用1x1卷积进行特征融合和变换
        self.conv1x1 = nn.Conv2d(in_channel * 3, in_channel * 3, kernel_size=1)
        # 全连接层用于计算softmax权重
        self.linear = nn.Linear(in_channel * 3, in_channel * 3)

        # 添加dropout层
        self.dropout = nn.Dropout(cfg.MODEL_CONFIG['spatial_shift']['dropout'])

    def forward(self, x):  # x shape: [bs, in_channel, freq, time]
        """
        空间移位注意力的前向传播
        
        Shape变化过程：
        1. 输入 x: [bs, in_channel, freq, time]
        2. 组卷积后: [bs, in_channel*3, freq, time]
        3. 分块后: 每块 [bs, in_channel, freq, time]
        4. 动态卷积后: x1,x2 各为 [bs, in_channel, freq, time]
        5. 空间移位后: S1,S2,S3 各为 [bs, in_channel, freq, time]
        6. 拼接后: [bs, in_channel*3, freq, time]
        7. 特征融合后: [bs, in_channel*3, freq, time]
        8. 全局池化后: [bs, in_channel*3]
        9. 最终输出: [bs, in_channel, freq, time]
        """
        # 通过组卷积扩展通道
        x = self.group_conv(x)  # [bs, in_channel*3, freq, time]
        x1, x2, x3 = torch.chunk(x, chunks=3, dim=1)  # 每块 [bs, in_channel, freq, time]

        # 应用动态卷积
        x1 = self.tdy_conv(x1)  # [bs, in_channel, freq, time]
        x2 = self.fdy_conv(x2)  # [bs, in_channel, freq, time]

        # 空间移位操作
        S1 = spatial_shift(x1, self.n)  # [bs, in_channel, freq, time]
        S2 = spatial_shift(x2.transpose(2, 3), self.n).transpose(2, 3)
        S3 = x3

        # 特征融合
        combined = torch.cat((S1, S2, S3), dim=1)  # [bs, in_channel*3, freq, time]
        fused = self.conv1x1(combined)  # [bs, in_channel*3, freq, time]

        # 特征融合后添加dropout
        fused = self.dropout(fused)

        # 全局池化和权重计算
        U = F.adaptive_avg_pool2d(fused, (1, 1)).view(fused.size(0), -1)  # [bs, in_channel*3]
        weights = F.softmax(self.linear(U), dim=1)  # [bs, in_channel*3]

        # 权重分配和加权求和
        a1, a2, a3 = weights.split(self.in_channel, dim=1)  # 每个 [bs, in_channel]
        Xout = (a1.unsqueeze(-1).unsqueeze(-1) * S1 +
                a2.unsqueeze(-1).unsqueeze(-1) * S2 +
                a3.unsqueeze(-1).unsqueeze(-1) * S3)  # [bs, in_channel, freq, time]

        return Xout


def spatial_shift(x, n):
    """
    这是空间移位函数,输入数据分为四个通道，分别进行时间、频率维的移位，最后拼接在一起

    """
    # 移除 global 声明，使用局部变量以支持多GPU训练
    C = x.shape[1]
    k = C // n
    X_shift = torch.zeros_like(x)
    # 将X依据通道数划为四部分
    for i in range(n):
        if i % 2 == 0:
            X = x[:, i * k:(i + 1) * k, :, :]
            X_shift_i = torch.roll(X, shifts=1, dims=2)
            X_shift_i[:, :, 0, :] = 0
        else:
            X = x[:, i * k:(i + 1) * k, :, :]
            X_shift_i = torch.roll(X, shifts=-1, dims=2)
            X_shift_i[:, :, -1, :] = 0
        X_shift[:, i * k:(i + 1) * k, :, :] = X_shift_i

    return X_shift