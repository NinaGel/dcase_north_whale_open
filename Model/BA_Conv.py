import torch
from torch import nn
from torch.nn import functional as F


class BSA_Conv(nn.Module):
    """双分支非对称卷积模块 / Bi-branch Asymmetric Convolution Module

    包含1×3和3×1两个分支，通过3×3卷积融合，支持残差连接
    Two branches (1×3 and 3×1 kernels), fused by 3×3 conv, with residual connection

    Args:
        in_channel: 输入通道数 / input channels
        c1_out: 1×3分支输出通道 / 1×3 branch output channels
        c2_out: 3×1分支输出通道 / 3×1 branch output channels
        c3_out: 融合后输出通道 / fused output channels
        use_1x1conv: 是否使用1×1残差 / use 1×1 residual connection
        strides: 步长 / stride
        avg_kernel_size: 池化核大小 / pooling kernel size
        debug: 调试模式 / debug mode
    """

    def __init__(self, in_channel, c1_out, c2_out, c3_out, use_1x1conv=False, strides=1, avg_kernel_size=(2, 1),
                 debug=False):
        super().__init__()
        # 分支1: 1×3卷积 / Branch 1: 1×3 conv
        self.conv1 = nn.Conv2d(in_channel, c1_out, kernel_size=(1, 3), padding=(0, 1), stride=strides)
        # 分支2: 3×1卷积 / Branch 2: 3×1 conv
        self.conv2 = nn.Conv2d(in_channel, c2_out, kernel_size=(3, 1), padding=(1, 0), stride=strides)
        self.conv3 = nn.Conv2d(c1_out, c3_out, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv4 = nn.Conv2d(in_channel, c3_out, kernel_size=1, stride=strides, padding=0)
        else:
            self.conv4 = None
        self.bn1 = nn.BatchNorm2d(c3_out)
        self.avg1 = nn.AvgPool2d(kernel_size=avg_kernel_size)
        self.drop = nn.Dropout(p=0.3)
        self.debug = debug

    def forward(self, x):
        outputs = {}
        # X = X.unsqueeze(1)
        P1 = self.conv1(x)
        if self.debug:
            outputs['conv1'] = P1

        P2 = self.conv2(x)
        if self.debug:
            outputs['conv2'] = P2

        P3 = self.conv3(P1)
        if self.debug:
            outputs['conv3'] = P3

        P4 = self.conv3(P2)
        if self.debug:
            outputs['conv4'] = P4
        # Y = self.conv3(torch.cat((P1, P2), dim=1))
        # if self.debug:
        #     outputs['conv3'] = Y

        Y = self.bn1(P3 + P4)
        if self.debug:
            outputs['conv5'] = Y

        if self.conv4:
            x = self.conv4(x)
            if self.debug:
                outputs['conv6'] = x

        Y = x + Y if self.conv4 else Y  # 残差连接 / residual connection
        Y = F.relu(Y)
        Y = self.avg1(Y)
        final_output = self.drop(Y)
        if self.debug:
            outputs['final_output'] = final_output
            return outputs
        else:
            return final_output
