import torch
from torch import nn
import torch.nn.functional as F
import config as cfg


class DynamicConv2D(nn.Module):
    """时域/频域动态卷积模块

    Args:
        in_channel (int): 输入通道数
        out_channel (int): 输出通道数
        kernel_size (int): 卷积核大小
        stride (int): 步长
        padding (int): 填充
        n_basis_kernels (int): 基础核数量
        temperature (float): 软化因子
        pool_dim (str): 池化维度 ('time'或'freq')
    """

    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1,
                 n_basis_kernels=4, temperature=31, pool_dim='freq'):
        super().__init__()
        self.in_planes = in_channel
        self.out_planes = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pool_dim = pool_dim
        self.n_basis_kernels = n_basis_kernels
        self.temperature = temperature

        # 注意力模块
        self.attention = DynamicAttention2D(
            in_channel, kernel_size, stride, padding,
            n_basis_kernels, temperature, pool_dim
        )

        # 多尺度动态核
        self.weight_small = nn.Parameter(
            torch.randn(n_basis_kernels, out_channel, in_channel, 3, 3)
        )
        self.weight_medium = nn.Parameter(
            torch.randn(n_basis_kernels, out_channel, in_channel, 5, 5)
        )
        
        # 初始化权重
        nn.init.kaiming_normal_(self.weight_small)
        nn.init.kaiming_normal_(self.weight_medium)

        # 残差连接
        self.shortcut = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        
        # 批归一化和激活函数
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # x shape: [bs, chan, freq, time]
        """
        动态卷积的前向传播

        Shape变化过程：
        1. 输入 x: [bs, chan, freq, time]
        2. attention: [bs, n_basis_kernels, freq/time]
        3. aggregate_weight: [n_basis_kernels * out_planes, in_planes, kernel_size, kernel_size]
        4. conv2d后: [bs, n_basis_kernels * out_planes, freq', time']
        5. 重塑后: [bs, n_basis_kernels, out_planes, freq', time']
        6. 最终输出: [bs, out_planes, freq', time']
        """
        batch_size = x.size(0)
        
        # 获取注意力权重
        attention = self.attention(x)  # [bs, n_basis_kernels, freq/time]
        
        # 处理小核和中核
        def process_kernel(weight, padding):
            aggregate_weight = weight.view(-1, self.in_planes, weight.size(-2), weight.size(-1))
            output = F.conv2d(x, aggregate_weight, None, self.stride, padding)
            output = output.view(batch_size, self.n_basis_kernels, self.out_planes, 
                               output.size(-2), output.size(-1))
            return output
            
        output_small = process_kernel(self.weight_small, 1)
        output_medium = process_kernel(self.weight_medium, 2)
        
        # 调整attention维度以匹配输出
        if self.pool_dim == 'time':
            attention = attention.unsqueeze(-1)  # [bs, n_basis_kernels, freq, 1]
        else:  # freq
            attention = attention.unsqueeze(2)  # [bs, n_basis_kernels, 1, time]
            
        # 应用注意力权重并求和
        output_small = torch.sum(output_small * attention.unsqueeze(2), dim=1)
        output_medium = torch.sum(output_medium * attention.unsqueeze(2), dim=1)
        
        # 加权融合不同尺度的输出
        output = output_small + output_medium
        
        # 残差连接
        shortcut = self.shortcut(x)
        output = output + shortcut
        
        # 归一化和激活
        output = self.bn(output)
        output = self.relu(output)
        
        return output  # [bs, out_planes, freq', time']


class DynamicAttention2D(nn.Module):
    """改进的动态注意力模块"""

    def __init__(self, in_planes, kernel_size, stride, padding,
                 n_basis_kernels, temperature, pool_dim):
        super().__init__()
        self.pool_dim = pool_dim
        self.temperature = temperature
        
        hidden_planes = max(in_planes // 4, 4)
        
        # 多尺度特征提取
        self.conv1d1_small = nn.Conv1d(in_planes, hidden_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1d1_medium = nn.Conv1d(in_planes, hidden_planes, kernel_size=3, stride=1, padding=1, bias=False)
        
        # BatchNorm和激活函数
        self.bn_small = nn.BatchNorm1d(hidden_planes)
        self.bn_medium = nn.BatchNorm1d(hidden_planes)
        self.relu = nn.ReLU(inplace=True)
        
        # 特征融合
        self.fusion_conv = nn.Conv1d(hidden_planes * 2, hidden_planes, kernel_size=1, bias=False)
        self.bn_fusion = nn.BatchNorm1d(hidden_planes)
        
        # 输出层
        self.conv1d2 = nn.Conv1d(hidden_planes, n_basis_kernels, kernel_size=1, bias=True)
        
        # 初始化参数
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):  # x shape: [bs, chan, freq, time]
        """
        动态注意力的前向传播

        Shape变化过程：
        1. 输入 x: [bs, chan, freq, time]
        2. 池化后: [bs, chan, time/freq]
        3. 多尺度特征提取: [bs, hidden_planes, time/freq]
        4. 特征融合: [bs, hidden_planes, time/freq]
        5. 输出层: [bs, n_basis_kernels, time/freq]
        """
        if self.pool_dim == 'freq':
            x = torch.mean(x, dim=2)  # [bs, chan, time]
        elif self.pool_dim == 'time':
            x = torch.mean(x, dim=3)  # [bs, chan, freq]

        # 多尺度特征提取
        feat_small = self.conv1d1_small(x)
        feat_small = self.bn_small(feat_small)
        feat_small = self.relu(feat_small)
        
        feat_medium = self.conv1d1_medium(x)
        feat_medium = self.bn_medium(feat_medium)
        feat_medium = self.relu(feat_medium)
        
        # 特征融合
        x = torch.cat([feat_small, feat_medium], dim=1)
        x = self.fusion_conv(x)
        x = self.bn_fusion(x)
        x = self.relu(x)
        
        # 生成注意力权重
        x = self.conv1d2(x)
        
        return F.softmax(x / self.temperature, dim=1)  # [bs, n_basis_kernels, time/freq]
