import torch
from torch import nn
from Model.BA_Conv import BSA_Conv
from Model.Spatial_shift_modified import Spatial_shift
from Model.MultiScale_Ldsa import MultiScaleLDSA
import config as cfg


class Whale_Model_Attention_MultiScale(nn.Module):
    """使用多尺度LDSA的声音事件检测模型"""

    def __init__(self):
        super().__init__()
        
        # 验证输入维度
        if cfg.AUDIO_CONFIG['freq'] & (cfg.AUDIO_CONFIG['freq'] - 1) != 0:
            raise ValueError(f"输入频率维度 {cfg.AUDIO_CONFIG['freq']} 必须是2的幂次")
        
        # 验证注意力头数
        n_feat = cfg.MODEL_CONFIG['attention']['n_feat']
        n_head = cfg.MODEL_CONFIG['attention']['n_head']
        if n_feat % n_head != 0:
            raise ValueError(f"特征维度 {n_feat} 必须能被注意力头数 {n_head} 整除")
        
        # 特征提取层
        self.bsa_conv1 = BSA_Conv(**cfg.MODEL_CONFIG['bsa_conv1'])
        self.spatial_shift = Spatial_shift(
            in_channel=cfg.MODEL_CONFIG['bsa_conv1']['c3_out']
        )
        self.bsa_conv2 = BSA_Conv(**cfg.MODEL_CONFIG['bsa_conv2'])
        
        # 使用多尺度LDSA
        self.ldsa = MultiScaleLDSA(
            n_head=cfg.MODEL_CONFIG['attention']['n_head'],
            n_feat=cfg.MODEL_CONFIG['feature_dims']['n_feat'],
            dropout_rate=cfg.MODEL_CONFIG['attention']['dropout'],
            context_sizes=cfg.MULTISCALE_CONFIG['context_sizes'],
            use_bias=cfg.LDSA_OPTIMIZED_CONFIG['use_bias']
        )
        # GRU层
        self.gru = nn.GRU(
            input_size=cfg.MODEL_CONFIG['attention']['n_feat'],
            hidden_size=cfg.MODEL_CONFIG['gru']['hidden_size'],
            num_layers=cfg.MODEL_CONFIG['gru']['num_layers'],
            batch_first=True,
            bidirectional=True,
            dropout=cfg.MODEL_CONFIG['gru']['dropout'] 
            if cfg.MODEL_CONFIG['gru']['num_layers'] > 1 else 0
        )
        
        # 分类层
        self.fc = nn.Linear(
            cfg.MODEL_CONFIG['gru']['hidden_size'] * 2,
            cfg.MODEL_CONFIG['num_classes']
        )

    def forward(self, x):
        # 特征提取
        x = self.bsa_conv1(x)  # [B, 8, H/3, W]
        x = self.spatial_shift(x)  # [B, 8, H/3, W]
        x = self.bsa_conv2(x)  # [B, 32, H/6, W]

        # 验证特征维度
        B, C, F, T = x.shape
        expected_freq = cfg.MODEL_CONFIG['feature_dims']['freq_dim']
        if F != expected_freq:
            raise ValueError(f"特征频率维度 {F} 与预期维度 {expected_freq} 不匹配")

        # 调整维度
        x = x.transpose(1, 2)  # [B, F, C, T]
        x = x.reshape(B, F * C, T)  # [B, F*C, T]
        x = x.transpose(1, 2)  # [B, T, F*C]

        # 多尺度LDSA注意力
        x = self.ldsa(x, x, x)  # [B, T, F*C]

        # GRU处理
        gru_out, _ = self.gru(x)  # [B, T, H*2]

        # 分类
        output = self.fc(gru_out)  # [B, T, num_classes]

        return output 