import sys
import os
import time
import logging
import torch
import matplotlib
matplotlib.use('Agg')  # 设置 matplotlib 后端为 Agg，避免 tkinter 相关错误
import torch.nn as nn
import torch.backends.cudnn as cudnn
from pathlib import Path
import argparse
from torch.nn import functional as F
from torch.optim import Adam
from typing import List, Dict
import numpy as np
import torch.cuda.amp as amp

# 添加项目根目录到 Python 路径（在导入本地模块前执行）
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model.MultiScale_Ldsa import MultiScaleLDSA
from Model.Whale_Model_Attention_MultiScale_Ldsa import Whale_Model_Attention_MultiScale
from Model.FAF_Filt import FAF_Filt_Model
from Train.train_utils import BaseTrainer
from evaluators.base_evaluator import BaseEvaluator
import config as cfg
from Data.audio_dataset import load_train_val_data

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CRNN(nn.Module):
    """CRNN基线模型
    
    结合CNN和RNN的经典声音事件检测架构
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # 设置混合精度训练标志
        self.use_mixed_precision = cfg.TRAIN_CONFIG.get('mixed_precision', False)
        
        # 获取特征维度
        self.freq_dim = cfg.AUDIO_CONFIG['freq']  # 从配置文件获取频率维度
        
        # CNN特征提取 - 降低通道数和特征维度
        self.features = nn.Sequential(
            # 第一次降维 (freq_dim, frame) -> (freq_dim/4, frame)
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((4, 1)),  # 频率维度降采样4倍
            
            # 第二次降维 (freq_dim/4, frame) -> (freq_dim/16, frame)
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((4, 1))  # 频率维度再降采样4倍
        )
        
        # 计算GRU输入维度
        self.gru_input_size = 32 * (self.freq_dim // 16)  # 32通道 * 降采样后的频率维度
        
        # BiGRU层 - 使用配置文件中的参数
        self.gru = nn.GRU(
            input_size=self.gru_input_size,
            hidden_size=cfg.MODEL_CONFIG['gru']['hidden_size'],
            num_layers=cfg.MODEL_CONFIG['gru']['num_layers'],
            batch_first=True,
            bidirectional=True,
            dropout=cfg.MODEL_CONFIG['gru']['dropout']
        )
        
        # 分类层
        self.classifier = nn.Linear(cfg.MODEL_CONFIG['gru']['hidden_size'] * 2, num_classes)

    def forward(self, x):
        # 确保输入维度正确
        if x.dim() == 3:  # [batch_size, freq, time]
            x = x.unsqueeze(1)  # 添加通道维度 [batch_size, channel, freq, time]
            
        # 适配混合精度训练
        if self.training and self.use_mixed_precision:
            with amp.autocast():
                x = self.features(x)  # [B, 32, freq_dim/16, frame]
                x = x.permute(0, 3, 1, 2)  # [B, frame, 32, freq_dim/16]
                x = x.reshape(x.size(0), x.size(1), -1)  # [B, frame, 32 * (freq_dim/16)]
                x, _ = self.gru(x)
                output = self.classifier(x)
        else:
            x = self.features(x)  # [B, 32, freq_dim/16, frame]
            x = x.permute(0, 3, 1, 2)  # [B, frame, 32, freq_dim/16]
            x = x.reshape(x.size(0), x.size(1), -1)  # [B, frame, 32 * (freq_dim/16)]
            x, _ = self.gru(x)
            output = self.classifier(x)
            
        return output


class CNN_Transformer(nn.Module):
    """CNN-Transformer模型
    
    使用CNN进行特征提取，Transformer进行序列建模
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # 设置混合精度训练标志
        self.use_mixed_precision = cfg.TRAIN_CONFIG.get('mixed_precision', False)
        
        # 获取特征维度
        self.freq_dim = cfg.AUDIO_CONFIG['freq']
        
        # CNN特征提取
        self.features = nn.Sequential(
            # 第一次降维
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((4, 1)),
            
            # 第二次降维
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((4, 1))
        )
        
        # 特征维度
        self.d_model = 32 * (self.freq_dim // 16)  # 32 channels * 降采样后的频率维度
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=cfg.MODEL_CONFIG['attention']['n_head'],
            dim_feedforward=1024,
            dropout=cfg.MODEL_CONFIG['attention']['dropout'],
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # 分类层
        self.classifier = nn.Linear(self.d_model, num_classes)

    def forward(self, x):
        # 确保输入维度正确
        if x.dim() == 3:  # [batch_size, freq, time]
            x = x.unsqueeze(1)  # 添加通道维度 [batch_size, channel, freq, time]
        elif x.dim() == 5:  # [batch_size, group, channel, freq, time]
            x = x.squeeze(2)  # 移除多余的维度
            
        # 适配混合精度训练
        if self.training and self.use_mixed_precision:
            with amp.autocast():
                x = self.features(x)  # [B, 32, freq_dim/16, frame]
                x = x.permute(0, 3, 1, 2)  # [B, frame, 32, freq_dim/16]
                x = x.reshape(x.size(0), x.size(1), -1)  # [B, frame, d_model]
                x = self.transformer(x)
                output = self.classifier(x)
        else:
            x = self.features(x)  # [B, 32, freq_dim/16, frame]
            x = x.permute(0, 3, 1, 2)  # [B, frame, 32, freq_dim/16]
            x = x.reshape(x.size(0), x.size(1), -1)  # [B, frame, d_model]
            x = self.transformer(x)
            output = self.classifier(x)
            
        return output


class Conformer(nn.Module):
    """Conformer模型实现
    
    结合卷积和Transformer的架构，专门用于音频处理
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # 设置混合精度训练标志
        self.use_mixed_precision = cfg.TRAIN_CONFIG.get('mixed_precision', False)
        
        # 获取特征维度
        self.freq_dim = cfg.AUDIO_CONFIG['freq']
        
        # 特征提取
        self.features = nn.Sequential(
            # 第一次降维
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((4, 1)),
            
            # 第二次降维
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((4, 1))
        )
        
        # 特征维度
        self.d_model = 32 * (self.freq_dim // 16)
        
        # 相对位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, cfg.AUDIO_CONFIG['frame'], self.d_model))
        
        # Conformer块
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                dim=self.d_model,
                dim_head=64,
                heads=cfg.MODEL_CONFIG['attention']['n_head'],
                ff_mult=4,
                conv_expansion_factor=2,
                conv_kernel_size=31,
                attn_dropout=cfg.MODEL_CONFIG['attention']['dropout'],
                ff_dropout=cfg.MODEL_CONFIG['attention']['dropout'],
                conv_dropout=cfg.MODEL_CONFIG['attention']['dropout']
            ) for _ in range(4)
        ])
        
        # 分类层
        self.classifier = nn.Linear(self.d_model, num_classes)

    def forward(self, x):
        # 确保输入维度正确
        if x.dim() == 3:  # [batch_size, freq, time]
            x = x.unsqueeze(1)  # 添加通道维度 [batch_size, channel, freq, time]
        elif x.dim() == 5:  # [batch_size, group, channel, freq, time]
            x = x.squeeze(2)  # 移除多余的维度
            
        # 适配混合精度训练
        if self.training and self.use_mixed_precision:
            with amp.autocast():
                x = self.features(x)  # [B, 32, freq_dim/16, frame]
                x = x.permute(0, 3, 1, 2)  # [B, frame, 32, freq_dim/16]
                x = x.reshape(x.size(0), x.size(1), -1)  # [B, frame, d_model]
                
                # 添加位置编码
                x = x + self.pos_embedding
                
                # Conformer处理
                for block in self.conformer_blocks:
                    x = block(x)
                    
                output = self.classifier(x)
        else:
            x = self.features(x)  # [B, 32, freq_dim/16, frame]
            x = x.permute(0, 3, 1, 2)  # [B, frame, 32, freq_dim/16]
            x = x.reshape(x.size(0), x.size(1), -1)  # [B, frame, d_model]
            
            # 添加位置编码
            x = x + self.pos_embedding
            
            # Conformer处理
            for block in self.conformer_blocks:
                x = block(x)
                
            output = self.classifier(x)
            
        return output


class PANNs(nn.Module):
    """PANNs模型实现
    
    基于预训练的音频神经网络，专门用于音频分类任务
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # 设置混合精度训练标志
        self.use_mixed_precision = cfg.TRAIN_CONFIG.get('mixed_precision', False)
        
        # 获取特征维度
        self.freq_dim = cfg.AUDIO_CONFIG['freq']
        
        # CNN特征提取器
        self.features = nn.Sequential(
            # 第一次降维
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((4, 1)),
            
            # 第二次降维
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((4, 1))
        )
        
        # 计算GRU输入维度
        self.gru_input_size = 32 * (self.freq_dim // 16)  # 32通道 * 降采样后的频率维度
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Conv2d(32, 32, 1),
            nn.Sigmoid()
        )
        
        # 时序处理
        self.gru = nn.GRU(
            input_size=self.gru_input_size,
            hidden_size=cfg.MODEL_CONFIG['gru']['hidden_size'],
            num_layers=cfg.MODEL_CONFIG['gru']['num_layers'],
            batch_first=True,
            bidirectional=True,
            dropout=cfg.MODEL_CONFIG['gru']['dropout']
        )
        
        # 分类层
        self.classifier = nn.Linear(cfg.MODEL_CONFIG['gru']['hidden_size'] * 2, num_classes)
        
        # 将模型移动到GPU
        self.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def forward(self, x):
        # 确保输入维度正确
        if x.dim() == 3:  # [batch_size, freq, time]
            x = x.unsqueeze(1)  # 添加通道维度 [batch_size, channel, freq, time]
        elif x.dim() == 5:  # [batch_size, group, channel, freq, time]
            x = x.squeeze(2)  # 移除多余的维度
            
        # 适配混合精度训练
        if self.training and self.use_mixed_precision:
            with amp.autocast():
                x = self.features(x)  # [B, 32, freq_dim/16, frame]
                
                # 应用注意力
                att = self.attention(x)
                x = x * att
                
                x = x.permute(0, 3, 1, 2)  # [B, frame, 32, freq_dim/16]
                x = x.reshape(x.size(0), x.size(1), -1)  # [B, frame, gru_input_size]
                
                x, _ = self.gru(x)
                output = self.classifier(x)
        else:
            x = self.features(x)  # [B, 32, freq_dim/16, frame]
            
            # 应用注意力
            att = self.attention(x)
            x = x * att
            
            x = x.permute(0, 3, 1, 2)  # [B, frame, 32, freq_dim/16]
            x = x.reshape(x.size(0), x.size(1), -1)  # [B, frame, gru_input_size]
            
            x, _ = self.gru(x)
            output = self.classifier(x)
            
        return output


class ConformerBlock(nn.Module):
    """Conformer块实现
    
    包含:
    1. Feed Forward Module
    2. Multi-Head Self Attention Module
    3. Convolution Module
    4. Feed Forward Module
    
    Args:
        dim: 输入维度
        dim_head: 注意力头的维度
        heads: 注意力头数
        ff_mult: 前馈网络扩展因子
        conv_expansion_factor: 卷积扩展因子
        conv_kernel_size: 卷积核大小
        attn_dropout: 注意力dropout率
        ff_dropout: 前馈网络dropout率
        conv_dropout: 卷积dropout率
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


def initialize_model(model_variant):
    """初始化指定的模型架构

    Args:
        model_variant (str): 模型变体名称
            - daap: DAAPNet模型
            - crnn: CRNN模型
            - cnn_transformer: CNN-Transformer模型
            - conformer: Conformer模型
            - panns: PANNs模型
            - faf: FAF-Filt模型

    Returns:
        nn.Module: 初始化的模型
    """
    if model_variant == 'daap':
        return Whale_Model_Attention_MultiScale()
    elif model_variant == 'crnn':
        return CRNN(cfg.MODEL_CONFIG['num_classes'])
    elif model_variant == 'cnn_transformer':
        return CNN_Transformer(cfg.MODEL_CONFIG['num_classes'])
    elif model_variant == 'conformer':
        return Conformer(cfg.MODEL_CONFIG['num_classes'])
    elif model_variant == 'panns':
        return PANNs(cfg.MODEL_CONFIG['num_classes'])
    elif model_variant == 'faf':
        # FAF-Filt Heavy模型
        # 基于ICASSP 2025论文 "FAF-Filt: Frequency-aware Fourier Filter for Sound Event Detection"
        conv_channels = [64, 128, 256, 256, 256]
        gru_hidden = 256
        return FAF_Filt_Model(
            num_classes=cfg.MODEL_CONFIG['num_classes'],
            input_freq_bins=cfg.AUDIO_CONFIG['freq'],
            conv_channels=conv_channels,
            gru_hidden=gru_hidden,
            gru_layers=2,
            reduction_ratio=4,
            use_projection=True,
            projection_method='conv1d',
            projection_target=128
        )
    else:
        raise ValueError(f"未知的模型架构: {model_variant}")


class ModelComparisonExperiment:
    """模型架构对比实验
    
    使用BaseEvaluator进行基础评估，记录和比较不同模型架构的性能。
    """
    
    def __init__(self, args, seed: int):
        """初始化实验
        
        Args:
            args: 命令行参数
            seed: 随机种子
        """
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 设置随机种子
        self._set_random_seed(seed)
            
        # 设置实验目录
        self.experiment_dir = cfg.EXPERIMENT_PATHS['model_comparison'] / f'seed_{seed}'
        self._setup_directories()
        
        # 设置模型变体
        self.available_variants = ['daap', 'crnn', 'cnn_transformer', 'conformer', 'panns', 'faf']

        # 根据命令行参数选择变体
        if hasattr(args, 'variant_group') and args.variant_group:
            self.variants = [args.variant_group]
        else:
            # 默认运行 daap 模型
            self.variants = ['daap']

        # 初始化模型字典
        self.models = self._initialize_models()
        
        # 初始化评估器
        self.evaluator = BaseEvaluator(self.experiment_dir)
        
        # 启用CUDA优化
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    def _set_random_seed(self, seed: int):
        """设置随机种子"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = cfg.GPU_CONFIG['cudnn_deterministic']
    
    def _setup_directories(self):
        """创建实验目录结构"""
        self.dirs = {}
        for name, subdir in cfg.EXPERIMENT_SUBDIRS.items():
            self.dirs[name] = self.experiment_dir / subdir
            self.dirs[name].mkdir(parents=True, exist_ok=True)
    
    def _initialize_models(self) -> dict:
        """初始化所有模型变体"""
        models = {}
        for variant in self.variants:
            models[variant] = initialize_model(variant)
            models[variant] = models[variant].to(self.device)
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                models[variant] = torch.nn.DataParallel(models[variant])
        return models
    
    def _train_variant(self, variant: str) -> dict:
        """训练单个模型变体

        Args:
            variant: 模型变体名称

        Returns:
            dict: 包含训练结果的字典
        """
        # 获取模型
        model = self.models[variant]

        # 记录模型基础指标
        try:
            dummy_input = torch.randn(1, 1, 256, 309, device=self.device)
            self.evaluator.add_model_metrics(variant, model, dummy_input)
        except Exception as e:
            logger.error(f"记录模型指标失败: {str(e)}")

        logger.info(f"\n训练 {variant} 模型...")

        # 加载训练和验证数据
        train_loader, val_loaders = load_train_val_data()
        
        # 配置优化器
        optimizer = Adam(
            model.parameters(),
            **cfg.OPTIMIZER_CONFIG
        )
        
        # 初始化训练器（包含早停配置）
        trainer = BaseTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=cfg.loss_fn,
            device=self.device,
            config=cfg.TRAIN_CONFIG  # 传递训练配置，包含 patience=20
        )
        
        # 训练模型
        results = trainer.train_model(
            train_loader=train_loader,
            val_loader=val_loaders,
            epochs=self.args.epochs,
            model_name=variant,
            checkpoint_dir=self.dirs['checkpoints']
        )
        
        # 更新评估器的训练指标
        if results:
            self.evaluator.add_training_metrics(
                variant_name=variant,
                training_time=time.time() - results['start_time'],
                val_losses=results['val_losses'],
                best_epoch=results['best_epoch'],
                best_loss=results['best_loss']
            )
            
            # 保存模型结果
            self.evaluator.save_model_results(
                variant=variant,
                best_model_state=results.get('best_model_state'),
                last_model_state=results.get('last_model_state'),
                train_losses=results.get('train_losses', []),
                val_losses=results.get('val_losses', []),
                save_dirs=self.dirs
            )
        
        return results
    
    def run_experiment(self):
        """运行模型架构对比实验"""
        try:
            # 记录实验开始时间
            experiment_start = time.time()
            
            # 训练所有变体
            variant_results = {}
            for variant in self.variants:
                logger.info(f"\n开始训练 {variant} 模型...")
                results = self._train_variant(variant)
                if results and 'best_loss' in results:
                    variant_results[variant] = results
                    logger.info(f"{variant} 模型训练完成")
            
            # 生成基础训练报告
            self.evaluator.generate_training_report()
            
            # 记录实验总结
            if variant_results:
                experiment_summary = {
                    'total_time': time.time() - experiment_start,
                    'completed_variants': list(variant_results.keys())
                }
                
                # 找出最佳模型
                best_variant = min(variant_results.items(), 
                                 key=lambda x: x[1].get('best_loss', float('inf')))[0]
                experiment_summary['best_model'] = best_variant
                
                # 保存实验总结
                summary_path = self.dirs['reports'] / 'experiment_summary.txt'
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write("实验总结\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"总耗时: {experiment_summary['total_time'] / 3600:.2f} 小时\n")
                    f.write(f"完成的变体: {', '.join(experiment_summary['completed_variants'])}\n")
                    f.write(f"最佳模型: {experiment_summary['best_model']}\n\n")
                    
                    f.write("各变体性能对比:\n")
                    f.write("-" * 50 + "\n")
                    for variant, results in variant_results.items():
                        f.write(f"\n{variant}:\n")
                        f.write(f"  最佳验证损失: {results.get('best_loss', 'N/A')}\n")
                        f.write(f"  最佳轮次: {results.get('best_epoch', 'N/A')}\n")
                        f.write(f"  训练轮次: {len(results.get('train_losses', []))}\n")
                
                logger.info("\n实验完成!")
                logger.info(f"基础训练报告已保存至: {self.dirs['reports'] / 'training_report.md'}")
                logger.info(f"实验总结已保存至: {summary_path}")
            
        except Exception as e:
            logger.error(f"实验执行过程中出错: {str(e)}")
            raise e


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="声音事件检测模型架构对比实验")
    parser.add_argument('--epochs', type=int, default=cfg.TRAIN_CONFIG['epochs'],
                        help='训练轮数')
    parser.add_argument('--num_runs', type=int, default=5,
                        help='实验重复次数')
    parser.add_argument('--seeds', nargs='+', type=int,
                        default=[205],
                        help='随机种子列表')
    parser.add_argument('--variant_group', type=str, default='daap',
                        choices=['daap', 'crnn', 'cnn_transformer', 'conformer', 'panns', 'faf'],
                        help='模型变体: daap, crnn, cnn_transformer, conformer, panns, faf')
    parser.add_argument('--gpu', type=str, default=None,
                        help='指定使用的GPU编号，例如 "0" 或 "0,1"')
    args = parser.parse_args()
    
    # 设置GPU
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        logger.info(f"设置 CUDA_VISIBLE_DEVICES = {args.gpu}")
    
    # 运行多次实验
    for run_idx, seed in enumerate(args.seeds):
        logger.info(f"\n开始第 {run_idx + 1}/{len(args.seeds)} 次实验 (seed={seed})...")
        experiment = ModelComparisonExperiment(args, seed)
        experiment.run_experiment()
    
    logger.info("\n所有实验完成!")
    logger.info("详细结果请查看各个seed目录下的分析报告")


if __name__ == '__main__':
    main() 