import os
# 设置环境变量以避免OpenMP运行时冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from pathlib import Path
import argparse
from Train.train_utils import BaseTrainer
import config as cfg
from Data.audio_dataset import load_train_val_data
import numpy as np
from Model.BA_Conv import BSA_Conv
from Model.T_FAC import DynamicConv2D
from Model.Spatial_shift_modified import spatial_shift
from Model.MultiScale_Ldsa import MultiScaleLDSA
from Model.Whale_Model_Attention_MultiScale_Ldsa import Whale_Model_Attention_MultiScale
import torch.nn.functional as F
import time
from evaluators.base_evaluator import BaseEvaluator
from tqdm import tqdm
import torch.cuda.amp as amp


class ModifiedSpatialShift(nn.Module):
    """修改的空间移位模块，支持动态卷积

    该模块结合了空间移位操作和动态卷积，用于增强特征的空间依赖关系。

    主要特点：
    1. 支持多种动态卷积类型：时域、频域或两者都有
    2. 使用组卷积进行通道扩展
    3. 包含自适应权重机制
    4. 集成了dropout正则化

    参数:
        in_channel (int): 输入通道数
        dynamic_type (str): 动态卷积类型，可选['none', 'time', 'freq', 'both']
        n (int): 空间移位步长
    """

    def __init__(self, in_channel, dynamic_type='both', n=cfg.MODEL_CONFIG['spatial_shift']['n']):
        super().__init__()
        self.in_channel = in_channel
        self.n = n
        self.dynamic_type = dynamic_type

        # 使用组卷积扩展通道数
        self.group_conv = nn.Conv2d(in_channel, in_channel * 3, kernel_size=1, groups=in_channel)

        # 根据dynamic_type配置动态卷积
        if dynamic_type == 'none':
            # 使用普通卷积替代动态卷积
            self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1)
        else:
            # 配置时域动态卷积
            if dynamic_type in ['time', 'both']:
                self.tdy_conv = DynamicConv2D(
                    in_channel, in_channel,
                    kernel_size=3, pool_dim='time'
                )
            else:
                self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1)

            # 配置频域动态卷积
            if dynamic_type in ['freq', 'both']:
                self.fdy_conv = DynamicConv2D(
                    in_channel, in_channel,
                    kernel_size=3, pool_dim='freq'
                )
            else:
                self.conv2 = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1)

        # 使用1x1卷积进行特征融合
        self.conv1x1 = nn.Conv2d(in_channel * 3, in_channel * 3, kernel_size=1)

        # 全连接层用于计算softmax权重
        self.linear = nn.Linear(in_channel * 3, in_channel * 3)

        # dropout层
        self.dropout = nn.Dropout(cfg.MODEL_CONFIG['spatial_shift']['dropout'])

    def forward(self, x):
        # 添加内存清理
        torch.cuda.empty_cache()  # 定期清理GPU缓存

        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        # 使用contiguous()确保内存连续
        x = self.group_conv(x).contiguous()
        x1, x2, x3 = torch.chunk(x, chunks=3, dim=1)

        # 根据dynamic_type应用不同的卷积操作
        if self.dynamic_type == 'none':
            x1 = self.conv1(x1)
            x2 = self.conv2(x2)
        else:
            if self.dynamic_type in ['time', 'both']:
                x1 = self.tdy_conv(x1)
            else:
                x1 = self.conv1(x1)

            if self.dynamic_type in ['freq', 'both']:
                x2 = self.fdy_conv(x2.transpose(2, 3)).transpose(2, 3)
            else:
                x2 = self.conv2(x2)

        # 空间移位操作
        S1 = spatial_shift(x1, self.n)
        S2 = spatial_shift(x2.transpose(2, 3), self.n).transpose(2, 3)
        S3 = x3

        # 特征融合
        combined = torch.cat((S1, S2, S3), dim=1)
        fused = self.conv1x1(combined)

        # 添加dropout
        fused = self.dropout(fused)

        # 全局池化和权重计算
        U = F.adaptive_avg_pool2d(fused, (1, 1)).view(fused.size(0), -1)
        weights = F.softmax(self.linear(U), dim=1)

        # 权重分配和加权求和
        a1, a2, a3 = weights.split(self.in_channel, dim=1)
        output = (a1.unsqueeze(-1).unsqueeze(-1) * S1 +
                  a2.unsqueeze(-1).unsqueeze(-1) * S2 +
                  a3.unsqueeze(-1).unsqueeze(-1) * S3).contiguous()

        return output


class DynamicConvModel(nn.Module):
    """动态卷积对比实验模型

    该模型是用于对比不同动态卷积策略效果的实验模型。

    架构组成：
    1. BSA卷积层：用于初始特征提取
    2. 修改的空间移位层：增强空间依赖关系
    3. MultiScaleLDSA：多尺度注意力机制
    4. GRU层：处理时序信息
    5. 分类层：输出预测结果

    参数:
        dynamic_type (str): 动态卷积类型，默认为'both'
    """

    def __init__(self, dynamic_type='both'):
        super().__init__()

        # 验证输入维度
        if cfg.AUDIO_CONFIG['freq'] & (cfg.AUDIO_CONFIG['freq'] - 1) != 0:
            raise ValueError(f"输入频率维度 {cfg.AUDIO_CONFIG['freq']} 必须是2的幂次")

        # 验证注意力头数
        n_feat = cfg.MODEL_CONFIG['attention']['n_feat']
        n_head = cfg.MODEL_CONFIG['attention']['n_head']
        if n_feat % n_head != 0:
            raise ValueError(f"特征维度 {n_feat} 必须能被注意力头数 {n_head} 整除")

        # 设置混合精度训练标志
        self.use_mixed_precision = cfg.TRAIN_CONFIG.get('mixed_precision', False)

        self.dynamic_type = dynamic_type

        # BSA卷积层
        self.bsa_conv1 = BSA_Conv(**cfg.MODEL_CONFIG['bsa_conv1'])
        self.bsa_conv2 = BSA_Conv(**cfg.MODEL_CONFIG['bsa_conv2'])

        # 修改的空间移位层
        self.spatial_shift = ModifiedSpatialShift(
            in_channel=cfg.MODEL_CONFIG['bsa_conv1']['c3_out'],
            dynamic_type=dynamic_type
        )

        # 多尺度LDSA注意力层
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
        )

        # 分类层
        self.classifier = nn.Linear(
            cfg.MODEL_CONFIG['gru']['hidden_size'] * 2,
            cfg.MODEL_CONFIG['num_classes']
        )

    def forward(self, x):
        # 输入维度检查和处理
        if x.dim() == 5:  # [B, 1, 1, F, T]
            x = x.squeeze(2)  # 移除多余的维度，变成 [B, 1, F, T]
        elif x.dim() == 3:  # [B, F, T]
            x = x.unsqueeze(1)  # 添加通道维度 [B, 1, F, T]
        elif x.dim() != 4:  # 不是 [B, C, F, T]
            raise ValueError(f"输入维度错误，期望3D、4D或5D，实际为{x.dim()}D: {x.shape}")

        # 如果启用了混合精度训练
        if self.training and self.use_mixed_precision:
            with amp.autocast():
                # BSA卷积处理
                x = self.bsa_conv1(x)

                # 空间移位处理
                x = self.spatial_shift(x)

                x = self.bsa_conv2(x)

                # 调整维度
                x = x.permute(0, 3, 1, 2).reshape(x.size(0), x.size(3), -1)

                # MultiScaleLDSA处理
                x = self.ldsa(x, x, x)

                # GRU处理
                x, _ = self.gru(x)

                # 分类
                output = self.classifier(x)
        else:
            # 非混合精度训练时的正常处理
            x = self.bsa_conv1(x)
            x = self.spatial_shift(x)
            x = self.bsa_conv2(x)
            x = x.permute(0, 3, 1, 2).reshape(x.size(0), x.size(3), -1)
            x = self.ldsa(x, x, x)
            x, _ = self.gru(x)
            output = self.classifier(x)

        return output


def initialize_model(model_variant):
    """初始化动态卷积模型变体

    根据指定的变体类型初始化相应的模型。

    Args:
        model_variant (str): 模型变体名称
            - daap: 基准模型
            - none: 不使用动态卷积
            - time: 仅使用时域动态卷积
            - freq: 仅使用频域动态卷积
            - both: 同时使用时域和频域动态卷积

    Returns:
        nn.Module: 初始化好的模型实例
    """
    if model_variant == 'daap':
        return Whale_Model_Attention_MultiScale()
    elif model_variant in ['none', 'time', 'freq', 'both']:
        return DynamicConvModel(dynamic_type=model_variant)
    else:
        raise ValueError(f"未知的模型变体: {model_variant}")


class DynamicConvComparison:
    """动态卷积对比实验管理器
    
    负责管理动态卷积对比实验的完整流程，包括：
    1. 模型训练和评估
    2. 基础性能评估
    3. 结果可视化和报告生成
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
        self.experiment_dir = cfg.EXPERIMENT_PATHS['dynamic_conv'] / f'seed_{seed}'
        self._setup_directories()
        
        # 设置注意力变体
        self.variants = ['freq', 'time', 'none', 'daap']
        
        # 初始化模型字典
        self.models = self._initialize_models()
        
        # 初始化评估器
        self.evaluator = BaseEvaluator(self.experiment_dir)
    
    def _set_random_seed(self, seed: int):
        """设置随机种子
        
        Args:
            seed: 随机种子
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    
    def _setup_directories(self):
        """设置实验相关目录"""
        # 创建主目录
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.dirs = {
            'checkpoints': self.experiment_dir / 'checkpoints',
            'models': self.experiment_dir / 'models',
            'curves': self.experiment_dir / 'curves',
            'metrics': self.experiment_dir / 'metrics',
            'summary': self.experiment_dir / 'summary'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _initialize_models(self):
        """初始化所有模型变体
        
        Returns:
            dict: 模型变体字典
        """
        models = {}
        for variant in self.variants:
            models[variant] = initialize_model(variant)
            models[variant] = models[variant].to(self.device)
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
            dummy_input = torch.randn(1, 1, 256, 309).to(self.device)
            self.evaluator.add_model_metrics(variant, model, dummy_input)
        except Exception as e:
            print(f"记录模型指标失败: {str(e)}")
        
        print(f"\n训练 {variant} 模型...")
        
        # 加载训练和验证数据
        train_loader, val_loaders = load_train_val_data()
        
        # 配置优化器
        optimizer = cfg.optimizer(model.parameters(), **cfg.OPTIMIZER_CONFIG)
        
        # 初始化训练器
        trainer = BaseTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=cfg.loss_fn,
            device=self.device
        )
        
        # 记录开始时间
        start_time = time.time()
        
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
                training_time=time.time() - start_time,
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
        """运行完整的实验流程"""
        results = {}
        
        # 训练所有变体
        for variant in self.variants:
            results[variant] = self._train_variant(variant)
        
        # 生成汇总报告
        self._generate_summary_report(results)
    
    def _generate_summary_report(self, results: dict):
        """生成实验汇总报告
        
        Args:
            results: 实验结果字典
        """
        summary_file = self.dirs['summary'] / 'experiment_summary.txt'
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=== 最终实验结果统计 ===\n")
            
            for variant in self.variants:
                if variant in results and results[variant]:  # 确保变体存在且有结果
                    f.write(f"\n{variant}:\n")
                    # 使用 get 方法安全地获取值，提供默认值
                    f.write(f"验证损失: {results[variant].get('best_loss', 'N/A'):.4f}\n")
                    f.write(f"最佳轮次: {results[variant].get('best_epoch', 'N/A')}\n")
                    
                    # 从 start_time 计算训练时间
                    if 'start_time' in results[variant]:
                        training_time = time.time() - results[variant]['start_time']
                        f.write(f"训练时间: {training_time:.2f}秒\n")
                    
                    # 检查并记录 SNR 指标
                    if 'snr_metrics' in results[variant]:
                        for snr_group, metrics in results[variant]['snr_metrics'].items():
                            if isinstance(metrics, dict) and 'mean_loss' in metrics:
                                f.write(f"{snr_group}: {metrics['mean_loss']:.4f}\n")
        
        print("\n实验完成!")
        print(f"汇总报告已保存至: {summary_file}")
        print("详细结果请查看实验目录下的分析报告")


def main():
    parser = argparse.ArgumentParser(description="声音事件检测模型动态卷积对比实验")
    parser.add_argument('--epochs', type=int, default=cfg.TRAIN_CONFIG['epochs'],
                       help='训练轮数')
    parser.add_argument('--num_runs', type=int, default=5,
                       help='实验重复次数')
    parser.add_argument('--seeds', nargs='+', type=int,
                       default=[63],
                       help='随机种子列表')
    args = parser.parse_args()
    
    # 对每个种子运行实验
    for seed in args.seeds:
        print(f"\n开始实验 (seed={seed})...")
        experiment = DynamicConvComparison(args, seed)
        experiment.run_experiment()


if __name__ == "__main__":
    main()