import sys
import os
# 设置环境变量以避免OpenMP运行时冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import argparse
from torch.optim import Adam
from torch import nn
# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import logging
from Train.train_utils import BaseTrainer
from evaluators.base_evaluator import BaseEvaluator
from evaluators.conv_module_evaluator import ConvModuleAnalyzer
from Data.audio_dataset import load_train_val_data
from Model.MultiScale_Ldsa import MultiScaleLDSA
from Model.Whale_Model_Attention_MultiScale_Ldsa import Whale_Model_Attention_MultiScale
import config as cfg
from Data.audio_dataset import load_train_val_data
import numpy as np
from torchvision.models import resnet18, efficientnet_b0, mobilenet_v2
from Model.Spatial_shift_modified import Spatial_shift
from Model.BA_Conv import BSA_Conv
import torch.cuda.amp as amp

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 1x3 convolution branch
        self.conv1x3 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1))
        
        # 3x3 convolution branch
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # 3x1 convolution branch
        self.conv3x1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0))
        
        # Batch Normalization and GELU
        self.bn = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()
        
        # 添加下采样层
        self.pool = nn.AvgPool2d(kernel_size=(4, 1))
        
    def forward(self, x):
        # 输入维度检查和处理
        if x.dim() == 5:  # [B, 1, 1, F, T]
            x = x.squeeze(2)  # 移除多余的维度，变成 [B, 1, F, T]
        elif x.dim() == 3:  # [B, F, T]
            x = x.unsqueeze(1)  # 添加通道维度 [B, 1, F, T]
        elif x.dim() != 4:  # 不是 [B, C, F, T]
            raise ValueError(f"输入维度错误，期望3D、4D或5D，实际为{x.dim()}D: {x.shape}")
        
        # Process through parallel branches
        y1 = self.conv1x3(x)
        y2 = self.conv3x3(x)
        y3 = self.conv3x1(x)
        
        # Combine features
        out = y1 + y2 + y3
        
        # Apply normalization and activation
        out = self.gelu(self.bn(out))
        
        # Apply downsampling
        out = self.pool(out)
        
        return out


class StandardConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=(4, 1))  # 添加下采样层
        )
        
        # 设置数据类型
        self.dtype = torch.float32
    
    def forward(self, x):
        # 输入维度检查和处理
        if x.dim() == 5:  # [B, 1, 1, F, T]
            x = x.squeeze(2)  # 移除多余的维度，变成 [B, 1, F, T]
        elif x.dim() == 3:  # [B, F, T]
            x = x.unsqueeze(1)  # 添加通道维度 [B, 1, F, T]
        elif x.dim() != 4:  # 不是 [B, C, F, T]
            raise ValueError(f"输入维度错误，期望3D、4D或5D，实际为{x.dim()}D: {x.shape}")
        
        # 确保输入在正确的设备和数据类型上
        device = next(self.parameters()).device
        x = x.to(device, self.dtype)
        
        # 应用卷积操作
        x = self.conv(x)
        x = x.to(device, self.dtype)
        
        return x


class InceptionConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 确保每个分支的输出通道数之和等于out_channels
        branch_channels = out_channels // 4
        
        # 1x1 branch
        self.branch1 = nn.Conv2d(in_channels, branch_channels, kernel_size=1)
        
        # 1x3 + 3x1 branch
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=(1,3), padding=(0,1)),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=(3,1), padding=(1,0))
        )
        
        # 3x3 branch
        self.branch3 = nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=1)
        
        # 3x3 + 3x3 branch
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=1),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1)
        )
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()
        
        # 添加下采样层
        self.pool = nn.AvgPool2d(kernel_size=(4, 1))
        
    def forward(self, x):
        # 输入维度检查和处理
        if x.dim() == 5:  # [B, 1, 1, F, T]
            x = x.squeeze(2)  # 移除多余的维度，变成 [B, 1, F, T]
        elif x.dim() == 3:  # [B, F, T]
            x = x.unsqueeze(1)  # 添加通道维度 [B, 1, F, T]
        elif x.dim() != 4:  # 不是 [B, C, F, T]
            raise ValueError(f"输入维度错误，期望3D、4D或5D，实际为{x.dim()}D: {x.shape}")
        
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        
        # 拼接各分支的输出
        out = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.gelu(self.bn(out))
        
        # 应用下采样
        out = self.pool(out)
        
        return out


class MobileConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Depthwise Separable Conv
        self.depthwise = nn.Conv2d(in_channels, in_channels, 
                                  kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # Additional enhancement layers
        self.conv1x3 = nn.Conv2d(out_channels, out_channels, 
                                kernel_size=(1,3), padding=(0,1))
        self.conv3x1 = nn.Conv2d(out_channels, out_channels, 
                                kernel_size=(3,1), padding=(1,0))
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()
        
        # 添加下采样层
        self.pool = nn.AvgPool2d(kernel_size=(4, 1))
        
    def forward(self, x):
        # 输入维度检查和处理
        if x.dim() == 5:  # [B, 1, 1, F, T]
            x = x.squeeze(2)  # 移除多余的维度，变成 [B, 1, F, T]
        elif x.dim() == 3:  # [B, F, T]
            x = x.unsqueeze(1)  # 添加通道维度 [B, 1, F, T]
        elif x.dim() != 4:  # 不是 [B, C, F, T]
            raise ValueError(f"输入维度错误，期望3D、4D或5D，实际为{x.dim()}D: {x.shape}")
        
        # Depthwise separable convolution
        out = self.depthwise(x)
        out = self.pointwise(out)
        
        # Enhancement with asymmetric convolutions
        time_feat = self.conv1x3(out)
        freq_feat = self.conv3x1(out)
        
        out = out + time_feat + freq_feat
        out = self.gelu(self.bn(out))
        
        # 应用下采样
        out = self.pool(out)
        
        return out


class ConvBackbone(nn.Module):
    """基础卷积骨干网络，使用不同的卷积模块进行特征提取，
    严格遵循Whale_Model_Attention的实现，只替换BSA_Conv部分"""
    def __init__(self, backbone_type='standard_conv'):
        super().__init__()
        self.backbone_type = backbone_type
        
        # 选择卷积模块
        if backbone_type == 'daap':
            # 使用原始的BSA_Conv模块
            self.conv1 = BSA_Conv(**cfg.MODEL_CONFIG['bsa_conv1'])
            self.conv2 = BSA_Conv(**cfg.MODEL_CONFIG['bsa_conv2'])
        else:
            # 其他卷积模块，但使用相同的配置参数
            conv_module = {
                'standard_conv': StandardConv,
                'ra_conv': RAConv,
                'inception_conv': InceptionConv,
                'mobile_conv': MobileConv
            }[backbone_type]
            
            # 使用与BSA_Conv相同的配置参数
            self.conv1 = conv_module(
                in_channels=cfg.MODEL_CONFIG['bsa_conv1']['in_channel'],
                out_channels=cfg.MODEL_CONFIG['bsa_conv1']['c3_out']
            )
            self.conv2 = conv_module(
                in_channels=cfg.MODEL_CONFIG['bsa_conv2']['in_channel'],
                out_channels=cfg.MODEL_CONFIG['bsa_conv2']['c3_out']
            )
        
        # 空间移位模块，与原始模型相同
        self.spatial_shift = Spatial_shift(
            in_channel=cfg.MODEL_CONFIG['bsa_conv1']['c3_out']
        )
        
        # 设置数据类型
        self.dtype = torch.float32
    
    def forward(self, x):
        """前向传播，严格遵循Whale_Model_Attention的实现
        
        Args:
            x (Tensor): 输入特征 [B, F, T] 或 [B, C, F, T]
            
        Returns:
            Tensor: 输出特征 [B, 32, 16, 309]
        """
        # 输入维度检查和处理
        if x.dim() == 5:  # [B, 1, 1, F, T]
            x = x.squeeze(2)  # 移除多余的维度，变成 [B, 1, F, T]
        elif x.dim() == 3:  # [B, F, T]
            x = x.unsqueeze(1)  # 添加通道维度 [B, 1, F, T]
        elif x.dim() != 4:  # 不是 [B, C, F, T]
            raise ValueError(f"输入维度错误，期望3D、4D或5D，实际为{x.dim()}D: {x.shape}")
        
        # 确保输入在正确的设备和数据类型上
        device = next(self.parameters()).device
        x = x.to(device, self.dtype)
        
        # 第一个卷积模块
        x = self.conv1(x)  # [B, 8, H/3, W]
        x = x.to(device, self.dtype)
        
        # 空间移位
        x = self.spatial_shift(x)  # [B, 8, H/3, W]
        x = x.to(device, self.dtype)
        
        # 第二个卷积模块
        x = self.conv2(x)  # [B, 32, H/6, W]
        x = x.to(device, self.dtype)
        
        return x


class ConvComparisonModel(nn.Module):
    """卷积模块对比实验模型，严格遵循Whale_Model_Attention的实现"""
    
    def __init__(self, backbone_type='standard_conv'):
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
        
        if backbone_type == 'bsa_conv':
            # 使用原始的Whale_Model_Attention
            self.model = Whale_Model_Attention_MultiScale()
        else:
            # 使用不同的卷积模块进行特征提取，但保持其他结构不变
            self.backbone = ConvBackbone(backbone_type)
            
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
                if cfg.MODEL_CONFIG['gru']['num_layers'] > 1 else 0
            )
            
            # 分类层
            self.fc = nn.Linear(
                cfg.MODEL_CONFIG['gru']['hidden_size'] * 2,
                cfg.MODEL_CONFIG['num_classes']
            )
    
    def forward(self, x):
        """前向传播，严格遵循Whale_Model_Attention的实现
        
        Args:
            x (Tensor): 输入特征 [B, F, T] 或 [B, C, F, T]
            
        Returns:
            Tensor: 模型输出 [B, T, num_classes]
        """
        # 输入维度检查和处理
        if x.dim() == 5:  # [B, 1, 1, F, T]
            x = x.squeeze(2)  # 移除多余的维度，变成 [B, 1, F, T]
        elif x.dim() == 3:  # [B, F, T]
            x = x.unsqueeze(1)  # 添加通道维度 [B, 1, F, T]
        elif x.dim() != 4:  # 不是 [B, C, F, T]
            raise ValueError(f"输入维度错误，期望3D、4D或5D，实际为{x.dim()}D: {x.shape}")
        
        # 确保输入在正确的设备和数据类型上
        device = next(self.parameters()).device
        x = x.to(device)
        
        if hasattr(self, 'model'):
            return self.model(x)
        
        # 使用混合精度训练
        if self.training and self.use_mixed_precision:
            with amp.autocast():
                # 特征提取
                x = self.backbone(x)  # [B, 32, 16, 309]
                
                # 验证特征维度
                B, C, F, T = x.shape
                expected_freq = cfg.MODEL_CONFIG['feature_dims']['freq_dim']
                if F != expected_freq:
                    raise ValueError(f"特征频率维度 {F} 与预期维度 {expected_freq} 不匹配")
                
                # 调整维度
                x = x.transpose(1, 2)  # [B, F, C, T]
                x = x.reshape(B, F * C, T)  # [B, F*C, T]
                x = x.transpose(1, 2)  # [B, T, F*C]
                
                # Dynamic注意力处理
                x = self.ldsa(x, x, x)  # [B, T, F*C]
                
                # GRU处理
                x, _ = self.gru(x)  # [B, T, hidden_size*2]
                
                # 分类
                output = self.fc(x)  # [B, T, num_classes]
        else:
            # 特征提取
            x = self.backbone(x)  # [B, 32, 16, 309]
            
            # 验证特征维度
            B, C, F, T = x.shape
            expected_freq = cfg.MODEL_CONFIG['feature_dims']['freq_dim']
            if F != expected_freq:
                raise ValueError(f"特征频率维度 {F} 与预期维度 {expected_freq} 不匹配")
            
            # 调整维度
            x = x.transpose(1, 2)  # [B, F, C, T]
            x = x.reshape(B, F * C, T)  # [B, F*C, T]
            x = x.transpose(1, 2)  # [B, T, F*C]
            
            # Dynamic注意力处理
            x = self.ldsa(x, x, x)  # [B, T, F*C]
            
            # GRU处理
            x, _ = self.gru(x)  # [B, T, hidden_size*2]
            
            # 分类
            output = self.fc(x)  # [B, T, num_classes]
        
        return output


def initialize_model(model_variant):
    """初始化卷积模块实验的模型变体
    
    Args:
        model_variant (str): 模型变体名称
            - standard_conv: 标准卷积模块
            - ra_conv: 残差注意力卷积模块
            - inception_conv: Inception风格卷积模块
            - mobile_conv: MobileNet风格卷积模块
            - bsa_conv: BSA卷积模块（基准模型）
            
    Returns:
        nn.Module: 初始化的模型
    """
    if model_variant == 'bsa_conv':
        return Whale_Model_Attention_MultiScale()
    
    # 验证模型变体名称
    valid_variants = ['daap', 'standard_conv', 'ra_conv', 'inception_conv', 'mobile_conv']
    if model_variant not in valid_variants:
        raise ValueError(f"未知的模型变体: {model_variant}")
    
    # 初始化对应的模型
    model = ConvComparisonModel(backbone_type=model_variant)
    
    return model


class ConvModuleComparison:
    """卷积模块对比实验
    
    专注于模型训练和训练过程的基础评估，不包含卷积模块分析。
    卷积模块分析请使用 ConvModuleAnalyzer。
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
        self.experiment_dir = cfg.EXPERIMENT_PATHS['conv_comparison'] / f'seed_{seed}'
        self._setup_directories()
        
        # 设置卷积变体
        self.variants = ['standard_conv', 'inception_conv', 'mobile_conv', 'ra_conv', 'daap']
        
        # 初始化模型字典
        self.models = self._initialize_models()
        
        # 初始化基础评估器
        self.evaluator = BaseEvaluator(self.experiment_dir)
        
        # 初始化卷积模块分析器（仅用于特定分析）
        self.conv_analyzer = ConvModuleAnalyzer(self.experiment_dir)
    
    def _set_random_seed(self, seed: int):
        """设置随机种子"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.benchmark = cfg.GPU_CONFIG['cudnn_benchmark']
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
            models[variant].to(self.device)
        return models
    
    def _train_variant(self, variant: str) -> dict:
        """训练单个卷积变体
        
        Args:
            variant: 卷积变体名称
            
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
            logger.error(f"记录模型指标失败: {str(e)}")

        logger.info(f"\n训练 {variant} 模型...")
        
        # 加载训练和验证数据
        train_loader, val_loaders = load_train_val_data()
        
        # 配置优化器
        optimizer = Adam(
            model.parameters(),
            **cfg.OPTIMIZER_CONFIG
        )
        
        # 初始化训练器
        trainer = BaseTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=cfg.loss_fn,
            device=self.device
        )
        
        # 训练模型
        results = trainer.train_model(
            train_loader=train_loader,
            val_loader=val_loaders,  # 使用SNR分组的验证加载器
            epochs=self.args.epochs,
            model_name=variant,
            checkpoint_dir=self.dirs['checkpoints']
        )
        
        # 更新评估器的训练指标
        if results:  # 添加结果检查
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
            
            # 将结果传递给卷积模块分析器进行特定分析
            self.conv_analyzer.analyze_conv_module(
                variant=variant,
                model=model,
                results=results
            )
        
        return results
    
    def run_experiment(self):
        """运行卷积模块对比实验（仅训练部分）"""
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
                continue
            
            # 生成基础训练报告
            self.evaluator.generate_training_report()
            
            # 生成卷积模块特定分析报告
            self.conv_analyzer.generate_analysis_report()
            
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
                
                logger.info("\n训练完成!")
                logger.info(f"训练报告已保存至: {self.dirs['reports'] / 'training_report.md'}")
                logger.info(f"实验总结已保存至: {summary_path}")
                logger.info("\n如需进行卷积模块分析，请查看卷积模块分析报告")
            else:
                logger.warning("没有成功训练的模型变体")
            
        except Exception as e:
            logger.error(f"实验执行过程中出错: {str(e)}")
            raise e


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="卷积模块对比实验（训练部分）")
    parser.add_argument('--epochs', type=int, default=cfg.TRAIN_CONFIG['epochs'], 
                       help='训练轮数')
    parser.add_argument('--num_runs', type=int, default=1,
                       help='实验重复次数')
    parser.add_argument('--seeds', nargs='+', type=int, 
                       default=[99],
                       help='随机种子列表')
    args = parser.parse_args()
    
    # 运行多次实验
    for run_idx, seed in enumerate(args.seeds):
        logger.info(f"\n开始第 {run_idx + 1}/{len(args.seeds)} 次实验 (seed={seed})...")
        experiment = ConvModuleComparison(args, seed)
        experiment.run_experiment()
    
    logger.info("\n所有训练完成!")
    logger.info("详细结果请查看各个seed目录下的训练报告")
    logger.info("如需进行卷积模块分析，请查看卷积模块分析报告")


if __name__ == '__main__':
    main() 