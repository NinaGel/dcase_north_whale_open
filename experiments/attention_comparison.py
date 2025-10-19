import matplotlib
matplotlib.use('Agg')  # 设置后端为非交互式

import sys
import os
import time
import logging
import torch
import numpy as np
from pathlib import Path
import argparse
from torch.optim import Adam

# 获取项目根目录
ROOT_DIR = Path(__file__).parent.parent

# 添加项目根目录到 Python 路径
sys.path.append(str(ROOT_DIR))

from Train.train_utils import BaseTrainer
from evaluators.base_evaluator import BaseEvaluator
from Data.audio_dataset import load_train_val_data
from Model.attention_models import initialize_model
import config as cfg

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AttentionComparison:
    """注意力机制对比实验"""
    
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
        self.experiment_dir = cfg.EXPERIMENT_PATHS['attention_comparison'] / f'seed_{seed}'
        self._setup_directories()
        
        # 设置注意力变体
        self.variants = ['MultiScale', 'DSA', 'LDSA', 'StandardSA']
        # 初始化模型字典
        self.models = self._initialize_models()
        
        # 初始化评估器
        self.evaluator = BaseEvaluator(self.experiment_dir)

        # 记录实验配置
        self._log_experiment_config()
    
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

    def _log_experiment_config(self):
        """记录实验配置"""
        config_path = self.dirs['reports'] / 'experiment_config.txt'
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write("实验配置\n")
            f.write("=" * 50 + "\n\n")

            # 记录音频配置
            f.write("音频配置:\n")
            f.write(f"- 采样率: {cfg.AUDIO_CONFIG['sr']} Hz\n")
            f.write(f"- 频率维度: {cfg.AUDIO_CONFIG['freq']}\n")
            f.write(f"- 时间帧数: {cfg.AUDIO_CONFIG['frame']}\n\n")

            # 记录MultiScale LDSA配置
            f.write("MultiScale LDSA配置:\n")
            f.write(f"- 上下文大小列表: {cfg.MULTISCALE_CONFIG['context_sizes']}\n")
            f.write(f"- 注意力头数: {cfg.MULTISCALE_CONFIG['n_head']}\n")
            f.write(f"- Dropout率: {cfg.MULTISCALE_CONFIG['dropout_rate']}\n")
            f.write(f"- 正则化配置: {cfg.MULTISCALE_CONFIG['regularization']}\n\n")

            # 记录LDSA配置
            f.write("LDSA配置:\n")
            f.write(f"- 上下文大小: {cfg.LDSA_CONFIG['context_size']}\n")
            f.write(f"- 注意力头数: {cfg.LDSA_CONFIG['n_head']}\n")
            f.write(f"- Dropout率: {cfg.LDSA_CONFIG['dropout_rate']}\n")
            f.write(f"- 使用偏置: {cfg.LDSA_OPTIMIZED_CONFIG['use_bias']}\n\n")

            # 记录DSA配置
            f.write("DSA配置:\n")
            f.write(f"- 注意力头数: {cfg.MODEL_CONFIG['attention']['n_head']}\n")
            f.write(f"- 特征维度: {cfg.MODEL_CONFIG['feature_dims']['n_feat']}\n")
            f.write(f"- 使用偏置: {cfg.LDSA_OPTIMIZED_CONFIG['use_bias']}\n\n")

            # 记录StandardSA配置
            f.write("StandardSA配置:\n")
            f.write(f"- 注意力头数: {cfg.MODEL_CONFIG['attention']['n_head']}\n")
            f.write(f"- 特征维度: {cfg.MODEL_CONFIG['feature_dims']['n_feat']}\n")
            f.write(f"- 使用偏置: {cfg.LDSA_OPTIMIZED_CONFIG['use_bias']}\n\n")

            # 记录训练配置
            f.write("训练配置:\n")
            f.write(f"- 训练轮数: {self.args.epochs}\n")
            f.write(f"- 批次大小: {cfg.TRAIN_CONFIG['batch_size']}\n")
            f.write(f"- 学习率: {cfg.TRAIN_CONFIG['optimizer_params']['lr']}\n")
            f.write(f"- 权重衰减: {cfg.TRAIN_CONFIG['optimizer_params']['weight_decay']}\n")
            f.write(f"- 随机种子: {self.args.seeds}\n")
            f.write(f"- 使用设备: {self.device}\n")
            f.write(f"- 混合精度训练: {cfg.TRAIN_CONFIG['mixed_precision']}\n")
            f.write(f"- 梯度累积步数: {cfg.TRAIN_CONFIG['gradient_accumulation_steps']}\n")
            f.write(f"- 早停轮数: {cfg.TRAIN_CONFIG['patience']}\n")

    def _train_variant(self, variant: str) -> dict:
        """训练单个注意力变体
        
        Args:
            variant: 注意力变体名称
            
        Returns:
            dict: 包含训练结果的字典
        """
        # 获取模型
        model = self.models[variant]
        
        # 记录模型基础指标
        try:
            # 计算模型参数量
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            logger.info(f"\n{variant} 模型参数统计:")
            logger.info(f"总参数量: {total_params:,}")
            logger.info(f"可训练参数量: {trainable_params:,}")
            
            # 记录模型结构
            dummy_input = torch.randn(1, cfg.AUDIO_CONFIG['freq'], cfg.AUDIO_CONFIG['frame']).to(self.device)
            self.evaluator.add_model_metrics(variant, model, dummy_input)
        except Exception as e:
            logger.error(f"记录模型指标失败: {str(e)}")

        logger.info(f"\n开始训练 {variant} 模型...")
        
        # 加载训练和验证数据
        train_loader, val_loaders = load_train_val_data(batch_size=cfg.TRAIN_CONFIG['batch_size'])
        
        # 配置优化器
        optimizer = cfg.optimizer(
            model.parameters(),
            **cfg.TRAIN_CONFIG['optimizer_params']
        )
        
        # 初始化混合精度训练
        if self.args.fp16 and torch.cuda.is_available():
            logger.info("启用混合精度训练...")
            scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = None
            if self.args.fp16:
                logger.warning("GPU不可用，无法启用混合精度训练")
        
        # 初始化训练器
        trainer = BaseTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=cfg.loss_fn,
            device=self.device,
            scaler=scaler  # 传入scaler
        )
        
        # 训练模型
        results = trainer.train_model(
            train_loader=train_loader,
            val_loader=val_loaders,  # 传入所有验证集
            epochs=self.args.epochs,
            model_name=variant,
            checkpoint_dir=self.dirs['checkpoints']
        )
        
        if results:
            # 更新评估器的训练指标
            self.evaluator.add_training_metrics(
                variant_name=variant,
                training_time=time.time() - results['start_time'],
                val_losses=results.get('val_losses', {}),  # 传入所有SNR组的验证损失
                best_epoch=results['best_epoch'],
                best_loss=results['best_loss']
            )
            
            # 保存模型结果
            self.evaluator.save_model_results(
                variant=variant,
                best_model_state=results.get('best_model_state'),
                last_model_state=results.get('last_model_state'),
                train_losses=results.get('train_losses', []),
                val_losses=results.get('val_losses', {}),  # 传入所有SNR组的验证损失
                save_dirs=self.dirs
            )
            
            # 记录SNR组性能统计
            if 'snr_metrics' in results:
                logger.info(f"\n{variant} SNR组性能统计:")
                for snr_group, metrics in results['snr_metrics'].items():
                    logger.info(f"{snr_group}:")
                    logger.info(f"  - 平均损失: {metrics['mean_loss']:.4f}")
                    logger.info(f"  - 最佳损失: {metrics['best_loss']:.4f}")
                    logger.info(f"  - 当前权重: {metrics['current_weight']:.4f}")
        
        return results

    def run_experiment(self):
        """运行注意力对比实验"""
        try:
            # 记录实验开始时间
            experiment_start = time.time()
            
            # 训练所有变体
            variant_results = {}
            variant_times = {}
            
            for variant in self.variants:
                logger.info(f"\n开始训练 {variant} 模型...")
                variant_start = time.time()
                
                results = self._train_variant(variant)
                if results and 'best_loss' in results:
                    variant_results[variant] = results
                    variant_times[variant] = time.time() - variant_start
                    
                    logger.info(f"{variant} 模型训练完成")
                    logger.info(f"\n{variant} 模型性能统计:")
                    logger.info(f"训练时长: {variant_times[variant]/3600:.2f} 小时")
                    logger.info(f"最佳验证损失: {results['best_loss']:.4f}")
                    logger.info(f"最佳轮次: {results['best_epoch']}")
                    
                    if 'snr_metrics' in results:
                        logger.info("\nSNR组性能:")
                        for snr_group, metrics in results['snr_metrics'].items():
                            logger.info(f"{snr_group}:")
                            logger.info(f"  - 平均损失: {metrics['mean_loss']:.4f}")
                            logger.info(f"  - 最佳损失: {metrics['best_loss']:.4f}")
            
            # 生成训练报告
            self.evaluator.generate_training_report()
            
            # 记录实验总结
            if variant_results:
                total_time = time.time() - experiment_start
                experiment_summary = {
                    'total_time': total_time,
                    'completed_variants': list(variant_results.keys()),
                    'variant_times': variant_times
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
                    f.write(f"总耗时: {total_time / 3600:.2f} 小时\n")
                    f.write(f"完成的变体: {', '.join(experiment_summary['completed_variants'])}\n")
                    f.write(f"最佳模型: {experiment_summary['best_model']}\n\n")
                    
                    f.write("各变体训练时间:\n")
                    for variant, train_time in variant_times.items():
                        f.write(f"- {variant}: {train_time/3600:.2f} 小时\n")
                    
                    f.write("\n各变体性能对比:\n")
                    f.write("-" * 50 + "\n")
                    for variant, results in variant_results.items():
                        f.write(f"\n{variant}:\n")
                        f.write(f"  最佳验证损失: {results.get('best_loss', 'N/A'):.4f}\n")
                        f.write(f"  最佳轮次: {results.get('best_epoch', 'N/A')}\n")
                        f.write(f"  训练轮次: {len(results.get('train_losses', []))}\n")
                        
                        if 'snr_metrics' in results:
                            f.write("\n  SNR组性能:\n")
                            for snr_group, metrics in results['snr_metrics'].items():
                                f.write(f"    {snr_group}:\n")
                                f.write(f"      - 平均损失: {metrics['mean_loss']:.4f}\n")
                                f.write(f"      - 最佳损失: {metrics['best_loss']:.4f}\n")
                                f.write(f"      - 当前权重: {metrics['current_weight']:.4f}\n")
                
                logger.info("\n训练完成!")
                logger.info(f"训练报告已保存至: {self.dirs['reports'] / 'training_report.md'}")
                logger.info(f"实验总结已保存至: {summary_path}")
                logger.info("\n如需进行注意力机制分析，请使用 attention_analysis.py 脚本")
            else:
                logger.warning("没有成功训练的模型变体")
            
        except Exception as e:
            logger.error(f"实验执行过程中出错: {str(e)}")
            raise e


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="attention模型对比实验")
    parser.add_argument('--epochs', type=int, default=cfg.TRAIN_CONFIG['epochs'], 
                       help='训练轮数')
    parser.add_argument('--num_runs', type=int, default=1,
                       help='实验重复次数')
    parser.add_argument('--seeds', nargs='+', type=int, 
                       default=[63, 64, 65, 66, 67],  # 使用两个种子进行对比
                       help='随机种子列表')
    parser.add_argument('--fp16', type=bool, default=cfg.TRAIN_CONFIG['mixed_precision'],
                       help='是否使用混合精度训练')
    args = parser.parse_args()
    
    # 运行多次实验
    for run_idx, seed in enumerate(args.seeds):
        logger.info(f"\n开始第 {run_idx + 1}/{len(args.seeds)} 次实验 (seed={seed})...")
        experiment = AttentionComparison(args, seed)
        experiment.run_experiment()
    
    logger.info("\n所有实验完成!")
    logger.info("详细结果请查看各个seed目录下的训练报告")


if __name__ == '__main__':
    main() 