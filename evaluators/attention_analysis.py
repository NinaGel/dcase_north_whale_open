import sys
import os
import torch
import logging
import argparse
from pathlib import Path
import numpy as np
from typing import Dict, Optional, List
from torch.utils.data import DataLoader
from collections import defaultdict
from os.path import dirname, abspath

# 添加项目根目录到Python路径
root_dir = dirname(dirname(abspath(__file__)))
sys.path.append(root_dir)

from evaluators.attention_evaluator import AttentionEvaluator
from Data.audio_dataset import load_test_data
from Model.attention_models import initialize_model
import config as cfg


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_model_path(experiment_name: str, seed: int, variant: str, model_type: str = 'best') -> Path:
    """获取模型文件路径
    
    Args:
        experiment_name: 实验名称
        seed: 随机种子
        variant: 模型变体名称
        model_type: 模型类型，'best' 或 'last'
        
    Returns:
        模型文件的完整路径
    """
    base_path = cfg.EXPERIMENT_PATHS[experiment_name] / f'seed_{seed}' / 'models' / variant
    model_file = f"{model_type}_epoch.pth"
    return base_path / model_file


def load_model(model_path: str, variant: str):
    """加载模型

    Args:
        model_path: 模型权重文件路径
        variant: 模型变体名称

    Returns:
        model: 加载了权重的模型实例
    """
    try:
        # 创建模型实例
        model = initialize_model(variant)

        # 加载权重
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)

        # 将模型设置为评估模式
        model.eval()

        # 如果有GPU则使用GPU
        if torch.cuda.is_available():
            model = model.cuda()

        logger.info(f"成功加载模型: {model_path}")
        return model

    except Exception as e:
        logger.error(f"加载模型时出错: {str(e)}")
        raise
        
    except Exception as e:
        logger.error(f"加载模型时出错: {str(e)}")
        raise


def analyze_attention(model_path: str, variant: str, save_dir: str, batch_size: int = 32):
    """分析注意力机制
    
    Args:
        model_path: 模型文件路径
        save_dir: 保存分析结果的目录
        batch_size: 批次大小
    """
    try:
        # 创建保存目录
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载模型
        logger.info("正在加载模型...")
        model = load_model(model_path, variant)
        
        # 加载测试数据
        logger.info("正在加载测试数据...")
        test_loaders = load_test_data(batch_size)
        
        # 初始化注意力评估器
        logger.info("初始化注意力评估器...")
        evaluator = AttentionEvaluator(save_dir)
        
        # 收集不同SNR级别的样本数据
        logger.info("收集注意力数据...")
        attention_data = {}
        for snr_level, loader in test_loaders.items():
            data = evaluator.collect_attention_data(model, loader, num_batches=5)
            if data is not None:
                attention_data[snr_level] = data
                logger.info(f"收集到 {snr_level} 级别的 {len(data['weights'])} 个样本")
        
        # 1. SNR级别对比分析
        logger.info("执行SNR级别对比分析...")
        evaluator.analyze_snr_attention(model, {snr: data['inputs'] for snr, data in attention_data.items()})
        
        # 2. 鲸鱼声音事件定位分析
        logger.info("执行鲸鱼声音事件定位分析...")
        for snr_level, loader in test_loaders.items():
            # 获取一个包含鲸鱼声音的样本
            for batch in loader:
                audio, label, _ = batch
                if label.sum() > 0:  # 确保样本中包含鲸鱼声音
                    evaluator.visualize_whale_detection(model, audio, label, snr_level)
                    break
        
        # 3. 注意力变体对比分析
        logger.info("执行注意力变体对比分析...")
        # 注：这部分需要在主实验脚本中调用，因为需要多个变体的模型
        
        # 4. 时序注意力分析
        logger.info("Executing temporal attention analysis...")
        for snr_level, data in attention_data.items():
            # 构建正确的权重字典格式
            weights_dict = {
                'weights': data['weights']
            }
            evaluator.analyze_temporal_attention(variant, {snr_level: weights_dict})
        
        # 5. 注意力与性能关联分析
        logger.info("Executing attention-performance correlation analysis...")
        for snr_level, loader in test_loaders.items():
            evaluator.attention_performance_analysis(model, loader)
        
        # 生成分析报告和图表
        logger.info("Generating analysis report and plots...")
        evaluator.generate_attention_report()
        evaluator.plot_attention_analysis()
        
        logger.info(f"Attention analysis completed, results saved to: {save_dir}")
        
    except Exception as e:
        logger.error(f"Error in attention mechanism analysis: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(description='分析注意力机制')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--variant', type=str, choices=['DSA', 'LDSA', 'Hybrid', 'StandardSA'], 
                      required=True, help='注意力变体类型')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    
    args = parser.parse_args()
    
    # 构建模型路径和保存目录
    base_dir = Path("experiments/Snr_Results/attention_comparison")
    model_path = base_dir / f"seed_{args.seed}/models/{args.variant}/best_epoch.pth"
    save_dir = base_dir / f"seed_{args.seed}/analysis/{args.variant}"
    
    # 分析注意力机制
    analyze_attention(str(model_path), args.variant, str(save_dir), args.batch_size)


if __name__ == '__main__':
    main() 