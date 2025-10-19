import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from scipy import stats
import pandas as pd
from thop import profile
import torch.cuda as cuda
from typing import Dict, List, Optional, Union, Tuple

from Train.train_utils import plot_losses


class ModelComparisonEvaluator:
    """模型架构对比实验分析器
    
    专注于模型架构对比的特定分析：
    1. 模型架构分析
       - 参数量对比
       - 计算效率对比
       - 推理性能对比
    
    2. SNR性能分析
       - 不同SNR组的性能对比
       - 模型鲁棒性分析
    
    3. 可视化和报告
       - 性能对比图表
       - 架构优劣分析
    """
    
    def __init__(self, save_dir: Path):
        self.save_dir = save_dir
        self.setup_logger()
        
        # 初始化评估指标
        self.metrics = {
            'parameter_count': {},
            'flops': {},
            'memory_usage': {},
            'inference_time': {},
            'throughput': {},
            'snr_metrics': {}
        }
        
        # SNR组定义
        self.snr_groups = ['snr_high', 'snr_medium', 'snr_low', 'snr_very_low']
        
        # 设置绘图样式
        plt.style.use('seaborn')
        self.colors = plt.cm.Set3(np.linspace(0, 1, 10))
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建保存目录
        self.analysis_dir = self.save_dir / 'analysis'
        self.figures_dir = self.analysis_dir / 'figures'
        self.reports_dir = self.analysis_dir / 'reports'
        
        for dir_path in [self.figures_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def add_model_metrics(self, model_name: str, model: torch.nn.Module, 
                         dummy_input: torch.Tensor):
        """记录模型架构相关指标"""
        try:
            # 参数量
            num_params = sum(p.numel() for p in model.parameters())
            self.metrics['parameter_count'][model_name] = num_params
            
            model.eval()
            with torch.no_grad():
                # FLOPs
                flops, _ = profile(model, inputs=(dummy_input,))
                self.metrics['flops'][model_name] = flops
                
                # 推理时间
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                for _ in range(100):
                    _ = model(dummy_input)
                end_time.record()
                
                torch.cuda.synchronize()
                inference_time = start_time.elapsed_time(end_time) / 100
                self.metrics['inference_time'][model_name] = inference_time
                self.metrics['throughput'][model_name] = (dummy_input.size(0) * 100) / (inference_time / 1000)
                
                # 显存使用
                torch.cuda.reset_peak_memory_stats()
                _ = model(dummy_input)
                memory_used = torch.cuda.max_memory_allocated() / 1024**2
                self.metrics['memory_usage'][model_name] = memory_used
            
            self.logger.info(f"{model_name} 架构指标记录完成")
            
        except Exception as e:
            self.logger.error(f"记录模型架构指标时出错: {str(e)}")
            raise
        finally:
            torch.cuda.empty_cache()
    
    def add_training_metrics(self, model_name: str, training_time: float,
                           val_losses: List[float], best_epoch: int, best_loss: float,
                           snr_metrics: Dict = None):
        """记录SNR相关的训练指标"""
        if snr_metrics:
            if model_name not in self.metrics['snr_metrics']:
                self.metrics['snr_metrics'][model_name] = {}
            
            # 直接使用 snr_metrics 中的值，因为它们已经是平均损失
            for group, value in snr_metrics.items():
                self.metrics['snr_metrics'][model_name][group] = {
                    'mean_loss': float(value) if isinstance(value, (int, float)) else float(value['mean_loss']),
                    'std_loss': float(value['std_loss']) if isinstance(value, dict) and 'std_loss' in value else 0.0
                }
    
    def plot_training_curves(self):
        """绘制模型对比分析图表"""
        try:
            plt.figure(figsize=(20, 15))
            
            # 计算资源使用对比
            plt.subplot(2, 2, 1)
            self._plot_resource_usage()
            
            # 推理性能对比
            plt.subplot(2, 2, 2)
            self._plot_inference_performance()
            
            # SNR组性能对比
            plt.subplot(2, 2, 3)
            self._plot_snr_performance()
            
            # 模型规模对比
            plt.subplot(2, 2, 4)
            self._plot_model_size_comparison()
            
            plt.tight_layout()
            save_path = self.figures_dir / 'model_architecture_comparison.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"保存分析图表到: {save_path}")
            
        except Exception as e:
            self.logger.error(f"绘制分析图表时出错: {str(e)}")
            plt.close()
            raise
    
    def _plot_resource_usage(self):
        """绘制资源使用对比"""
        try:
            models = list(self.metrics['parameter_count'].keys())
            x = np.arange(len(models))
            width = 0.35
            
            ax1 = plt.gca()
            ax2 = ax1.twinx()
            
            rects1 = ax1.bar(x - width/2, 
                            [self.metrics['parameter_count'][m]/1e6 for m in models],
                            width, label='参数量(M)', color='skyblue')
            ax1.set_ylabel('参数量(M)', fontsize=10)
            
            rects2 = ax2.bar(x + width/2, 
                            [self.metrics['memory_usage'][m] for m in models],
                            width, label='显存使用(MB)', color='lightcoral')
            ax2.set_ylabel('显存使用(MB)', fontsize=10)
            
            plt.title('计算资源使用对比', fontsize=12)
            plt.xticks(x, models, rotation=45, fontsize=8)
            
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
            
        except Exception as e:
            self.logger.error(f"绘制资源使用对比图时出错: {str(e)}")
            raise
    
    def _plot_inference_performance(self):
        """绘制推理性能对比"""
        models = list(self.metrics['inference_time'].keys())
        x = np.arange(len(models))
        width = 0.35
        
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        rects1 = ax1.bar(x - width/2, 
                        [self.metrics['inference_time'][m] for m in models],
                        width, label='推理时间(ms)', color='lightgreen')
        ax1.set_ylabel('推理时间(ms)', fontsize=10)
        
        rects2 = ax2.bar(x + width/2, 
                        [self.metrics['throughput'][m] for m in models],
                        width, label='吞吐量(样本/秒)', color='plum')
        ax2.set_ylabel('吞吐量(样本/秒)', fontsize=10)
        
        plt.title('推理性能对比', fontsize=12)
        plt.xticks(x, models, rotation=45, fontsize=8)
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
    
    def _plot_snr_performance(self):
        """绘制SNR组性能对比"""
        if not self.metrics['snr_metrics']:
            return
            
        models = list(self.metrics['snr_metrics'].keys())
        
        # 获取所有可用的SNR组
        available_groups = set()
        for model in models:
            available_groups.update(self.metrics['snr_metrics'][model].keys())
        available_groups = sorted(list(available_groups))
        
        if not available_groups:
            self.logger.warning("没有找到SNR组数据，跳过SNR性能对比图的绘制")
            return
            
        x = np.arange(len(models))
        width = 0.8 / len(available_groups)
        
        for i, group in enumerate(available_groups):
            means = []
            for model in models:
                if group in self.metrics['snr_metrics'][model]:
                    group_data = self.metrics['snr_metrics'][model][group]
                    if isinstance(group_data, dict):
                        means.append(group_data.get('mean_loss', 0))
                    else:
                        means.append(float(group_data))
                else:
                    means.append(0)  # 如果没有该组的数据，使用0填充
                    
            plt.bar(x + i*width - width*len(available_groups)/2, 
                   means, width, label=group)
        
        plt.title('SNR组性能对比', fontsize=12)
        plt.xlabel('模型', fontsize=10)
        plt.ylabel('平均损失', fontsize=10)
        plt.xticks(x, models, rotation=45)
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
    
    def _plot_model_size_comparison(self):
        """绘制模型规模对比"""
        models = list(self.metrics['parameter_count'].keys())
        x = np.arange(len(models))
        width = 0.35
        
        plt.bar(x, [self.metrics['flops'][m]/1e9 for m in models],
               width, label='FLOPs(G)')
        
        plt.title('模型规模对比', fontsize=12)
        plt.xlabel('模型', fontsize=10)
        plt.ylabel('FLOPs(G)', fontsize=10)
        plt.xticks(x, models, rotation=45)
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
    
    def generate_report(self):
        """生成模型架构对比分析报告"""
        try:
            report = ["# 模型架构对比分析报告\n"]
            
            # 1. 模型架构分析
            report.append("## 1. 模型架构分析")
            for model_name in self.metrics['parameter_count'].keys():
                report.append(f"\n### {model_name}")
                report.append(f"- 参数量: {self.metrics['parameter_count'][model_name]/1e6:.2f}M")
                report.append(f"- FLOPs: {self.metrics['flops'][model_name]/1e9:.2f}G")
                report.append(f"- 显存使用: {self.metrics['memory_usage'][model_name]:.2f}MB")
                report.append(f"- 推理时间: {self.metrics['inference_time'][model_name]:.2f}ms")
                report.append(f"- 吞吐量: {self.metrics['throughput'][model_name]:.2f}样本/秒")
            
            # 2. SNR性能分析
            report.append("\n## 2. SNR性能分析")
            if self.metrics['snr_metrics']:
                for group in self.snr_groups:
                    report.append(f"\n### {group}")
                    best_model = min(
                        self.metrics['snr_metrics'].items(),
                        key=lambda x: x[1][group]['mean_loss']
                    )[0]
                    report.append(f"最佳模型: {best_model}")
                    report.append(f"平均损失: {self.metrics['snr_metrics'][best_model][group]['mean_loss']:.4f}")
                    report.append(f"标准差: {self.metrics['snr_metrics'][best_model][group]['std_loss']:.4f}")
            
            # 3. 架构优劣分析
            report.append("\n## 3. 架构优劣分析")
            
            # 计算效率排名
            efficiency_ranking = sorted(
                self.metrics['inference_time'].items(),
                key=lambda x: x[1]
            )
            report.append("\n### 计算效率排名")
            for rank, (model, time) in enumerate(efficiency_ranking, 1):
                report.append(f"{rank}. {model}: {time:.2f}ms")
            
            # 资源效率排名
            resource_ranking = sorted(
                self.metrics['memory_usage'].items(),
                key=lambda x: x[1]
            )
            report.append("\n### 资源效率排名")
            for rank, (model, memory) in enumerate(resource_ranking, 1):
                report.append(f"{rank}. {model}: {memory:.2f}MB")
            
            # 保存报告
            report_path = self.reports_dir / 'model_architecture_analysis.md'
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report))
            
            self.logger.info(f"架构分析报告已保存至: {report_path}")
            
        except Exception as e:
            self.logger.error(f"生成架构分析报告时出错: {str(e)}")
            raise
    
    def setup_logger(self):
        """设置日志记录器"""
        self.logger = logging.getLogger('model_comparison_analyzer')
        self.logger.setLevel(logging.INFO)
        
        # 创建日志目录
        log_dir = self.save_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建文件处理器
        log_file = log_dir / 'model_comparison.log'
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # 创建控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        
        # 创建格式化器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # 添加处理器
        if not self.logger.handlers:
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)
        
        self.logger.info("初始化model_comparison分析器")