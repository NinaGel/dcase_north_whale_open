import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

from Train.train_utils import plot_losses
from .base_evaluator import BaseEvaluator
import logging


class SpatialShiftKAnalyzer(BaseEvaluator):
    """空间移位K值实验分析器
    
    用于分析和可视化不同K值变体的性能比较，包括：
    - 性能指标对比
    - 计算效率分析
    - 特征响应可视化
    - 自动生成分析报告
    """
    
    def __init__(self, save_dir):
        super().__init__(save_dir)
        self.metrics = {
            'training_time': {},    # 训练时间
            'memory_usage': {},     # 显存使用
            'parameter_count': {},  # 参数量
            'flops': {},           # 计算量
            'val_losses': {},      # 验证损失历史
            'best_epochs': {},     # 最佳轮次
            'best_losses': {},     # 最佳损失
            'feature_responses': {} # 特征响应
        }
        
        # 设置绘图样式
        plt.style.use('seaborn')
        self.colors = plt.cm.Set3(np.linspace(0, 1, 10))
        
        # 设置日志
        self.setup_logger()
    
    def setup_logger(self):
        """设置日志记录器"""
        self.logger = logging.getLogger('spatial_shift_k_analyzer')
        self.logger.setLevel(logging.INFO)
        
        # 创建文件处理器
        fh = logging.FileHandler(self.save_dir / 'analysis.log')
        fh.setLevel(logging.INFO)
        
        # 创建格式化器
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        
        # 添加处理器
        if not self.logger.handlers:
            self.logger.addHandler(fh)
    
    def add_model_metrics(self, k, model, dummy_input):
        """记录模型的基础指标"""
        try:
            # 确保模型和输入在同一个设备上
            device = next(model.parameters()).device
            dummy_input = dummy_input.to(device)
            
            # 计算参数量
            num_params = sum(p.numel() for p in model.parameters())
            self.metrics['parameter_count'][k] = num_params
            
            # 计算FLOPs
            from thop import profile
            model.eval()
            with torch.no_grad():
                # 临时禁用batch norm的统计更新
                def disable_bn_stats(m):
                    if isinstance(m, torch.nn.BatchNorm2d):
                        m.track_running_stats = False
                model.apply(disable_bn_stats)
                
                flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
                
                # 恢复batch norm的统计更新
                def enable_bn_stats(m):
                    if isinstance(m, torch.nn.BatchNorm2d):
                        m.track_running_stats = True
                model.apply(enable_bn_stats)
                
                self.metrics['flops'][k] = flops
            
            # 记录特征响应
            with torch.no_grad():
                x = model.bsa_conv1(dummy_input.unsqueeze(1))
                x = model.spatial_shift(x)
                self.metrics['feature_responses'][k] = x.cpu().numpy()
            
            self.logger.info(f"K={k} - 参数量: {num_params:,}, FLOPs: {flops:,}")
            
        except Exception as e:
            self.logger.error(f"记录K={k}指标时出错: {str(e)}")
            # 确保即使出错也有一些默认值
            if k not in self.metrics['parameter_count']:
                self.metrics['parameter_count'][k] = 0
            if k not in self.metrics['flops']:
                self.metrics['flops'][k] = 0
    
    def add_training_metrics(self, k, training_time, val_losses, best_epoch, best_loss):
        """添加训练相关的指标"""
        self.metrics['training_time'][k] = training_time
        self.metrics['val_losses'][k] = val_losses
        self.metrics['best_epochs'][k] = best_epoch
        self.metrics['best_losses'][k] = best_loss
    
    def plot_feature_responses(self):
        """可视化不同K值的特征响应"""
        if not self.metrics['feature_responses']:
            self.logger.warning("没有可用的特征响应数据，跳过特征响应可视化")
            return
        
        fig, axes = plt.subplots(len(self.metrics['feature_responses']), 1, 
                               figsize=(12, 4*len(self.metrics['feature_responses'])))
        
        # 确保axes是数组，即使只有一个子图
        if len(self.metrics['feature_responses']) == 1:
            axes = [axes]
        
        for i, (k, response) in enumerate(self.metrics['feature_responses'].items()):
            try:
                # 计算平均响应
                mean_response = np.mean(response, axis=(0,1))
                
                # 绘制热力图
                sns.heatmap(mean_response, ax=axes[i], cmap='viridis')
                axes[i].set_title(f'K={k} Feature Response')
                axes[i].set_xlabel('Time')
                axes[i].set_ylabel('Channel')
            except Exception as e:
                self.logger.error(f"绘制K={k}特征响应时出错: {str(e)}")
                continue
        
        plt.tight_layout()
        try:
            plt.savefig(self.save_dir / 'figures' / 'feature_responses.png')
        except Exception as e:
            self.logger.error(f"保存特征响应图时出错: {str(e)}")
        finally:
            plt.close()
    
    def plot_comparison_metrics(self):
        """绘制不同K值的性能对比图"""
        metrics_to_plot = {
            'Training Time (s)': 'training_time',
            'Memory Usage (MB)': 'memory_usage',
            'Parameters (M)': 'parameter_count',
            'Best Loss': 'best_losses'
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, (title, metric) in enumerate(metrics_to_plot.items()):
            data = self.metrics[metric]
            if metric == 'parameter_count':
                data = {k: v/1e6 for k, v in data.items()}  # 转换为M
                
            k_values = list(data.keys())
            values = list(data.values())
            
            axes[i].bar(k_values, values, color=self.colors)
            axes[i].set_title(title)
            axes[i].set_xlabel('K Value')
            axes[i].tick_params(axis='x', rotation=0)
            
            # 添加数值标签
            for j, v in enumerate(values):
                axes[i].text(k_values[j], v, f'{v:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'figures' / 'comparison_metrics.png')
        plt.close()
    
    def plot_training_curves(self):
        """绘制所有K值的训练曲线对比"""
        plt.figure(figsize=(12, 6))
        
        for k, losses in self.metrics['val_losses'].items():
            plt.plot(losses, label=f'K={k}')
        
        plt.title('Validation Loss Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(self.save_dir / 'figures' / 'training_curves.png')
        plt.close()
    
    def statistical_analysis(self):
        """进行统计分析并保存结果"""
        stats = {
            'K': [],
            'Best Loss': [],
            'Best Epoch': [],
            'Training Time': [],
            'Parameters (M)': [],
            'FLOPs (G)': []
        }
        
        for k in self.metrics['best_losses'].keys():
            stats['K'].append(k)
            stats['Best Loss'].append(self.metrics['best_losses'][k])
            stats['Best Epoch'].append(self.metrics['best_epochs'][k])
            stats['Training Time'].append(self.metrics['training_time'][k])
            stats['Parameters (M)'].append(self.metrics['parameter_count'][k] / 1e6)
            stats['FLOPs (G)'].append(self.metrics['flops'][k] / 1e9)
        
        df = pd.DataFrame(stats)
        df.to_csv(self.save_dir / 'metrics' / 'model_metrics.csv', index=False)
        
        return df
    
    def generate_report(self):
        """生成分析报告"""
        report = ["# 空间移位K值对比实验分析报告\n"]
        
        # 性能对比
        df = self.statistical_analysis()
        report.append("## 1. 性能指标对比")
        report.append(df.to_markdown())
        
        # 计算资源分析
        report.append("\n## 2. 计算资源分析")
        for k in df['K']:
            report.append(f"\n### K={k}")
            report.append(f"- 参数量: {df[df['K']==k]['Parameters (M)'].values[0]:.2f}M")
            report.append(f"- FLOPs: {df[df['K']==k]['FLOPs (G)'].values[0]:.2f}G")
            report.append(f"- 训练时间: {df[df['K']==k]['Training Time'].values[0]:.2f}s")
        
        # 结论和建议
        report.append("\n## 3. 结论和建议")
        
        # 找出最佳K值
        best_k = df.loc[df['Best Loss'].idxmin(), 'K']
        report.append(f"\n### 最佳K值: {best_k}")
        report.append(f"- 验证损失: {df[df['K']==best_k]['Best Loss'].values[0]:.4f}")
        report.append(f"- 最佳轮次: {df[df['K']==best_k]['Best Epoch'].values[0]}")
        
        # 性能收益分析
        baseline_k = 4  # 使用K=4作为基准
        baseline_loss = df[df['K']==baseline_k]['Best Loss'].values[0]
        for k in df['K']:
            if k != baseline_k:
                current_loss = df[df['K']==k]['Best Loss'].values[0]
                improvement = (baseline_loss - current_loss) / baseline_loss * 100
                report.append(f"\nK={k}相对于K=4的改进: {improvement:.2f}%")
        
        # 保存报告
        with open(self.save_dir / 'reports' / 'spatial_shift_k_report.md', 'w') as f:
            f.write('\n'.join(report))
    
    def analyze(self):
        """执行完整分析"""
        try:
            # 创建必要的子目录
            (self.save_dir / 'figures').mkdir(exist_ok=True)
            (self.save_dir / 'metrics').mkdir(exist_ok=True)
            (self.save_dir / 'reports').mkdir(exist_ok=True)
            
            if self.metrics['feature_responses']:
                self.plot_feature_responses()
            self.plot_comparison_metrics()
            self.plot_training_curves()
            self.generate_report()
            self.logger.info("分析完成，报告已生成")
        except Exception as e:
            self.logger.error(f"执行分析时出错: {str(e)}") 
    
    def save_model_results(self, k, best_model_state, last_model_state, train_losses, val_losses, save_dirs):
        """保存模型训练结果
        
        Args:
            k (int): 空间移位模块的分组数
            best_model_state (dict): 最佳模型状态字典
            last_model_state (dict): 最后一个epoch的模型状态字典
            train_losses (list): 训练损失历史
            val_losses (list): 验证损失历史
            save_dirs (dict): 保存目录字典
        """
        try:
            # 创建模型保存目录
            model_dir = save_dirs['models'] / f'k{k}'
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存最佳模型权重
            torch.save(best_model_state, model_dir / 'best_epoch.pth')
            
            # 保存最后一个epoch的模型权重
            torch.save(last_model_state, model_dir / 'last_epoch.pth')
            
            # 保存训练曲线
            curves_path = save_dirs['curves'] / f'k{k}'
            curves_path.mkdir(parents=True, exist_ok=True)
            plot_losses(
                train_losses=train_losses,
                val_losses=val_losses,
                epochs=len(train_losses),
                save_path=curves_path / 'training_curve'
            )
            
            # 保存训练指标
            metrics_path = save_dirs['analysis'] / 'metrics'
            metrics_path.mkdir(parents=True, exist_ok=True)
            with open(metrics_path / f'k{k}_metrics.txt', 'w') as f:
                f.write(f'K value: {k}\n')
                f.write(f'Best validation loss: {min(val_losses):.4f}\n')
                f.write(f'Final validation loss: {val_losses[-1]:.4f}\n')
                f.write(f'Final training loss: {train_losses[-1]:.4f}\n')
                f.write(f'Total epochs: {len(train_losses)}\n')
            
            self.logger.info(f"成功保存 K={k} 模型的训练结果")
            
        except Exception as e:
            self.logger.error(f"保存 K={k} 模型的训练结果时出错: {str(e)}")
            raise e 