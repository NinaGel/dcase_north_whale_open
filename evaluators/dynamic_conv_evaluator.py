import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional
import logging
import time


class DynamicConvAnalyzer:
    """动态卷积特色分析器
    
    专注于分析动态卷积的特定特征：
    1. 动态卷积权重分布分析
    2. 时域和频域响应对比
    3. 不同SNR条件下的卷积行为分析
    4. 计算复杂度和效率分析
    """
    
    def __init__(self, save_dir: Path):
        self.save_dir = Path(save_dir)
        self.setup_logger()
        
        # 创建必要的目录
        self.dirs = {
            'figures': self.save_dir / 'figures',
            'metrics': self.save_dir / 'metrics',
            'reports': self.save_dir / 'reports'
        }
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化指标存储
        self.conv_metrics = {
            'weight_distributions': {},  # 动态卷积权重分布
            'response_patterns': {},     # 时频响应模式
            'snr_behaviors': {},         # SNR相关行为
            'complexity_metrics': {}     # 复杂度指标
        }
        
        # 设置绘图样式
        plt.style.use('seaborn')
        self.colors = plt.cm.Set3(np.linspace(0, 1, 10))
    
    def setup_logger(self):
        """设置日志记录器"""
        self.logger = logging.getLogger('dynamic_conv_analyzer')
        self.logger.setLevel(logging.INFO)
        
        # 创建文件处理器
        fh = logging.FileHandler(self.save_dir / 'dynamic_conv_analysis.log')
        fh.setLevel(logging.INFO)
        
        # 创建格式化器
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        
        if not self.logger.handlers:
            self.logger.addHandler(fh)
    
    def analyze_conv_weights(self, model: torch.nn.Module, variant: str):
        """分析动态卷积权重分布
        
        Args:
            model: 模型实例
            variant: 模型变体名称
        """
        try:
            # 收集动态卷积层的权重
            weights = []
            for name, module in model.named_modules():
                if 'tdy_conv' in name or 'fdy_conv' in name:
                    weights.append(module.weight.detach().cpu().numpy())
            
            if not weights:
                self.logger.info(f"{variant} 没有动态卷积层")
                return
            
            # 分析权重分布
            weight_stats = {
                'mean': np.mean(weights),
                'std': np.std(weights),
                'sparsity': np.mean(np.abs(weights) < 1e-5)
            }
            
            self.conv_metrics['weight_distributions'][variant] = weight_stats
            
            # 绘制权重分布图
            plt.figure(figsize=(10, 6))
            for w in weights:
                sns.kdeplot(w.flatten(), label=f'Layer weights')
            plt.title(f'{variant} Dynamic Conv Weight Distribution')
            plt.xlabel('Weight Value')
            plt.ylabel('Density')
            plt.legend()
            plt.savefig(self.dirs['figures'] / f'{variant}_weight_dist.png')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"分析 {variant} 权重时出错: {str(e)}")
    
    def analyze_response_patterns(self, model: torch.nn.Module, variant: str,
                                input_data: torch.Tensor):
        """分析时域和频域响应模式
        
        Args:
            model: 模型实例
            variant: 模型变体名称
            input_data: 示例输入数据
        """
        try:
            model.eval()
            with torch.no_grad():
                # 获取动态卷积层的输出
                time_responses = []
                freq_responses = []
                
                def hook_fn(module, input, output):
                    if hasattr(module, 'pool_dim'):
                        if module.pool_dim == 'time':
                            time_responses.append(output.detach().cpu().numpy())
                        else:
                            freq_responses.append(output.detach().cpu().numpy())
                
                # 注册钩子
                hooks = []
                for name, module in model.named_modules():
                    if 'tdy_conv' in name or 'fdy_conv' in name:
                        hooks.append(module.register_forward_hook(hook_fn))
                
                # 前向传播
                _ = model(input_data)
                
                # 移除钩子
                for hook in hooks:
                    hook.remove()
                
                # 分析响应模式
                response_stats = {
                    'time_mean_activation': np.mean([r.mean() for r in time_responses]) if time_responses else 0,
                    'freq_mean_activation': np.mean([r.mean() for r in freq_responses]) if freq_responses else 0,
                    'time_response_std': np.mean([r.std() for r in time_responses]) if time_responses else 0,
                    'freq_response_std': np.mean([r.std() for r in freq_responses]) if freq_responses else 0
                }
                
                self.conv_metrics['response_patterns'][variant] = response_stats
                
                # 可视化响应模式
                if time_responses or freq_responses:
                    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
                    
                    if time_responses:
                        im = axes[0].imshow(time_responses[0].mean(axis=1), aspect='auto', cmap='viridis')
                        axes[0].set_title('Time Domain Response')
                        plt.colorbar(im, ax=axes[0])
                    
                    if freq_responses:
                        im = axes[1].imshow(freq_responses[0].mean(axis=1), aspect='auto', cmap='viridis')
                        axes[1].set_title('Frequency Domain Response')
                        plt.colorbar(im, ax=axes[1])
                    
                    plt.tight_layout()
                    plt.savefig(self.dirs['figures'] / f'{variant}_response_patterns.png')
                    plt.close()
        
        except Exception as e:
            self.logger.error(f"分析 {variant} 响应模式时出错: {str(e)}")
    
    def analyze_snr_behavior(self, model: torch.nn.Module, variant: str,
                           snr_data: Dict[str, torch.Tensor]):
        """分析不同SNR条件下的卷积行为
        
        Args:
            model: 模型实例
            variant: 模型变体名称
            snr_data: 不同SNR级别的输入数据
        """
        try:
            model.eval()
            snr_behaviors = {}
            
            for snr_level, data in snr_data.items():
                with torch.no_grad():
                    # 收集动态卷积层在不同SNR下的行为
                    conv_outputs = []
                    
                    def hook_fn(module, input, output):
                        conv_outputs.append(output.detach().cpu().numpy())
                    
                    # 注册钩子
                    hooks = []
                    for name, module in model.named_modules():
                        if 'tdy_conv' in name or 'fdy_conv' in name:
                            hooks.append(module.register_forward_hook(hook_fn))
                    
                    # 前向传播
                    _ = model(data)
                    
                    # 移除钩子
                    for hook in hooks:
                        hook.remove()
                    
                    # 分析该SNR级别下的行为
                    if conv_outputs:
                        snr_behaviors[snr_level] = {
                            'mean_activation': np.mean([o.mean() for o in conv_outputs]),
                            'activation_std': np.mean([o.std() for o in conv_outputs]),
                            'activation_range': np.mean([o.max() - o.min() for o in conv_outputs])
                        }
            
            self.conv_metrics['snr_behaviors'][variant] = snr_behaviors
            
            # 可视化SNR行为
            if snr_behaviors:
                plt.figure(figsize=(10, 6))
                snr_levels = list(snr_behaviors.keys())
                mean_activations = [snr_behaviors[snr]['mean_activation'] for snr in snr_levels]
                
                plt.bar(snr_levels, mean_activations)
                plt.title(f'{variant} Dynamic Conv Activation vs SNR')
                plt.xlabel('SNR Level')
                plt.ylabel('Mean Activation')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(self.dirs['figures'] / f'{variant}_snr_behavior.png')
                plt.close()
        
        except Exception as e:
            self.logger.error(f"分析 {variant} SNR行为时出错: {str(e)}")
    
    def analyze_complexity(self, model: torch.nn.Module, variant: str,
                         input_data: torch.Tensor):
        """分析计算复杂度和效率
        
        Args:
            model: 模型实例
            variant: 模型变体名称
            input_data: 示例输入数据
        """
        try:
            # 计算FLOPs和参数量
            from thop import profile
            flops, params = profile(model, inputs=(input_data,))
            
            # 测量推理时间
            model.eval()
            with torch.no_grad():
                start_time = time.time()
                for _ in range(100):  # 重复100次以获得更准确的测量
                    _ = model(input_data)
                inference_time = (time.time() - start_time) / 100
            
            # 收集复杂度指标
            complexity_metrics = {
                'flops': flops,
                'params': params,
                'inference_time': inference_time,
                'flops_per_param': flops / params if params > 0 else 0
            }
            
            self.conv_metrics['complexity_metrics'][variant] = complexity_metrics
            
            # 可视化复杂度指标
            metrics_df = pd.DataFrame([complexity_metrics])
            metrics_df.index = [variant]
            
            plt.figure(figsize=(12, 6))
            metrics_df.plot(kind='bar')
            plt.title('Complexity Metrics Comparison')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.dirs['figures'] / f'{variant}_complexity.png')
            plt.close()
        
        except Exception as e:
            self.logger.error(f"分析 {variant} 复杂度时出错: {str(e)}")
    
    def generate_analysis_report(self):
        """生成动态卷积分析报告"""
        report = ["# 动态卷积特色分析报告\n"]
        
        # 1. 权重分布分析
        report.append("## 1. 动态卷积权重分布分析")
        if self.conv_metrics['weight_distributions']:
            for variant, stats in self.conv_metrics['weight_distributions'].items():
                report.append(f"\n### {variant}")
                report.append(f"- 平均权重: {stats['mean']:.4f}")
                report.append(f"- 权重标准差: {stats['std']:.4f}")
                report.append(f"- 稀疏度: {stats['sparsity']:.2%}")
        
        # 2. 响应模式分析
        report.append("\n## 2. 时频响应模式分析")
        if self.conv_metrics['response_patterns']:
            for variant, stats in self.conv_metrics['response_patterns'].items():
                report.append(f"\n### {variant}")
                report.append(f"- 时域平均激活: {stats['time_mean_activation']:.4f}")
                report.append(f"- 频域平均激活: {stats['freq_mean_activation']:.4f}")
                report.append(f"- 时域响应标准差: {stats['time_response_std']:.4f}")
                report.append(f"- 频域响应标准差: {stats['freq_response_std']:.4f}")
        
        # 3. SNR行为分析
        report.append("\n## 3. SNR相关行为分析")
        if self.conv_metrics['snr_behaviors']:
            for variant, behaviors in self.conv_metrics['snr_behaviors'].items():
                report.append(f"\n### {variant}")
                for snr_level, metrics in behaviors.items():
                    report.append(f"\n#### {snr_level}")
                    report.append(f"- 平均激活: {metrics['mean_activation']:.4f}")
                    report.append(f"- 激活标准差: {metrics['activation_std']:.4f}")
                    report.append(f"- 激活范围: {metrics['activation_range']:.4f}")
        
        # 4. 复杂度分析
        report.append("\n## 4. 计算复杂度分析")
        if self.conv_metrics['complexity_metrics']:
            metrics_df = pd.DataFrame(self.conv_metrics['complexity_metrics']).T
            report.append("\n" + metrics_df.to_markdown())
        
        # 5. 结论和建议
        report.append("\n## 5. 结论和建议")
        report.append("\n### 主要发现")
        
        # 根据分析结果生成结论
        if self.conv_metrics['weight_distributions']:
            most_sparse = min(self.conv_metrics['weight_distributions'].items(),
                            key=lambda x: x[1]['sparsity'])
            report.append(f"\n- 权重分布: {most_sparse[0]} 表现出最优的权重分布特征")
        
        if self.conv_metrics['response_patterns']:
            best_response = max(self.conv_metrics['response_patterns'].items(),
                              key=lambda x: x[1]['time_mean_activation'])
            report.append(f"\n- 响应模式: {best_response[0]} 展现出最强的特征提取能力")
        
        if self.conv_metrics['complexity_metrics']:
            most_efficient = min(self.conv_metrics['complexity_metrics'].items(),
                               key=lambda x: x[1]['inference_time'])
            report.append(f"\n- 计算效率: {most_efficient[0]} 具有最优的计算效率")
        
        # 保存报告
        report_path = self.dirs['reports'] / 'dynamic_conv_analysis.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        self.logger.info(f"分析报告已保存至: {report_path}")
    
    def analyze(self, model: torch.nn.Module, variant: str,
                input_data: torch.Tensor,
                snr_data: Optional[Dict[str, torch.Tensor]] = None):
        """执行完整的动态卷积分析
        
        Args:
            model: 模型实例
            variant: 模型变体名称
            input_data: 示例输入数据
            snr_data: 不同SNR级别的输入数据（可选）
        """
        try:
            # 分析权重分布
            self.analyze_conv_weights(model, variant)
            
            # 分析响应模式
            self.analyze_response_patterns(model, variant, input_data)
            
            # 分析SNR行为
            if snr_data is not None:
                self.analyze_snr_behavior(model, variant, snr_data)
            
            # 分析复杂度
            self.analyze_complexity(model, variant, input_data)
            
            # 生成报告
            self.generate_analysis_report()
            
            self.logger.info(f"{variant} 模型分析完成")
            
        except Exception as e:
            self.logger.error(f"分析 {variant} 时出错: {str(e)}")
            raise 