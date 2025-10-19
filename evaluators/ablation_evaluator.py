from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import seaborn as sns
from thop import profile
import torch.cuda as cuda

from Train.train_utils import plot_losses
from .base_evaluator import BaseEvaluator
import logging


class AblationAnalyzer(BaseEvaluator):
    """消融实验分析器
    
    专注于消融实验特有的分析功能：
    1. 模块贡献度分析
    2. 模块交互分析
    3. 性能-复杂度权衡分析
    4. 统计显著性分析
    """
    
    def __init__(self, save_dir):
        """初始化分析器"""
        super().__init__(save_dir)
        
        # 消融实验特有的指标
        self.ablation_metrics = {
            'module_contributions': {},  # 各模块的贡献度
            'module_interactions': {},   # 模块间的交互影响
            'efficiency_metrics': {},    # 性能-复杂度权衡指标
            'statistical_tests': {}      # 统计检验结果
        }
        
        # 设置绘图样式
        plt.style.use('seaborn')
        self.colors = plt.cm.Set3(np.linspace(0, 1, 10))
        
        # 设置日志
        self.setup_logger()
    
    def add_ablation_metrics(self, variant_name, results):
        """添加消融实验特有的指标
        
        Args:
            variant_name: 模型变体名称
            results: 训练结果字典
        """
        # 1. 计算模块贡献度
        if 'baseline' in variant_name.lower():
            baseline_loss = results['best_loss']
            self.ablation_metrics['baseline_loss'] = baseline_loss
        else:
            # 计算相对于基线的性能下降
            if hasattr(self.ablation_metrics, 'baseline_loss'):
                contribution = (results['best_loss'] - self.ablation_metrics['baseline_loss']) / self.ablation_metrics['baseline_loss']
                self.ablation_metrics['module_contributions'][variant_name] = contribution
        
        # 2. 记录效率指标
        self.ablation_metrics['efficiency_metrics'][variant_name] = {
            'performance': results['best_loss'],
            'training_time': results.get('training_time', 0),
            'params_count': self.model_metrics['params_count'].get(variant_name, 0),
            'flops': self.model_metrics['flops'].get(variant_name, 0)
        }
    
    def analyze_module_contributions(self):
        """分析各模块的贡献度"""
        if not self.ablation_metrics['module_contributions']:
            return pd.DataFrame()
        
        contributions = pd.DataFrame({
            'Module': list(self.ablation_metrics['module_contributions'].keys()),
            'Performance Impact': list(self.ablation_metrics['module_contributions'].values())
        })
        
        contributions = contributions.sort_values('Performance Impact', ascending=False)
        return contributions
    
    def analyze_efficiency_tradeoffs(self):
        """分析性能-复杂度权衡"""
        if not self.ablation_metrics['efficiency_metrics']:
            return pd.DataFrame()
        
        metrics = []
        for variant, data in self.ablation_metrics['efficiency_metrics'].items():
            metrics.append({
                'Variant': variant,
                'Performance': data['performance'],
                'Training Time (h)': data['training_time'] / 3600,
                'Parameters (M)': data['params_count'] / 1e6,
                'FLOPs (G)': data['flops'] / 1e9
            })
        
        return pd.DataFrame(metrics)
    
    def perform_statistical_tests(self):
        """执行统计显著性检验"""
        if not self.ablation_metrics['efficiency_metrics']:
            return pd.DataFrame()
        
        baseline_perf = self.ablation_metrics['baseline_loss']
        test_results = []
        
        for variant, data in self.ablation_metrics['efficiency_metrics'].items():
            if 'baseline' in variant.lower():
                continue
            
            # 计算效果量（Cohen's d）
            d = (data['performance'] - baseline_perf) / np.sqrt(
                (data['performance']**2 + baseline_perf**2) / 2
            )
            
            test_results.append({
                'Variant': variant,
                'Performance Delta': data['performance'] - baseline_perf,
                'Effect Size': d,
                'Effect Magnitude': self._interpret_effect_size(d)
            })
        
        return pd.DataFrame(test_results)
    
    def _interpret_effect_size(self, d):
        """解释效果量大小"""
        if abs(d) < 0.2:
            return "可忽略"
        elif abs(d) < 0.5:
            return "小"
        elif abs(d) < 0.8:
            return "中等"
        else:
            return "大"
    
    def plot_ablation_analysis(self):
        """绘制消融分析图表"""
        plt.figure(figsize=(20, 15))
        
        # 1. 模块贡献度对比
        plt.subplot(2, 2, 1)
        self._plot_module_contributions()
        
        # 2. 性能-参数量权衡
        plt.subplot(2, 2, 2)
        self._plot_performance_vs_params()
        
        # 3. 性能-计算量权衡
        plt.subplot(2, 2, 3)
        self._plot_performance_vs_flops()
        
        # 4. 训练时间对比
        plt.subplot(2, 2, 4)
        self._plot_training_times()
        
        plt.tight_layout()
        save_path = self.save_dir / 'ablation_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_module_contributions(self):
        """绘制模块贡献度图"""
        contributions = self.analyze_module_contributions()
        if contributions.empty:
            return
        
        plt.bar(contributions['Module'], contributions['Performance Impact'])
        plt.title('模块贡献度分析')
        plt.xticks(rotation=45)
        plt.ylabel('性能影响 (%)')
        plt.grid(True, alpha=0.3)
    
    def _plot_performance_vs_params(self):
        """绘制性能-参数量权衡图"""
        efficiency = self.analyze_efficiency_tradeoffs()
        if efficiency.empty:
            return
        
        plt.scatter(efficiency['Parameters (M)'], efficiency['Performance'])
        for i, txt in enumerate(efficiency['Variant']):
            plt.annotate(txt, (efficiency['Parameters (M)'].iloc[i], efficiency['Performance'].iloc[i]))
        plt.title('性能-参数量权衡')
        plt.xlabel('参数量 (M)')
        plt.ylabel('验证损失')
        plt.grid(True, alpha=0.3)
    
    def _plot_performance_vs_flops(self):
        """绘制性能-计算量权衡图"""
        efficiency = self.analyze_efficiency_tradeoffs()
        if efficiency.empty:
            return
        
        plt.scatter(efficiency['FLOPs (G)'], efficiency['Performance'])
        for i, txt in enumerate(efficiency['Variant']):
            plt.annotate(txt, (efficiency['FLOPs (G)'].iloc[i], efficiency['Performance'].iloc[i]))
        plt.title('性能-计算量权衡')
        plt.xlabel('FLOPs (G)')
        plt.ylabel('验证损失')
        plt.grid(True, alpha=0.3)
    
    def _plot_training_times(self):
        """绘制训练时间对比图"""
        efficiency = self.analyze_efficiency_tradeoffs()
        if efficiency.empty:
            return
        
        plt.bar(efficiency['Variant'], efficiency['Training Time (h)'])
        plt.title('训练时间对比')
        plt.xticks(rotation=45)
        plt.ylabel('训练时间 (小时)')
        plt.grid(True, alpha=0.3)
    
    def generate_ablation_report(self):
        """生成消融分析报告"""
        try:
            # 创建reports目录
            reports_dir = self.save_dir / 'reports'
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            report = ["# 消融实验分析报告\n"]
            
            # 1. 模块贡献度分析
            report.append("\n## 1. 模块贡献度分析")
            contributions = self.analyze_module_contributions()
            if not contributions.empty:
                report.append("\n各模块对模型性能的影响：")
                report.append(contributions.to_markdown(index=False))
            
            # 2. 性能-复杂度权衡分析
            report.append("\n## 2. 性能-复杂度权衡分析")
            efficiency = self.analyze_efficiency_tradeoffs()
            if not efficiency.empty:
                report.append("\n各变体的效率指标：")
                report.append(efficiency.to_markdown(index=False))
            
            # 3. 统计显著性分析
            report.append("\n## 3. 统计显著性分析")
            stats = self.perform_statistical_tests()
            if not stats.empty:
                report.append("\n与基线模型的对比：")
                report.append(stats.to_markdown(index=False))
            
            # 4. 结论和建议
            report.append("\n## 4. 结论和建议")
            
            # 4.1 最重要的模块
            if not contributions.empty:
                most_important = contributions.iloc[0]
                report.append(f"\n### 4.1 最关键模块")
                report.append(f"- {most_important['Module']}")
                report.append(f"- 性能影响: {most_important['Performance Impact']:.2%}")
            
            # 4.2 效率建议
            report.append("\n### 4.2 效率建议")
            if not efficiency.empty:
                best_efficiency = efficiency.sort_values('Performance').iloc[0]
                report.append(f"最佳性能-效率平衡的变体：{best_efficiency['Variant']}")
                report.append(f"- 验证损失: {best_efficiency['Performance']:.4f}")
                report.append(f"- 参数量: {best_efficiency['Parameters (M)']:.2f}M")
                report.append(f"- FLOPs: {best_efficiency['FLOPs (G)']:.2f}G")
            
            # 生成分析图表
            self.plot_ablation_analysis()
            
            # 保存报告
            report_path = reports_dir / 'ablation_report.md'
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report))
            
            self.logger.info(f"消融分析报告已保存至: {report_path}")
            
        except Exception as e:
            self.logger.error(f"生成消融分析报告时出错: {str(e)}")
            raise
    
    def setup_logger(self):
        """设置日志记录器"""
        self.logger = logging.getLogger('ablation_analyzer')
        self.logger.setLevel(logging.INFO)
        
        # 创建日志目录
        log_dir = self.save_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建文件处理器
        log_file = log_dir / 'AblationAnalyzer.log'
        fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')  # 使用'w'模式并指定编码
        fh.setLevel(logging.INFO)
        
        # 创建控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)  # 改为INFO级别以显示更多信息
        
        # 创建格式化器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # 清除现有的处理器
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # 添加处理器
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        self.logger.info("初始化ablation分析器")
