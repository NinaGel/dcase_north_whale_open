import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import seaborn as sns
from thop import profile
import torch.cuda as cuda
from scipy.stats import entropy
from scipy.spatial.distance import cosine
from torch.utils.data import DataLoader

from .base_evaluator import BaseEvaluator
import config as cfg
from Train.train_utils import plot_losses, BaseTrainer
from Data.audio_dataset import load_test_data

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_test_data(batch_size: int) -> torch.Tensor:
    """生成测试数据
    
    Args:
        batch_size: 批次大小
        
    Returns:
        torch.Tensor: 形状为 [batch_size, freq, time] 的测试数据
    """
    # 从配置中获取维度
    freq = cfg.AUDIO_CONFIG['freq']
    time = cfg.AUDIO_CONFIG['frame']
    
    # 生成正确形状的数据
    x = torch.randn(batch_size, freq, time)
    return x.to(torch.float32)  # 确保数据类型为 float32


class AttentionEvaluator:
    """注意力机制评估器
    
    专注于分析注意力机制的特定特征：
    1. 注意力模式分析
    2. 注意力权重分布分析
    3. 注意力头贡献度分析
    4. 跨头注意力一致性分析
    5. 时序注意力分析
    """
    
    def __init__(self, save_dir: Path):
        """初始化注意力评估器"""
        self.save_dir = Path(save_dir)
        self.setup_logger()
        
        # 注意力评估指标
        self.attention_metrics = {
            'attention_patterns': {},     # 注意力模式
            'head_importance': {},        # 注意力头重要性
            'head_diversity': {},         # 注意力头多样性
            'temporal_attention': {},     # 时序注意力分布
            'cross_head_similarity': {},  # 跨头相似度
            'hybrid_gates': {},            # 混合注意力门控机制
            'snr_analysis': {},          # SNR级别对比分析
            'event_detection': {},       # 事件检测分析
            'variant_comparison': {},    # 变体对比分析
            'temporal_analysis': {},     # 时序分析
            'performance_analysis': {}   # 性能关联分析
        }
        
        # 创建保存目录
        for metric in self.attention_metrics.keys():
            (self.save_dir / metric).mkdir(parents=True, exist_ok=True)
    
    def setup_logger(self):
        """设置日志记录器"""
        self.logger = logging.getLogger('attention_evaluator')
        self.logger.setLevel(logging.INFO)
        
        log_dir = self.save_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / 'AttentionEvaluator.log'
        fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        fh.setLevel(logging.INFO)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        self.logger.info("初始化attention评估器")
    
    def analyze_attention_pattern(self, variant_name: str, attention_weights: Dict[str, torch.Tensor],
                                layer_idx: Optional[int] = None):
        """分析注意力模式"""
        for snr_level, weights in attention_weights.items():
            if weights['weights'].dim() != 4:
                raise ValueError(f"{snr_level}级别的注意力权重应为4维张量: [batch, heads, seq_len, seq_len]")
                
            # 计算平均注意力模式
            avg_pattern = weights['weights'].mean(0)  # [heads, seq_len, seq_len]
            
            # 计算注意力熵
            attention_probs = F.softmax(avg_pattern, dim=-1)
            attention_entropy = -(attention_probs * torch.log(attention_probs + 1e-9)).sum(dim=-1)
            attention_entropy = attention_entropy.mean(dim=-1).cpu().numpy()
            
            # 计算注意力稀疏度
            sparsity = (attention_probs < 0.1).float().mean(dim=(1,2)).cpu().numpy()
            
            # 计算注意力集中度
            max_attention = attention_probs.max(dim=-1)[0].mean(dim=-1).cpu().numpy()
            
            # 计算行和列的注意力分布
            row_attention = attention_probs.mean(dim=2).cpu().numpy()  # 按列平均
            col_attention = attention_probs.mean(dim=1).cpu().numpy()  # 按行平均
            
            # 对于Hybrid模型，额外分析门控机制
            if variant_name == 'Hybrid':
                gate_values = self._analyze_hybrid_gates(weights['weights'])
                if snr_level not in self.attention_metrics['hybrid_gates']:
                    self.attention_metrics['hybrid_gates'][snr_level] = {}
                self.attention_metrics['hybrid_gates'][snr_level][variant_name] = gate_values
            
            # 保存结果
            key = f"{variant_name}_{snr_level}"
            if layer_idx is not None:
                key += f"_layer{layer_idx}"
                
            self.attention_metrics['attention_patterns'][key] = {
                'pattern': avg_pattern.cpu().numpy(),
                'entropy': attention_entropy,
                'sparsity': sparsity,
                'max_attention': max_attention,
                'row_attention': row_attention,
                'col_attention': col_attention,
                'mean_attention': attention_probs.mean().item(),
                'std_attention': attention_probs.std().item()
            }
            
            self.logger.info(f"完成{key}的注意力模式分析")
    
    def analyze_head_diversity(self, variant_name: str, attention_weights: Dict[str, torch.Tensor]):
        """分析注意力头的多样性"""
        for snr_level, weights in attention_weights.items():
            try:
                if weights is None:
                    self.logger.warning(f"未收集到{snr_level}级别的注意力数据")
                    continue
                    
                num_heads = weights['weights'].size(1)
                diversity_matrix = torch.zeros((num_heads, num_heads))
                
                self.logger.info(f"正在分析{snr_level}级别的头多样性（{num_heads}个头）...")
                
                # 计算头间相似度
                for i in range(num_heads):
                    for j in range(num_heads):
                        try:
                            # 将注意力权重展平为2D矩阵 [batch_size, seq_len * seq_len]
                            head_i = weights['weights'][:,i].reshape(weights['weights'].size(0), -1)
                            head_j = weights['weights'][:,j].reshape(weights['weights'].size(0), -1)
                            
                            # 计算余弦相似度
                            similarity = F.cosine_similarity(head_i, head_j, dim=1).mean()
                            diversity_matrix[i,j] = 1 - similarity  # 转换为多样性分数
                        except Exception as e:
                            self.logger.error(f"计算头{i}和头{j}的相似度时出错: {str(e)}")
                            diversity_matrix[i,j] = 0.0
                
                key = f"{variant_name}_{snr_level}"
                self.attention_metrics['head_diversity'][key] = diversity_matrix.cpu().numpy()
                self.logger.info(f"完成{key}的头多样性分析")
                
            except Exception as e:
                self.logger.error(f"分析{snr_level}级别的头多样性时出错: {str(e)}")
                continue
    
    def analyze_temporal_attention(self, variant_name: str, attention_weights: Dict[str, torch.Tensor]):
        """分析时序注意力分布"""
        for snr_level, weights in attention_weights.items():
            try:
                if weights is None:
                    self.logger.warning(f"未收集到{snr_level}级别的注意力数据")
                    continue
                
                self.logger.info(f"正在分析{snr_level}级别的时序注意力...")
                
                # 计算平均时序注意力分布
                temporal_dist = weights['weights'].mean(dim=(0,1))  # [seq_len, seq_len]
                
                # 计算局部注意力比例（对角线附近）
                local_attention = torch.zeros_like(temporal_dist)
                window_size = 5
                for i in range(temporal_dist.size(0)):
                    start = max(0, i - window_size)
                    end = min(temporal_dist.size(0), i + window_size + 1)
                    local_attention[i, start:end] = 1
                
                local_ratio = (temporal_dist * local_attention).sum() / temporal_dist.sum()
                
                # 计算全局注意力模式
                seq_len = temporal_dist.size(0)
                forward_mask = torch.triu(torch.ones_like(temporal_dist), diagonal=1)
                backward_mask = torch.tril(torch.ones_like(temporal_dist), diagonal=-1)
                diagonal_mask = torch.eye(seq_len, device=temporal_dist.device)
                
                total_attention = temporal_dist.sum()
                forward_ratio = (temporal_dist * forward_mask).sum() / total_attention
                backward_ratio = (temporal_dist * backward_mask).sum() / total_attention
                diagonal_ratio = (temporal_dist * diagonal_mask).sum() / total_attention
                
                key = f"{variant_name}_{snr_level}"
                self.attention_metrics['temporal_attention'][key] = {
                    'distribution': temporal_dist.cpu().numpy(),
                    'local_ratio': local_ratio.item(),
                    'global_pattern': {
                        'forward_ratio': forward_ratio.item(),
                        'backward_ratio': backward_ratio.item(),
                        'diagonal_ratio': diagonal_ratio.item()
                    }
                }
                
                self.logger.info(f"完成{key}的时序注意力分析")
                
            except Exception as e:
                self.logger.error(f"分析{snr_level}级别的时序注意力时出错: {str(e)}")
                continue
    
    def _analyze_hybrid_gates(self, attention_weights):
        """分析混合注意力的门控机制"""
        # 提取门控值（假设在attention_weights中的最后一个维度）
        if attention_weights.size(-1) > 1:
            gate_values = attention_weights[..., -1]
            return {
                'mean_gate': gate_values.mean().item(),
                'std_gate': gate_values.std().item(),
                'max_gate': gate_values.max().item(),
                'min_gate': gate_values.min().item(),
                'histogram': torch.histc(gate_values, bins=10, min=0, max=1).cpu().numpy()
            }
        return None
    
    def generate_attention_report(self):
        """生成注意力分析报告"""
        if not any(self.attention_metrics.values()):
            self.logger.warning("没有可用的注意力分析数据，生成空报告")
            report = ["# 注意力机制分析报告\n", "**注意**: 没有可用的注意力分析数据"]
            report_path = self.save_dir / 'attention_analysis_report.md'
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report))
            return
        
        report = ["# 注意力机制分析报告\n"]
        
        # 1. 注意力模式分析
        if self.attention_metrics['attention_patterns']:
            report.append("## 1. 注意力模式分析")
            for variant, patterns in self.attention_metrics['attention_patterns'].items():
                report.append(f"\n### {variant}")
                report.append("#### 注意力特征")
                report.append(f"- 平均注意力强度: {patterns['mean_attention']:.4f}")
                report.append(f"- 注意力分布标准差: {patterns['std_attention']:.4f}")
                report.append(f"- 平均注意力熵: {patterns['entropy'].mean():.4f}")
                report.append(f"- 平均稀疏度: {patterns['sparsity'].mean():.4f}")
                report.append(f"- 平均最大注意力值: {patterns['max_attention'].mean():.4f}")
                
                report.append("\n#### 注意力分布特征")
                report.append("- 行注意力分布（按时间步长）:")
                for i, row_attn in enumerate(patterns['row_attention']):
                    report.append(f"  - 头 {i}: 最大值 {row_attn.max():.4f}, 最小值 {row_attn.min():.4f}")
                
                report.append("\n- 列注意力分布（按特征）:")
                for i, col_attn in enumerate(patterns['col_attention']):
                    report.append(f"  - 头 {i}: 最大值 {col_attn.max():.4f}, 最小值 {col_attn.min():.4f}")
        
        # 2. 头多样性分析
        if self.attention_metrics['head_diversity']:
            report.append("\n## 2. 头多样性分析")
            for variant, diversity in self.attention_metrics['head_diversity'].items():
                if not isinstance(diversity, np.ndarray):
                    continue
                report.append(f"\n### {variant}")
                report.append("#### 多样性统计")
                report.append(f"- 平均多样性: {diversity.mean():.4f}")
                report.append(f"- 多样性标准差: {diversity.std():.4f}")
                report.append(f"- 最大多样性: {diversity.max():.4f}")
                report.append(f"- 最小多样性: {diversity.min():.4f}")
                
                # 添加头间相似度分析
                report.append("\n#### 头间相似度分析")
                for i in range(len(diversity)):
                    similar_heads = np.where(diversity[i] < 0.3)[0]  # 相似度阈值0.7
                    if len(similar_heads) > 1:  # 排除自身
                        similar_heads = similar_heads[similar_heads != i]
                        if len(similar_heads) > 0:
                            report.append(f"- 头 {i} 与头 {similar_heads.tolist()} 具有高相似度")
        
        # 3. 时序注意力分析
        if self.attention_metrics['temporal_attention']:
            report.append("\n## 3. 时序注意力分析")
            for variant, data in self.attention_metrics['temporal_attention'].items():
                if not isinstance(data, dict):
                    continue
                report.append(f"\n### {variant}")
                report.append("#### 时序特征")
                report.append(f"- 局部注意力比例: {data['local_ratio']:.4f}")
                
                # 分析全局注意力模式
                global_pattern = data['global_pattern']
                report.append("\n#### 全局注意力模式")
                report.append(f"- 前向注意力比例: {global_pattern['forward_ratio']:.4f}")
                report.append(f"- 后向注意力比例: {global_pattern['backward_ratio']:.4f}")
                report.append(f"- 对角注意力比例: {global_pattern['diagonal_ratio']:.4f}")
                
                # 添加时序模式分析
                report.append("\n#### 时序模式分析")
                pattern_type = "前向主导" if global_pattern['forward_ratio'] > max(global_pattern['backward_ratio'], 
                                                                            global_pattern['diagonal_ratio']) else \
                             "后向主导" if global_pattern['backward_ratio'] > max(global_pattern['forward_ratio'], 
                                                                            global_pattern['diagonal_ratio']) else \
                             "局部主导"
                report.append(f"- 主导模式: {pattern_type}")
                report.append(f"- 时序依赖强度: {max(global_pattern['forward_ratio'], global_pattern['backward_ratio']):.4f}")
                report.append(f"- 局部依赖强度: {global_pattern['diagonal_ratio']:.4f}")
        
        # 4. 混合注意力分析（如果有）
        if self.attention_metrics['hybrid_gates']:
            report.append("\n## 4. 混合注意力分析")
            for variant, gates in self.attention_metrics['hybrid_gates'].items():
                if not isinstance(gates, dict):
                    continue
                report.append(f"\n### {variant}")
                report.append("#### 门控机制统计")
                report.append(f"- 平均DSA权重: {gates['dsa_weight_mean']:.4f}")
                report.append(f"- 平均LDSA权重: {gates['ldsa_weight_mean']:.4f}")
                report.append(f"- DSA权重标准差: {gates['dsa_weight_std']:.4f}")
                report.append(f"- LDSA权重标准差: {gates['ldsa_weight_std']:.4f}")
                
                # 添加门控行为分析
                report.append("\n#### 门控行为分析")
                dominant = "DSA" if gates['dsa_weight_mean'] > gates['ldsa_weight_mean'] else "LDSA"
                report.append(f"- 主导机制: {dominant}")
                report.append(f"- 机制选择稳定性: {1 - min(gates['dsa_weight_std'], gates['ldsa_weight_std']):.4f}")
                
                if 'gate_patterns' in gates:
                    report.append("\n#### 门控模式分析")
                    for pattern, freq in gates['gate_patterns'].items():
                        report.append(f"- {pattern}: {freq:.2%}")
        
        # 保存报告
        report_path = self.save_dir / 'attention_analysis_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        self.logger.info(f"注意力分析报告已保存至: {report_path}")
    
    def plot_attention_analysis(self):
        """绘制注意力分析图表"""
        try:
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
            plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
            
            snr_levels = ['high', 'medium', 'low', 'very_low']
            for snr_level in snr_levels:
                plt.figure(figsize=(20, 15))
                
                # 1. 注意力模式热图
                plt.subplot(2, 2, 1)
                self._plot_attention_patterns(snr_level)
                
                # 2. 头多样性热图
                plt.subplot(2, 2, 2)
                self._plot_head_diversity(snr_level)
                
                # 3. 时序注意力分布
                plt.subplot(2, 2, 3)
                self._plot_temporal_attention(snr_level)
                
                # 4. 混合注意力门控分析（如果有）
                plt.subplot(2, 2, 4)
                self._plot_hybrid_gates(snr_level)
                
                plt.suptitle(f'SNR {snr_level} 级别注意力分析', fontsize=16)
                plt.tight_layout()
                
                # 为每个SNR级别保存单独的图表
                save_path = self.save_dir / f'attention_analysis_{snr_level}.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                self.logger.info(f"SNR {snr_level} 级别的注意力分析图表已保存至: {save_path}")
            
        except Exception as e:
            self.logger.error(f"绘制注意力分析图表时出错: {str(e)}")
            raise
    
    def _plot_attention_patterns(self, snr_level: str):
        """绘制注意力模式热图"""
        for key, data in self.attention_metrics['attention_patterns'].items():
            if snr_level in key:
                pattern = data['pattern']
                plt.imshow(pattern.mean(0), aspect='auto', cmap='viridis')
                plt.title(f'注意力模式\n熵值: {data["entropy"].mean():.2f}, 稀疏度: {data["sparsity"].mean():.2f}')
                plt.colorbar(label='注意力权重')
                plt.xlabel('序列位置')
                plt.ylabel('序列位置')
    
    def _plot_head_diversity(self, snr_level: str):
        """绘制头多样性热图"""
        for key, diversity in self.attention_metrics['head_diversity'].items():
            if snr_level in key:
                sns.heatmap(diversity, cmap='YlOrRd', center=0)
                plt.title(f'头多样性\n平均多样性: {diversity.mean():.2f}')
                plt.xlabel('注意力头')
                plt.ylabel('注意力头')
    
    def _plot_temporal_attention(self, snr_level: str):
        """绘制时序注意力分布"""
        for key, data in self.attention_metrics['temporal_attention'].items():
            if snr_level in key:
                temporal_dist = data['distribution']
                plt.imshow(temporal_dist, aspect='auto', cmap='viridis')
                plt.title(f'时序注意力\n局部注意力比例: {data["local_ratio"]:.2f}')
                plt.colorbar(label='时序注意力权重')
                plt.xlabel('时间步')
                plt.ylabel('时间步')
    
    def _plot_hybrid_gates(self, snr_level: str):
        """绘制混合注意力门控分析"""
        if snr_level not in self.attention_metrics['hybrid_gates']:
            plt.text(0.5, 0.5, '无混合注意力数据', 
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=plt.gca().transAxes)
            plt.title('混合注意力门控分析')
            return
            
        for variant, gates in self.attention_metrics['hybrid_gates'][snr_level].items():
            if 'histogram' in gates:
                plt.bar(range(10), gates['histogram'], alpha=0.6)
                plt.title(f'门控值分布\n均值: {gates["mean_gate"]:.2f}, 标准差: {gates["std_gate"]:.2f}')
                plt.xlabel('门控值区间')
                plt.ylabel('频次')
    
    def collect_attention_data(self, model: torch.nn.Module, data_loader: DataLoader, num_batches: int = 20) -> Dict[str, torch.Tensor]:
        """收集注意力数据
        
        Args:
            model: 模型实例
            data_loader: 数据加载器
            num_batches: 要分析的批次数
            
        Returns:
            Dict[str, torch.Tensor]: 包含注意力权重和其他相关数据的字典
        """
        device = next(model.parameters()).device
        attention_data = {
            'weights': [],
            'outputs': [],
            'inputs': []
        }
        
        model.eval()
        batch_count = 0
        with torch.no_grad():
            for inputs, labels, _ in data_loader:
                if batch_count >= num_batches:
                    break
                    
                inputs = inputs.to(device)
                
                # 前向传播，收集注意力权重
                outputs = model(inputs)
                attention_data['outputs'].append(outputs.cpu())
                attention_data['inputs'].append(inputs.cpu())
                
                # 收集注意力权重
                if hasattr(model, 'attention_weights'):
                    attention_data['weights'].append(model.attention_weights.cpu())
                
                batch_count += 1
        
        # 检查是否收集到数据
        if not attention_data['weights']:
            self.logger.warning("未收集到注意力权重")
            return None
            
        # 合并数据
        attention_data['weights'] = torch.cat(attention_data['weights'], dim=0)
        attention_data['outputs'] = torch.cat(attention_data['outputs'], dim=0)
        attention_data['inputs'] = torch.cat(attention_data['inputs'], dim=0)
        
        self.logger.info(f"收集到{len(attention_data['weights'])}个样本的注意力数据")
        
        return attention_data
    
    def analyze_snr_attention(self, model: torch.nn.Module, audio_samples: Dict[str, torch.Tensor]):
        """不同SNR级别下的注意力机制对比分析"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(20, 15))
            snr_levels = ['high', 'medium', 'low', 'very_low']
            
            # 获取模型所在设备
            device = next(model.parameters()).device
            
            for idx, snr in enumerate(snr_levels):
                if snr not in audio_samples:
                    continue
                    
                row = idx // 2
                col = idx % 2
                ax = axes[row, col]
                
                # 获取当前SNR级别的样本
                audio = audio_samples[snr]
                if isinstance(audio, torch.Tensor):
                    audio = audio.to(device)
                
                # 绘制ACT频谱图
                spectrogram = audio[0].cpu().numpy()
                im = ax.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
                plt.colorbar(im, ax=ax, label='Magnitude')
                
                # 获取并叠加注意力热力图
                with torch.no_grad():
                    _ = model(audio)
                    if hasattr(model, 'attention_weights'):
                        attention = model.attention_weights.mean(dim=1)[0].cpu().numpy()
                        attention_normalized = (attention - attention.min()) / (attention.max() - attention.min())
                        ax.imshow(attention_normalized, alpha=0.5, cmap='hot')
                
                ax.set_title(f'SNR {snr.upper()}')
                ax.set_xlabel('Time Frame')
                ax.set_ylabel('Frequency')
            
            plt.suptitle('Attention Analysis Across SNR Levels', fontsize=16)
            plt.tight_layout()
            save_path = self.save_dir / 'snr_analysis' / 'snr_attention_comparison.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"SNR level attention comparison saved to: {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error in SNR level attention analysis: {str(e)}")
            raise
    
    def visualize_whale_detection(self, model: torch.nn.Module, audio_sample: torch.Tensor, 
                                true_events: torch.Tensor, snr_level: str):
        """分析模型如何通过注意力机制定位鲸鱼声音"""
        try:
            fig = plt.figure(figsize=(15, 12))
            gs = plt.GridSpec(3, 1, height_ratios=[2, 2, 1.5])
            
            # 确保数据在正确的设备上
            device = next(model.parameters()).device
            audio_sample = audio_sample.to(device)
            true_events = true_events.to(device)
            
            # 1. 原始频谱图
            ax1 = plt.subplot(gs[0])
            spectrogram = audio_sample[0].cpu().numpy()
            im1 = ax1.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
            plt.colorbar(im1, ax=ax1, label='Magnitude')
            ax1.set_title('Input ACT Spectrogram', pad=10)
            
            # 2. 注意力分布
            ax2 = plt.subplot(gs[1])
            with torch.no_grad():
                outputs = model(audio_sample)
                if hasattr(model, 'attention_weights'):
                    attention = model.attention_weights.mean(dim=1)[0].cpu().numpy()
                    attention_normalized = (attention - attention.min()) / (attention.max() - attention.min())
                    im2 = ax2.imshow(attention_normalized, aspect='auto', cmap='hot', origin='lower')
                    plt.colorbar(im2, ax=ax2, label='Attention Weight')
                    ax2.set_title('Attention Distribution', pad=10)
            
            # 3. 检测结果与真实标签对比
            ax3 = plt.subplot(gs[2])
            time_steps = np.arange(outputs.size(-1))
            
            # 确保预测和标签是一维的
            pred_probs = outputs[0].squeeze().cpu().detach().numpy()
            true_labels = true_events[0].squeeze().cpu().numpy()
            
            # 绘制预测概率
            ax3.fill_between(time_steps, np.zeros_like(pred_probs), pred_probs, 
                            alpha=0.5, color='blue', label='Prediction')
            
            # 绘制真实标签
            ax3.fill_between(time_steps, np.zeros_like(true_labels), true_labels, 
                            alpha=0.5, color='red', label='Ground Truth')
            
            ax3.set_ylim(0, 1.1)
            ax3.set_title('Detection Results', pad=10)
            ax3.set_xlabel('Time Frame')
            ax3.set_ylabel('Probability')
            ax3.legend(loc='upper right')
            
            # 添加网格线
            ax3.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            save_path = self.save_dir / 'event_detection' / f'whale_detection_{snr_level}.png'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Whale sound detection analysis saved to: {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error in whale sound detection analysis: {str(e)}")
            raise
    
    def compare_attention_variants(self, audio_sample: torch.Tensor, models: Dict[str, torch.nn.Module]):
        """比较不同注意力变体在同一样本上的注意力差异
        
        Args:
            audio_sample: 输入音频样本
            models: 不同变体的模型字典，格式为 {variant_name: model}
        """
        try:
            fig = plt.figure(figsize=(15, 12))
            
            for idx, (name, model) in enumerate(models.items(), 1):
                plt.subplot(3, 1, idx)
                
                # 获取该变体的注意力
                with torch.no_grad():
                    _ = model(audio_sample)
                    if hasattr(model, 'attention_weights'):
                        attention = model.attention_weights[0]  # [num_heads, seq_len, seq_len]
                        
                        # 使用不同颜色显示不同头的注意力
                        for head_idx in range(attention.size(0)):
                            plt.plot(attention[head_idx].mean(dim=0).cpu().numpy(), 
                                   alpha=0.5, 
                                   label=f'Head {head_idx}')
                
                plt.title(f'{name} 变体的注意力分布')
                plt.xlabel('时间帧')
                plt.ylabel('注意力权重')
                plt.legend()
            
            plt.tight_layout()
            save_path = self.save_dir / 'variant_comparison' / 'attention_variants_comparison.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"注意力变体对比分析已保存至: {save_path}")
            
        except Exception as e:
            self.logger.error(f"注意力变体对比分析出错: {str(e)}")
            raise
    
    def attention_performance_analysis(self, model: torch.nn.Module, test_dataset: DataLoader):
        """分析注意力特征与检测性能的关系
        
        Args:
            model: 训练好的模型
            test_dataset: 测试数据加载器
        """
        try:
            results = {
                'snr_level': [],
                'attention_entropy': [],
                'detection_accuracy': [],
                'attention_focus': []
            }
            
            # 获取模型所在设备
            device = next(model.parameters()).device
            
            model.eval()
            with torch.no_grad():
                for batch_idx, (audio, label, snr_indices) in enumerate(test_dataset):
                    if batch_idx >= 100:  # 限制分析的批次数
                        break
                        
                    # 将数据移到正确的设备上
                    audio = audio.to(device)
                    label = label.to(device)
                    
                    # 获取预测和注意力
                    outputs = model(audio)
                    if not hasattr(model, 'attention_weights'):
                        continue
                        
                    attention = model.attention_weights
                    
                    # 计算注意力特征
                    entropy = self._calculate_attention_entropy(attention)
                    focus = self._calculate_attention_focus(attention)
                    accuracy = (outputs.round() == label).float().mean()
                    
                    # 记录结果
                    results['snr_level'].extend([snr.item() for snr in snr_indices])
                    results['attention_entropy'].extend(entropy.cpu().numpy())
                    results['attention_focus'].extend(focus.cpu().numpy())
                    results['detection_accuracy'].extend([accuracy.item()] * len(snr_indices))
            
            # 转换为DataFrame
            results_df = pd.DataFrame(results)
            
            # 可视化
            plt.figure(figsize=(15, 5))
            
            # 1. SNR vs 检测准确率
            plt.subplot(131)
            sns.boxplot(x='snr_level', y='detection_accuracy', data=results_df)
            plt.title('SNR vs 检测准确率')
            
            # 2. 注意力熵 vs 检测准确率
            plt.subplot(132)
            plt.scatter(results['attention_entropy'], results['detection_accuracy'])
            plt.xlabel('注意力熵')
            plt.ylabel('检测准确率')
            
            # 3. 注意力聚焦度 vs 检测准确率
            plt.subplot(133)
            plt.scatter(results['attention_focus'], results['detection_accuracy'])
            plt.xlabel('注意力聚焦度')
            plt.ylabel('检测准确率')
            
            plt.tight_layout()
            save_path = self.save_dir / 'performance_analysis' / 'attention_performance_analysis.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 保存统计结果
            stats_path = self.save_dir / 'performance_analysis' / 'attention_performance_stats.csv'
            results_df.to_csv(stats_path, index=False)
            
            self.logger.info(f"注意力性能分析图表已保存至: {save_path}")
            self.logger.info(f"注意力性能统计数据已保存至: {stats_path}")
            
        except Exception as e:
            self.logger.error(f"注意力性能分析出错: {str(e)}")
            raise
    
    def _calculate_attention_entropy(self, attention: torch.Tensor) -> torch.Tensor:
        """计算注意力熵"""
        # 确保注意力权重和为1
        attention_probs = F.softmax(attention, dim=-1)
        # 计算熵
        entropy = -(attention_probs * torch.log(attention_probs + 1e-9)).sum(dim=-1)
        return entropy.mean(dim=1)  # 平均所有头的熵
    
    def _calculate_attention_focus(self, attention: torch.Tensor) -> torch.Tensor:
        """计算注意力聚焦度"""
        # 使用注意力权重的方差作为聚焦度度量
        attention_probs = F.softmax(attention, dim=-1)
        return attention_probs.var(dim=-1).mean(dim=1)  # 平均所有头的方差 