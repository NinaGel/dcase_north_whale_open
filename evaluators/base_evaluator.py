import matplotlib
matplotlib.use('Agg')  # 设置后端为非交互式
from pathlib import Path
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from typing import Dict, Any, List, Union
from torch.profiler import profile, record_function, ProfilerActivity


class BaseEvaluator:
    """基础评估器类
    
    负责记录和评估模型的基础指标，生成基础训练报告
    """
    
    def __init__(self, save_dir: Path):
        """初始化评估器
        
        Args:
            save_dir: 保存目录
        """
        self.save_dir = Path(save_dir)
        self.setup_logger()
        
        # 训练过程中记录的指标
        self.training_metrics = {
            'train_losses': {},      # 训练损失历史
            'val_losses': {},        # 验证损失历史
            'snr_val_losses': {},    # 每个SNR组的验证损失
            'best_epochs': {},       # 最佳epoch
            'training_times': {},    # 训练时间
            'memory_usage': {},      # 内存使用
        }
        
        # 模型基础指标
        self.model_metrics = {
            'params_count': {},      # 参数量
            'flops': {},            # 计算量
            'model_size': {},       # 模型大小
        }
    
    def add_training_metrics(self, variant_name: str, training_time: float,
                           val_losses: Union[Dict[str, List[float]], List[float]], best_epoch: int,
                           best_loss: float):
        """记录训练过程中的指标
        
        Args:
            variant_name: 模型变体名称
            training_time: 训练耗时
            val_losses: 验证损失历史（可以是字典或列表）
            best_epoch: 最佳epoch
            best_loss: 最佳损失
        """
        self.training_metrics['training_times'][variant_name] = training_time
        self.training_metrics['val_losses'][variant_name] = val_losses
        self.training_metrics['best_epochs'][variant_name] = best_epoch
        
        # 记录每个SNR组的验证损失
        snr_losses = {}
        if isinstance(val_losses, dict):
            # 如果是字典类型，按SNR组处理
            for snr_group, losses in val_losses.items():
                snr_losses[snr_group] = {
                    'final_loss': losses[-1],
                    'best_loss': min(losses),
                    'mean_loss': sum(losses) / len(losses)
                }
        else:
            # 如果是列表类型，作为单个组处理
            snr_losses['default'] = {
                'final_loss': val_losses[-1],
                'best_loss': min(val_losses),
                'mean_loss': sum(val_losses) / len(val_losses)
            }
        
        self.training_metrics['snr_val_losses'][variant_name] = snr_losses
        
        # 记录内存使用情况
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            memory_cached = torch.cuda.memory_reserved() / 1024**2  # MB
            self.training_metrics['memory_usage'][variant_name] = {
                'allocated': memory_allocated,
                'cached': memory_cached
            }
    
    def add_model_metrics(self, variant_name: str, model: torch.nn.Module,
                         dummy_input: torch.Tensor):
        """记录模型基础指标
        
        Args:
            variant_name: 模型变体名称
            model: 模型实例
            dummy_input: 示例输入
        """
        try:
            # 检查输入维度
            self.logger.info(f"输入张量维度: {dummy_input.shape}")
            if dummy_input.dim() != 4:
                self.logger.warning(f"输入张量维度不是4D: {dummy_input.shape}")
                if dummy_input.dim() == 5:
                    dummy_input = dummy_input.squeeze(2)
                    self.logger.info(f"已将5D张量转换为4D: {dummy_input.shape}")
            
            # 保存原始模型状态
            original_dtype = next(model.parameters()).dtype
            
            # 确保使用 float32 类型
            dummy_input = dummy_input.to(torch.float32)
            model = model.to(torch.float32)
            
            # 确保所有参数和缓冲区都是 float32
            for param in model.parameters():
                param.data = param.data.to(torch.float32)
            for buffer in model.buffers():
                buffer.data = buffer.data.to(torch.float32)
            
            # 计算参数量
            params_count = sum(p.numel() for p in model.parameters())
            
            # 计算FLOPs和内存使用
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_flops=True,
                profile_memory=True,
                with_stack=True
            ) as prof:
                with record_function("model_inference"):
                    # 再次检查维度
                    self.logger.info(f"推理前输入维度: {dummy_input.shape}")
                    with torch.no_grad():
                        _ = model(dummy_input)
            
            flops = sum(event.flops for event in prof.key_averages() if event.flops > 0)
            
            # 计算模型大小
            model_size = sum(p.nelement() * p.element_size() for p in model.parameters()) / 1024**2  # MB
            
            self.model_metrics['params_count'][variant_name] = params_count
            self.model_metrics['flops'][variant_name] = flops
            self.model_metrics['model_size'][variant_name] = model_size
            
            self.logger.info(f"{variant_name} 模型指标:")
            self.logger.info(f"参数量: {params_count:,}")
            self.logger.info(f"FLOPs: {flops:,}")
            self.logger.info(f"模型大小: {model_size:.2f} MB")
            
            # 恢复模型的原始精度
            model = model.to(original_dtype)
            for param in model.parameters():
                param.data = param.data.to(original_dtype)
            for buffer in model.buffers():
                buffer.data = buffer.data.to(original_dtype)
            
        except Exception as e:
            self.logger.error(f"计算 {variant_name} 模型指标时出错: {str(e)}")
            raise  # 抛出异常以便进行调试
    
    def generate_training_report(self):
        """生成训练过程的报告"""
        report = ["# 模型训练报告\n"]
        
        # 1. 训练概况
        report.append("## 1. 训练概况")
        for variant, time in self.training_metrics['training_times'].items():
            report.append(f"\n### {variant}")
            report.append(f"- 训练时间: {time/3600:.2f} 小时")
            report.append(f"- 最佳epoch: {self.training_metrics['best_epochs'][variant]}")
            
            # 添加内存使用信息
            if variant in self.training_metrics['memory_usage']:
                memory = self.training_metrics['memory_usage'][variant]
                report.append(f"- 显存占用:")
                report.append(f"  - 分配: {memory['allocated']:.2f} MB")
                report.append(f"  - 缓存: {memory['cached']:.2f} MB")
        
        # 2. 模型规模
        report.append("\n## 2. 模型规模")
        report.append("\n| 模型变体 | 参数量 | FLOPs | 模型大小(MB) |")
        report.append("|----------|---------|--------|--------------|")
        for variant in self.model_metrics['params_count'].keys():
            params = self.model_metrics['params_count'][variant]
            flops = self.model_metrics['flops'][variant]
            size = self.model_metrics['model_size'][variant]
            report.append(f"| {variant} | {params:,} | {flops:,} | {size:.2f} |")
        
        # 3. SNR组性能分析
        report.append("\n## 3. SNR组性能分析")
        for variant, snr_losses in self.training_metrics['snr_val_losses'].items():
            report.append(f"\n### {variant}")
            report.append("\n| SNR组 | 最终损失 | 最佳损失 | 平均损失 |")
            report.append("|--------|----------|----------|----------|")
            for snr_group, metrics in snr_losses.items():
                report.append(f"| {snr_group} | {metrics['final_loss']:.4f} | {metrics['best_loss']:.4f} | {metrics['mean_loss']:.4f} |")
        
        # 保存报告
        report_path = self.save_dir / 'training_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
            
        self.logger.info(f"训练报告已保存至: {report_path}")
    
    def setup_logger(self):
        """设置日志记录器"""
        self.logger = logging.getLogger('base_evaluator')
        self.logger.setLevel(logging.INFO)
        
        # 创建日志目录
        log_dir = self.save_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建文件处理器
        log_file = log_dir / 'BaseEvaluator.log'
        fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        fh.setLevel(logging.INFO)
        
        # 创建控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # 创建格式化器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        self.logger.info("初始化base评估器")
    
    def save_model_results(self, variant: str, best_model_state: Dict, 
                          last_model_state: Dict, train_losses: list, 
                          val_losses: list, save_dirs: Dict[str, Path]):
        """保存模型训练结果"""
        try:
            # 创建模型保存目录
            model_dir = save_dirs['models'] / variant
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存模型权重
            torch.save(best_model_state, model_dir / 'best_epoch.pth')
            torch.save(last_model_state, model_dir / 'last_epoch.pth')
            
            # 保存训练曲线
            curves_path = save_dirs['curves'] / variant
            curves_path.mkdir(parents=True, exist_ok=True)
            
            # 使用 Agg 后端绘图
            plt.switch_backend('Agg')
            fig = plt.figure(figsize=(10, 6))
            
            # 绘制训练损失
            plt.plot(train_losses, label='Train Loss')
            
            # 处理验证损失
            if isinstance(val_losses, dict):
                # 如果是字典类型，为每个SNR组绘制一条曲线
                for snr_group, losses in val_losses.items():
                    plt.plot(losses, label=f'Val Loss ({snr_group})')
            else:
                # 如果是列表类型，直接绘制
                plt.plot(val_losses, label='Val Loss')
            
            plt.title(f'{variant} Training Curves')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(curves_path / 'training_curve.png')
            plt.close(fig)  # 确保关闭图形
            
            # 保存训练指标
            metrics_path = save_dirs['metrics'] / f'{variant}_metrics.json'
            
            # 计算最佳损失和最终损失
            if isinstance(val_losses, dict):
                # 获取第一个SNR组的损失列表
                first_group_losses = list(val_losses.values())[0]
                best_loss = min(first_group_losses)
                final_loss = first_group_losses[-1]
            else:
                best_loss = min(val_losses)
                final_loss = val_losses[-1]
            
            metrics = {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_loss': best_loss,
                'final_loss': final_loss
            }
            
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            self.logger.info(f"成功保存 {variant} 模型的训练结果")
            
        except Exception as e:
            self.logger.error(f"保存 {variant} 模型的训练结果时出错: {str(e)}")
            raise