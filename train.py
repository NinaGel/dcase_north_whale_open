import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch import nn
import torch.backends.cudnn as cudnn
from torch.cuda import amp
from torch.nn.utils import clip_grad_norm_
import numpy as np

# 导入自定义模块
from Data.audio_dataset import load_train_val_data, load_test_data
import sys
sys.path.append('Model')
from Model.Whale_Model_Attention_MultiScale_Ldsa import Whale_Model_Attention_MultiScale
from Train.train_utils import BaseTrainer, plot_losses
from evaluation_metrics import WhaleEventDetectionEvaluator, load_ground_truth
import config as cfg


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="声音事件检测训练脚本")
    parser.add_argument('--epochs', type=int, default=cfg.TRAIN_CONFIG['epochs'], 
                       help='训练轮数')
    parser.add_argument('--save_model_name', type=str, default='DDSA_Model', 
                       help='模型保存名称')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--fp16', action='store_true',
                       help='是否使用混合精度训练')
    return parser.parse_args()


def save_training_results(model_state, train_losses, val_losses, test_metrics, args):
    """保存训练结果
    
    Args:
        model_state: 模型状态字典
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        test_metrics: 测试指标字典
        args: 命令行参数
    """
    save_dir = Path(cfg.PATH_CONFIG['model_save_path']) / args.save_model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存模型
    model_path = save_dir / 'best_model.pth'
    torch.save(model_state, model_path)
    
    # 保存损失曲线
    plot_losses(
        train_losses=train_losses,
        val_losses=val_losses,
        epochs=args.epochs,
        save_path=save_dir / 'training_curves'
    )
    
    # 保存训练信息和测试指标
    with open(save_dir / 'training_info.txt', 'w') as f:
        f.write(f'Model: DDSA\n')
        f.write(f'Best validation loss: {min(val_losses):.4f}\n')
        f.write(f'Final training loss: {train_losses[-1]:.4f}\n')
        f.write(f'Mixed Precision: {args.fp16}\n\n')
        
        # 写入测试指标
        f.write('Test Metrics:\n')
        for model_type, snr_metrics in test_metrics.items():
            f.write(f'\n{model_type}:\n')
            for snr_group, metrics in snr_metrics.items():
                f.write(f'\n  {snr_group}:\n')
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        f.write(f'    {metric_name}: {value:.4f}\n')
                    else:
                        f.write(f'    {metric_name}: {value}\n')


def evaluate_model(model, test_loaders, txt_folder_path):
    """评估模型性能
    
    Args:
        model: 训练好的模型
        test_loaders: 测试数据加载器
        txt_folder_path: 标签文件路径
        
    Returns:
        dict: 评估指标结果
    """
    evaluator = WhaleEventDetectionEvaluator()
    snr_metrics = {}
    device = next(model.parameters()).device
    model.eval()
    
    for snr_group, loader in test_loaders.items():
        print(f"\n评估 {snr_group} 组...")
        
        # 加载当前SNR组的标签数据
        ground_truth_df, ground_truth_dict, audio_durations = load_ground_truth(
            Path(txt_folder_path) / snr_group / 'txt'
        )
        
        all_preds = []
        all_filenames = []
        
        with torch.cuda.amp.autocast(), torch.no_grad():
            for batch in loader:
                audio, _, filenames = batch
                
                # 确保输入维度正确
                if audio.dim() == 3:
                    audio = audio.unsqueeze(1)
                
                # 转换数据类型并移动到设备
                audio = audio.to(dtype=torch.float32, device=device, non_blocking=True)
                predictions = model(audio)
                
                # 获取预测分数
                batch_scores = predictions.cpu()
                all_preds.append(batch_scores)
                
                # 修改文件名以包含SNR组信息
                modified_filenames = [f'soundscape_whale_test_{snr_group}_{Path(f).stem}' for f in filenames]
                all_filenames.extend(modified_filenames)
        
        if all_preds:  # 确保有预测结果
            all_preds = torch.cat(all_preds, dim=0)
            
            # 计算评估指标
            metrics = evaluator.compute_all_metrics(
                all_preds,
                all_filenames,
                ground_truth_dict,
                ground_truth_df,
                audio_durations
            )
            
            snr_metrics[snr_group] = metrics
            
            # 打印当前SNR组的主要指标
            print(f"PSDS分数: {metrics['psds_score']:.4f}")
            print(f"宏观F1分数: {metrics['segment_based_f1_macro']:.4f}")
            print(f"事件级F1分数: {metrics['event_based_f1']:.4f}")
    
    return snr_metrics


class DataTypeConverter:
    """数据类型转换器"""
    @staticmethod
    def convert_batch(batch, fp16=False):
        """转换批次数据类型
        
        Args:
            batch: 数据批次
            fp16: 是否使用float16
            
        Returns:
            转换后的批次数据
        """
        if isinstance(batch, (tuple, list)):
            x, y = batch[:2]
            snrs = batch[2] if len(batch) > 2 else None
        else:
            x, y = batch.x, batch.y
            snrs = batch.snrs if hasattr(batch, 'snrs') else None
        
        # 确保输入维度正确
        if x.dim() == 3:  # [batch_size, freq, time]
            x = x.unsqueeze(1)  # 添加通道维度 [batch_size, channel, freq, time]
        
        # 转换数据类型
        x = x.float()  # 先转换为float32
        if fp16:
            x = x.half()  # 如果需要，再转换为float16
        y = y.float()  # 标签始终使用float32
        
        return x, y, snrs


def train_and_evaluate(args):
    """训练和评估模型"""
    # 打印训练配置
    print("\n" + "="*50)
    print("训练配置:")
    print("-"*50)
    print(f"混合精度训练: {cfg.TRAIN_CONFIG['mixed_precision']}")
    print(f"训练轮数: {cfg.TRAIN_CONFIG['epochs']}")
    print(f"批次大小: {cfg.TRAIN_CONFIG['batch_size']}")
    print(f"学习率: {cfg.TRAIN_CONFIG['optimizer_params']['lr']}")
    print(f"权重衰减: {cfg.TRAIN_CONFIG['optimizer_params']['weight_decay']}")
    print(f"梯度裁剪: {cfg.TRAIN_CONFIG['grad_clip']}")
    print(f"梯度裁剪阈值: {cfg.TRAIN_CONFIG['grad_clip_value']}")
    print(f"早停轮数: {cfg.TRAIN_CONFIG['patience']}")
    print("="*50 + "\n")

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 设置CUDA和cuDNN
    cudnn.benchmark = cfg.GPU_CONFIG['cudnn_benchmark']
    cudnn.deterministic = cfg.GPU_CONFIG['cudnn_deterministic']
    
    # 初始化GradScaler用于混合精度训练
    scaler = amp.GradScaler(enabled=args.fp16)
    
    # 加载训练和验证数据
    train_loader, val_loaders = load_train_val_data()
    
    # 加载测试数据
    test_loaders = load_test_data()
    
    # 初始化DDSA模型
    model = Whale_Model_Attention_MultiScale()
    
    # 打印模型架构和参数量
    print("模型架构:")
    print("-"*50)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print("="*50 + "\n")
    
    # 检查初始内存占用
    if torch.cuda.is_available():
        print("GPU内存使用情况:")
        print("-"*50)
        print(f"当前分配: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"当前缓存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        print("="*50 + "\n")
    
    model = model.to(dtype=torch.float32, device=device)
    
    # 初始化优化器
    optimizer = cfg.optimizer(
        model.parameters(),
        **cfg.TRAIN_CONFIG['optimizer_params']
    )
    
    # 创建训练器
    trainer = BaseTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=cfg.loss_fn,
        device=device
    )
    
    # 设置检查点目录
    checkpoint_dir = Path(cfg.PATH_CONFIG['model_save_path']) / args.save_model_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练模型
    print("\n开始训练DDSA模型...")
    train_losses = []
    val_losses = []
    best_model_state = None
    last_model_state = None
    best_val_loss = float('inf')
    patience_counter = 0
    
    try:
        for epoch in range(args.epochs):
            trainer.current_epoch = epoch
            
            # 训练阶段
            train_loss, batch_time = trainer.train_epoch(train_loader)
            train_losses.append(train_loss)
            
            # 验证阶段
            val_loss, val_metrics = trainer.validate(val_loaders)
            val_losses.append(val_loss)
            
            # 打印当前epoch的结果
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            print(f"Train Loss: {train_loss:.4f}, Avg Batch Time: {batch_time:.3f}s")
            print(f"Val Loss: {val_loss:.4f}")
            
            # 更新最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {
                    k: v.cpu().clone() for k, v in trainer.model.state_dict().items()
                }
                patience_counter = 0
                
                # 保存最佳模型
                torch.save(best_model_state, checkpoint_dir / 'best_model.pth')
            else:
                patience_counter += 1
            
            # 早停检查
            if patience_counter >= cfg.TRAIN_CONFIG['patience']:
                print(f"\n早停触发于轮次 {epoch + 1}")
                break
            
            # 更新学习率
            if trainer.scheduler:
                trainer.scheduler.step()
        
        # 保存最后一轮的模型
        last_model_state = {
            k: v.cpu().clone() for k, v in trainer.model.state_dict().items()
        }
        torch.save(last_model_state, checkpoint_dir / 'last_model.pth')
                
    except KeyboardInterrupt:
        print("\n训练被中断！保存当前模型...")
        last_model_state = {
            k: v.cpu().clone() for k, v in trainer.model.state_dict().items()
        }
        torch.save(last_model_state, checkpoint_dir / 'interrupted_model.pth')
    
    # 评估最佳模型
    print("\n开始评估最佳模型...")
    trainer.model.load_state_dict(best_model_state)
    best_test_metrics = evaluate_model(
        trainer.model,
        test_loaders,
        cfg.PATH_CONFIG['test_path']
    )
    
    # 评估最后一轮模型
    print("\n开始评估最后一轮模型...")
    trainer.model.load_state_dict(last_model_state)
    last_test_metrics = evaluate_model(
        trainer.model,
        test_loaders,
        cfg.PATH_CONFIG['test_path']
    )
    
    # 保存训练结果
    save_training_results(
        best_model_state,
        train_losses,
        val_losses,
        {
            'best_model': best_test_metrics,
            'last_model': last_test_metrics
        },
        args
    )
    
    print("\n训练完成！")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print("\n最佳模型测试指标:")
    for snr_group, metrics in best_test_metrics.items():
        print(f"\n{snr_group}:")
        print(f"PSDS: {metrics['psds_score']:.4f}")
        print(f"Macro F1: {metrics['segment_based_f1_macro']:.4f}")
        print(f"Event F1: {metrics['event_based_f1']:.4f}")
    
    print("\n最后一轮模型测试指标:")
    for snr_group, metrics in last_test_metrics.items():
        print(f"\n{snr_group}:")
        print(f"PSDS: {metrics['psds_score']:.4f}")
        print(f"Macro F1: {metrics['segment_based_f1_macro']:.4f}")
        print(f"Event F1: {metrics['event_based_f1']:.4f}")


def main():
    """主函数"""
    args = parse_args()
    train_and_evaluate(args)


if __name__ == "__main__":
    main()