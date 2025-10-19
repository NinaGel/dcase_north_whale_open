import torch
import torch.nn as nn
from pathlib import Path
import sys
import time
from tqdm.auto import tqdm
import torch.cuda as cuda
sys.path.append('..')

from Model.Whale_Model_Attention_Ldsa import Whale_Model_Attention_Original
from Model.Whale_Model_Attention_Hybrid import Whale_Model_Attention_Optimized
from Data.audio_dataset import load_fold_data
from Train.train_utils import (
    train_model, validate_model, plot_losses,
    WarmupCosineAnnealingLR
)
import config as cfg
from eval_example import evaluate_model, load_ground_truth


def setup_gpu():
    """配置GPU训练环境"""
    if not torch.cuda.is_available():
        print("警告: 未检测到可用的CUDA设备，将使用CPU训练")
        return torch.device('cpu')
    
    # 获取GPU信息
    device = torch.device('cuda')
    gpu_count = torch.cuda.device_count()
    current_gpu = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(current_gpu)
    
    print(f"\nGPU信息:")
    print(f"  设备名称: {gpu_name}")
    print(f"  设备数量: {gpu_count}")
    print(f"  当前设备ID: {current_gpu}")
    print(f"  显存总量: {torch.cuda.get_device_properties(current_gpu).total_memory / 1024**3:.1f}GB")
    print(f"  CUDA版本: {torch.version.cuda}")
    
    # 设置GPU内存策略
    torch.backends.cudnn.benchmark = True  # 优化性能
    torch.cuda.empty_cache()  # 清空缓存
    
    return device


def print_stage_header(stage_name: str, model_name: str):
    """打印阶段标题
    
    Args:
        stage_name: 阶段名称
        model_name: 模型名称
    """
    print(f"\n{'='*20} {stage_name} - {model_name} {'='*20}")


def print_metrics(metrics: dict, prefix: str = ""):
    """打印评估指标
    
    Args:
        metrics: 评估指标字典
        prefix: 指标前缀
    """
    for metric_type in ['raw', 'processed']:
        print(f"\n{prefix}{metric_type.capitalize()} Metrics:")
        print(f"  准确率: {metrics[metric_type]['accuracy']:.4f}")
        print(f"  精确率: {metrics[metric_type]['precision']:.4f}")
        print(f"  召回率: {metrics[metric_type]['recall']:.4f}")
        print(f"  F1分数: {metrics[metric_type]['f1']:.4f}")


def print_gpu_memory_status():
    """打印GPU内存状态"""
    if torch.cuda.is_available():
        current_gpu = torch.cuda.current_device()
        memory_allocated = torch.cuda.memory_allocated(current_gpu) / 1024**2
        memory_reserved = torch.cuda.memory_reserved(current_gpu) / 1024**2
        print(f"\nGPU内存状态:")
        print(f"  已分配: {memory_allocated:.1f}MB")
        print(f"  已预留: {memory_reserved:.1f}MB")


def train_model_with_monitoring(model, train_loader, val_loader, loss_fn, optimizer, 
                              device, epochs, model_name, fold):
    """带GPU监控的模型训练函数
    
    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        loss_fn: 损失函数
        optimizer: 优化器
        device: 训练设备
        epochs: 训练轮数
        model_name: 模型名称
        fold: 当前折序号
    
    Returns:
        tuple: 训练结果
    """
    print(f"\n开始训练 {model_name}...")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 显示初始GPU内存状态
    print("\n初始GPU内存状态:")
    print_gpu_memory_status()
    
    # 确保模型参数为float32类型
    model = model.float()
    
    # 加载一个批次数据来估算内存使用
    print("\n加载一个批次后的GPU内存状态:")
    sample_batch = next(iter(train_loader))
    sample_x = sample_batch[0].to(device, dtype=torch.float32)  # 确保为float32
    sample_y = sample_batch[1].to(device, dtype=torch.float32)  # 确保为float32
    print(f"批次数据形状: {sample_x.shape}")
    print(f"数据类型: {sample_x.dtype}")
    print_gpu_memory_status()
    
    # 进行一次前向传播来估算完整的内存使用
    print("\n前向传播后的GPU内存状态:")
    with torch.no_grad():
        _ = model(sample_x)
    print_gpu_memory_status()
    
    # 清理示例数据
    del sample_x, sample_y
    torch.cuda.empty_cache()
    
    train_start = time.time()
    
    # 使用train_utils中的训练函数
    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        model_name=model_name,
        fold=fold
    )
    
    train_time = time.time() - train_start
    print(f"\n{model_name}训练完成! 耗时: {train_time/60:.2f}分钟")
    
    # 显示训练后的GPU内存状态
    print("\n训练后GPU内存状态:")
    print_gpu_memory_status()
    
    # 绘制损失曲线
    plot_losses(
        train_losses=results[0],
        val_losses=results[1],
        epochs=epochs,
        save_path=f"Result/ldsa_comparison/{model_name}"
    )
    
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("\n清理后GPU内存状态:")
        print_gpu_memory_status()
    
    return results


def train_and_evaluate_models():
    """训练和评估两个LDSA模型版本"""
    
    # 设置GPU
    device = setup_gpu()
    
    # 禁用自动混合精度
    torch.cuda.amp.autocast(enabled=False)
    
    # 加载Fold 1数据
    print_stage_header("数据加载", "Fold 1")
    train_loader, val_loader = load_fold_data(fold=1)
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    print(f"批次大小: {train_loader.batch_size}")
    
    # 创建保存目录
    save_dir = Path("Result/ldsa_comparison")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练参数
    loss_fn = nn.BCEWithLogitsLoss()
    
    # 训练和评估优化LDSA模型
    print_stage_header("模型训练", "优化LDSA")
    optimized_model = Whale_Model_Attention_Optimized(debug_mode=False).to(device).float()  # 确保为float32
    optimizer_optimized = torch.optim.Adam(
        optimized_model.parameters(),
        **cfg.OPTIMIZER_CONFIG
    )
    
    optimized_results = train_model_with_monitoring(
        model=optimized_model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer_optimized,
        device=device,
        epochs=cfg.TRAIN_CONFIG['epochs'],
        model_name="optimized_ldsa",
        fold=1
    )
    
    # 保存优化模型结果
    torch.save(optimized_results[4], save_dir / "optimized_ldsa_best.pth")
    torch.save(optimized_results[5], save_dir / "optimized_ldsa_last.pth")
    
    # 训练和评估原始LDSA模型
    print_stage_header("模型训练", "原始LDSA")
    original_model = Whale_Model_Attention_Original().to(device).float()  # 确保为float32
    optimizer_original = torch.optim.Adam(
        original_model.parameters(),
        **cfg.OPTIMIZER_CONFIG
    )
    
    original_results = train_model_with_monitoring(
        model=original_model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer_original,
        device=device,
        epochs=cfg.TRAIN_CONFIG['epochs'],
        model_name="original_ldsa",
        fold=1
    )
    
    # 保存原始模型结果
    torch.save(original_results[4], save_dir / "original_ldsa_best.pth")
    torch.save(original_results[5], save_dir / "original_ldsa_last.pth")
    
    # 评估模型
    print_stage_header("模型评估", "两个版本")
    txt_folder_path = 'Data/scaper_audio/scaper_k_fold/test/txt/'
    ground_truth_df, ground_truth_dict, audio_durations = load_ground_truth(txt_folder_path)
    
    # 评估优化模型
    print("\n评估优化LDSA模型...")
    optimized_model.load_state_dict(optimized_results[4])
    optimized_metrics = evaluate_model(
        optimized_model, 
        val_loader, 
        txt_folder_path
    )
    print_metrics(optimized_metrics, "优化LDSA - ")
    
    # 评估原始模型
    print("\n评估原始LDSA模型...")
    original_model.load_state_dict(original_results[4])
    original_metrics = evaluate_model(
        original_model, 
        val_loader, 
        txt_folder_path
    )
    print_metrics(original_metrics, "原始LDSA - ")
    
    # 生成评估报告
    report = generate_comparison_report(
        original_metrics,
        optimized_metrics,
        original_results,
        optimized_results
    )
    
    # 保存报告
    report_path = save_dir / "comparison_report.txt"
    with open(report_path, "w", encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n评估完成！详细报告已保存到: {report_path}")
    print("\n性能对比摘要:")
    print(report.split("3. 性能提升")[1])
    
    # 最终的GPU内存状态
    print_gpu_memory_status()


def generate_comparison_report(original_metrics, optimized_metrics, 
                             original_results, optimized_results):
    """生成比较报告
    
    Args:
        original_metrics: 原始模型评估指标
        optimized_metrics: 优化模型评估指标
        original_results: 原始模型训练结果
        optimized_results: 优化模型训练结果
    
    Returns:
        str: 比较报告
    """
    report = """LDSA模型比较报告
====================

1. 训练性能
-----------
原始LDSA:
  - 最佳验证损失: {:.4f} (epoch {})
  - 最终训练损失: {:.4f}
  - 训练轮数: {}
  - 收敛轮数: {}

优化LDSA:
  - 最佳验证损失: {:.4f} (epoch {})
  - 最终训练损失: {:.4f}
  - 训练轮数: {}
  - 收敛轮数: {}

2. 评估指标
-----------
原始LDSA:
  原始分数:
    - 准确率: {:.4f}
    - 精确率: {:.4f}
    - 召回率: {:.4f}
    - F1分数: {:.4f}
  
  后处理分数:
    - 准确率: {:.4f}
    - 精确率: {:.4f}
    - 召回率: {:.4f}
    - F1分数: {:.4f}

优化LDSA:
  原始分数:
    - 准确率: {:.4f}
    - 精确率: {:.4f}
    - 召回率: {:.4f}
    - F1分数: {:.4f}
  
  后处理分数:
    - 准确率: {:.4f}
    - 精确率: {:.4f}
    - 召回率: {:.4f}
    - F1分数: {:.4f}

3. 性能提升
-----------
准确率提升: {:.2f}%
精确率提升: {:.2f}%
召回率提升: {:.2f}%
F1分数提升: {:.2f}%""".format(
        original_results[2], original_results[3], original_results[0][-1], 
        len(original_results[0]), original_results[3],
        optimized_results[2], optimized_results[3], optimized_results[0][-1], 
        len(optimized_results[0]), optimized_results[3],
        
        original_metrics['raw']['accuracy'],
        original_metrics['raw']['precision'],
        original_metrics['raw']['recall'],
        original_metrics['raw']['f1'],
        
        original_metrics['processed']['accuracy'],
        original_metrics['processed']['precision'],
        original_metrics['processed']['recall'],
        original_metrics['processed']['f1'],
        
        optimized_metrics['raw']['accuracy'],
        optimized_metrics['raw']['precision'],
        optimized_metrics['raw']['recall'],
        optimized_metrics['raw']['f1'],
        
        optimized_metrics['processed']['accuracy'],
        optimized_metrics['processed']['precision'],
        optimized_metrics['processed']['recall'],
        optimized_metrics['processed']['f1'],
        
        (optimized_metrics['processed']['accuracy'] - original_metrics['processed']['accuracy']) / original_metrics['processed']['accuracy'] * 100,
        (optimized_metrics['processed']['precision'] - original_metrics['processed']['precision']) / original_metrics['processed']['precision'] * 100,
        (optimized_metrics['processed']['recall'] - original_metrics['processed']['recall']) / original_metrics['processed']['recall'] * 100,
        (optimized_metrics['processed']['f1'] - original_metrics['processed']['f1']) / original_metrics['processed']['f1'] * 100
    )
    
    return report


if __name__ == "__main__":
    train_and_evaluate_models() 