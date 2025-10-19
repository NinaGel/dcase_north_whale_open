"""
eval_batch_models.py - 批量评估多个seed的模型并生成MD报告
Batch Model Evaluation Script - Evaluate multiple seeds and generate MD report

支持两种数据集 / Supported Datasets:
- whale: 鲸鱼声音事件检测数据集（原始）/ Whale sound event detection
- dcase: DCASE2020声音事件检测数据集 / DCASE2020 sound event detection

支持模型类型 / Supported Model Types:
- daap: DAAPNet模型
- faf: FAF-Filt模型
- crnn: CRNN基线模型
- conformer: Conformer模型
- cnn_transformer: CNN-Transformer模型
- panns: PANNs模型

用法 / Usage:
    # Whale数据集评估 / Whale dataset evaluation
    python eval_batch_models.py --seeds 63 64 65 66 67 --model faf
    python eval_batch_models.py --seeds 63 64 65 --model daap faf

    # DCASE数据集模型评估 / DCASE dataset evaluation
    python eval_batch_models.py --dataset dcase --seeds 42 43 44 45 46 --model daap
    python eval_batch_models.py --dataset dcase --seeds 42 43 44 45 46 --model faf --output dcase_faf_eval
"""

import torch
from pathlib import Path
import sys
import numpy as np
import argparse
from tqdm import tqdm
from datetime import datetime
import io


# ============================================================================
# Whale数据集相关函数
# ============================================================================

def load_whale_model(model_path, model_type, device='cuda'):
    """加载whale数据集的模型"""
    import config as cfg
    from experiments.model_comparison import initialize_model

    model = initialize_model(model_type)
    state_dict = torch.load(model_path, map_location=device)

    filtered_state_dict = {}
    for k, v in state_dict.items():
        if 'total_ops' not in k and 'total_params' not in k:
            if k.startswith('module.'):
                k = k[7:]
            filtered_state_dict[k] = v

    model.load_state_dict(filtered_state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model


def evaluate_whale_model(model, test_loaders, txt_folder_path, verbose=False):
    """评估whale数据集模型 / Evaluate whale dataset model

    Args:
        model: 模型实例 / model instance
        test_loaders: 测试数据加载器字典 / test data loaders dict
        txt_folder_path: 标注文件路径 / annotation file path
        verbose: 是否显示详细进度 / show detailed progress
    """
    import config as cfg
    from evaluation_metrics import WhaleEventDetectionEvaluator, load_ground_truth

    evaluator = WhaleEventDetectionEvaluator()
    snr_metrics = {}
    model.eval()
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    for snr_group, loader in test_loaders.items():
        ground_truth_df, ground_truth_dict, audio_durations = load_ground_truth(
            Path(txt_folder_path) / snr_group / 'txt'
        )

        all_preds = []
        all_filenames = []

        with torch.cuda.amp.autocast(enabled=cfg.TRAIN_CONFIG['mixed_precision']), torch.no_grad():
            for batch in tqdm(loader, desc=f"评估 {snr_group}", disable=not verbose, leave=False):
                torch.cuda.empty_cache()

                audio, label, filenames = batch
                audio = audio.to(device=device, dtype=model_dtype)
                predictions = model(audio)

                batch_scores = predictions.cpu().detach()
                all_preds.append(batch_scores)
                modified_filenames = [f'soundscape_whale_test_{snr_group}_{Path(f).stem}' for f in filenames]
                all_filenames.extend(modified_filenames)

        all_preds = torch.cat(all_preds, dim=0)

        # 临时禁用打印
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        metrics = evaluator.compute_all_metrics(
            all_preds,
            all_filenames,
            ground_truth_dict,
            ground_truth_df,
            audio_durations
        )

        sys.stdout = old_stdout
        snr_metrics[snr_group] = metrics

    return snr_metrics


# ============================================================================
# DCASE数据集相关函数
# ============================================================================

def load_dcase_model(model_path, model_type, device='cuda'):
    """加载DCASE数据集的模型 / Load DCASE dataset model"""
    import config_dcase as cfg

    if model_type == 'beats':
        from Model.BEATsBaseline import BEATsBaseline
        model = BEATsBaseline(num_classes=cfg.DCASE_MODEL_CONFIG['num_classes'])
    elif model_type in ['daap', 'ddsa', 'multiscale']:
        from Model.DCASE_Model_Attention_MultiScale import DCASE_Model_Attention_MultiScale
        model = DCASE_Model_Attention_MultiScale()
    elif model_type == 'faf':
        from Model.FAF_Filt import FAF_Filt_Model
        model = FAF_Filt_Model(
            num_classes=cfg.DCASE_MODEL_CONFIG['num_classes'],
            input_freq_bins=cfg.DCASE_AUDIO_CONFIG['freq'],
            conv_channels=[64, 128, 256, 256, 256],
            gru_hidden=256,
            gru_layers=2,
            use_projection=True,
            projection_method='conv1d',
            projection_target=128
        )
    elif model_type == 'conformer':
        sys.path.insert(0, str(Path(__file__).parent / 'experiments_dcase'))
        from model_comparison_dcase import Conformer
        model = Conformer(cfg.DCASE_MODEL_CONFIG['num_classes'])
    elif model_type == 'conformer_opt':
        from Model.Conformer_DCASE_Optimized import ConformerDCASE_Optimized
        model = ConformerDCASE_Optimized(
            num_classes=cfg.DCASE_MODEL_CONFIG['num_classes'],
            input_freq=cfg.DCASE_AUDIO_CONFIG['freq'],
            input_frame=cfg.DCASE_AUDIO_CONFIG['frame']
        )
    elif model_type == 'crnn':
        sys.path.insert(0, str(Path(__file__).parent / 'experiments_dcase'))
        from model_comparison_dcase import CRNN
        model = CRNN(cfg.DCASE_MODEL_CONFIG['num_classes'])
    else:
        raise ValueError(f"未知的DCASE模型类型: {model_type}")

    # 加载权重
    state_dict = torch.load(model_path, map_location=device, weights_only=False)

    filtered_state_dict = {}
    for k, v in state_dict.items():
        if 'total_ops' not in k and 'total_params' not in k:
            if k.startswith('module.'):
                k = k[7:]
            filtered_state_dict[k] = v

    model.load_state_dict(filtered_state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def evaluate_dcase_model(model, test_loaders, verbose=False):
    """评估DCASE数据集模型 / Evaluate DCASE dataset model

    Args:
        model: 模型实例 / model instance
        test_loaders: 测试数据加载器字典 / test data loaders dict
        verbose: 是否显示详细进度 / show detailed progress
    """
    import config_dcase as cfg
    from evaluation_metrics_dcase import DCASEEventDetectionEvaluator, load_ground_truth_dcase

    evaluator = DCASEEventDetectionEvaluator()
    snr_metrics = {}
    model.eval()
    device = next(model.parameters()).device

    base_path = cfg.DCASE_PATH_CONFIG['snr_data_path']

    for snr_group, loader in test_loaders.items():
        snr_suffix = snr_group.replace('snr_', '')
        metadata_file = base_path / snr_group / 'test' / f'test_{snr_suffix}.tsv'

        if not metadata_file.exists():
            print(f"警告: 找不到元数据文件 {metadata_file}")
            continue

        ground_truth_df, ground_truth_dict, audio_durations = load_ground_truth_dcase(metadata_file)

        all_preds = []
        all_filenames = []

        with torch.cuda.amp.autocast(enabled=cfg.DCASE_TRAIN_CONFIG.get('mixed_precision', True)), torch.no_grad():
            for batch in tqdm(loader, desc=f"评估 {snr_group}", disable=not verbose, leave=False):
                torch.cuda.empty_cache()

                features, label, filenames = batch
                features = features.to(device=device, dtype=torch.float32)
                predictions = model(features)

                batch_scores = predictions.cpu().detach()
                all_preds.append(batch_scores)
                all_filenames.extend(filenames)

        all_preds = torch.cat(all_preds, dim=0)

        # 临时禁用打印
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        metrics = evaluator.compute_all_metrics(
            all_preds,
            all_filenames,
            ground_truth_dict,
            ground_truth_df,
            audio_durations
        )

        sys.stdout = old_stdout
        snr_metrics[snr_group] = metrics

    return snr_metrics


# ============================================================================
# 通用辅助函数
# ============================================================================

def calculate_average_metrics(seed_results, dataset='whale'):
    """计算所有seed的平均指标"""
    if dataset == 'dcase':
        snr_groups = ['snr_high', 'snr_medium', 'snr_low']
        metrics_keys = ['psds1_score', 'psds2_score', 'macro_pauc', 'optimal_macro_f1',
                        'segment_based_f1_micro', 'segment_based_f1_macro', 'event_based_f1']
    else:
        snr_groups = ['snr_high', 'snr_medium', 'snr_low', 'snr_very_low']
        metrics_keys = ['psds_score', 'macro_pauc', 'optimal_macro_f1',
                        'segment_based_f1_micro', 'segment_based_er_micro',
                        'segment_based_f1_macro', 'event_based_f1']

    # 获取所有模型名
    all_models = set()
    for seed_data in seed_results.values():
        all_models.update(seed_data.keys())

    avg_results = {}
    for model_name in all_models:
        avg_results[model_name] = {}
        for snr_group in snr_groups:
            avg_results[model_name][snr_group] = {}
            for metric in metrics_keys:
                values = []
                for seed, seed_data in seed_results.items():
                    if model_name in seed_data and snr_group in seed_data[model_name]:
                        val = seed_data[model_name][snr_group].get(metric, None)
                        if val is not None and not np.isnan(val):
                            values.append(val)
                if values:
                    avg_results[model_name][snr_group][metric] = np.mean(values)
                    avg_results[model_name][snr_group][f'{metric}_std'] = np.std(values)
                else:
                    avg_results[model_name][snr_group][metric] = float('nan')
                    avg_results[model_name][snr_group][f'{metric}_std'] = float('nan')

    return avg_results


def format_seed_table(results_dict, snr_group, dataset='whale'):
    """格式化单个seed的指标表格"""
    if dataset == 'dcase':
        col_names = ['PSDS1', 'PSDS2', 'pAUC', 'Opt-F1', 'Seg-F1', 'MacroF1', 'Event-F1']
        metrics_map = {
            'PSDS1': 'psds1_score',
            'PSDS2': 'psds2_score',
            'pAUC': 'macro_pauc',
            'Opt-F1': 'optimal_macro_f1',
            'Seg-F1': 'segment_based_f1_micro',
            'MacroF1': 'segment_based_f1_macro',
            'Event-F1': 'event_based_f1'
        }
    else:
        col_names = ['PSDS', 'pAUC', 'Opt-F1', 'Seg-F1', 'Seg-ER', 'MacroF1', 'Event-F1']
        metrics_map = {
            'PSDS': 'psds_score',
            'pAUC': 'macro_pauc',
            'Opt-F1': 'optimal_macro_f1',
            'Seg-F1': 'segment_based_f1_micro',
            'Seg-ER': 'segment_based_er_micro',
            'MacroF1': 'segment_based_f1_macro',
            'Event-F1': 'event_based_f1'
        }

    # 表头
    header = f"| {'模型':<10} |"
    for name in col_names:
        header += f" {name:^8} |"

    separator = "|" + "-" * 12 + "|" + (("-" * 10 + "|") * len(col_names))

    lines = [header, separator]

    for model_name, snr_data in results_dict.items():
        if snr_group in snr_data:
            metrics = snr_data[snr_group]
            row = f"| {model_name:<10} |"
            for col_name in col_names:
                metric_key = metrics_map[col_name]
                val = metrics.get(metric_key, float('nan'))
                if not np.isnan(val):
                    row += f" {val:^8.4f} |"
                else:
                    row += f" {'N/A':^8} |"
            lines.append(row)

    return '\n'.join(lines)


def format_average_table(avg_results, snr_group, dataset='whale'):
    """格式化平均指标表格（含标准差）"""
    if dataset == 'dcase':
        col_names = ['PSDS1', 'PSDS2', 'pAUC', 'Opt-F1', 'Seg-F1', 'MacroF1', 'Event-F1']
        metrics_map = {
            'PSDS1': 'psds1_score',
            'PSDS2': 'psds2_score',
            'pAUC': 'macro_pauc',
            'Opt-F1': 'optimal_macro_f1',
            'Seg-F1': 'segment_based_f1_micro',
            'MacroF1': 'segment_based_f1_macro',
            'Event-F1': 'event_based_f1'
        }
    else:
        col_names = ['PSDS', 'pAUC', 'Opt-F1', 'Seg-F1', 'Seg-ER', 'MacroF1', 'Event-F1']
        metrics_map = {
            'PSDS': 'psds_score',
            'pAUC': 'macro_pauc',
            'Opt-F1': 'optimal_macro_f1',
            'Seg-F1': 'segment_based_f1_micro',
            'Seg-ER': 'segment_based_er_micro',
            'MacroF1': 'segment_based_f1_macro',
            'Event-F1': 'event_based_f1'
        }

    # 表头
    header = f"| {'模型':<10} |"
    for name in col_names:
        header += f" {name:^15} |"

    separator = "|" + "-" * 12 + "|" + (("-" * 17 + "|") * len(col_names))

    lines = [header, separator]

    for model_name, snr_data in avg_results.items():
        if snr_group in snr_data:
            metrics = snr_data[snr_group]
            row = f"| {model_name:<10} |"
            for col_name in col_names:
                metric_key = metrics_map[col_name]
                mean_val = metrics.get(metric_key, float('nan'))
                std_val = metrics.get(f'{metric_key}_std', float('nan'))
                if not np.isnan(mean_val):
                    row += f" {mean_val:.4f}±{std_val:.4f} |"
                else:
                    row += f" {'N/A':^15} |"
            lines.append(row)

    return '\n'.join(lines)


def generate_model_file_report(all_results, model_file, seeds, models, output_path, dataset='whale'):
    """为单个模型文件生成MD报告"""
    if dataset == 'dcase':
        snr_groups = ['snr_high', 'snr_medium', 'snr_low']
        snr_display = {
            'snr_high': 'SNR High (5-10 dB)',
            'snr_medium': 'SNR Medium (0-5 dB)',
            'snr_low': 'SNR Low (-5-0 dB)'
        }
    else:
        snr_groups = ['snr_high', 'snr_medium', 'snr_low', 'snr_very_low']
        snr_display = {
            'snr_high': 'SNR High (5-10 dB)',
            'snr_medium': 'SNR Medium (0-5 dB)',
            'snr_low': 'SNR Low (-5-0 dB)',
            'snr_very_low': 'SNR Very Low (-10--5 dB)'
        }

    lines = []
    file_display = 'Best Model' if 'best' in model_file else 'Last Epoch Model'
    lines.append(f"# {file_display} 评估报告 ({dataset.upper()} 数据集)")
    lines.append(f"\n**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\n**评估模型**: {', '.join(models)}")
    lines.append(f"\n**评估Seeds**: {', '.join(map(str, seeds))}")
    lines.append(f"\n**模型文件**: {model_file}")
    lines.append(f"\n**数据集**: {dataset}")
    lines.append("\n---\n")

    # 收集该 model_file 下所有 seed 的结果
    file_results = {}
    for seed in seeds:
        if seed in all_results and model_file in all_results[seed]:
            file_results[seed] = all_results[seed][model_file]

    if not file_results:
        lines.append("\n*无可用数据*\n")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        return

    # 1. 平均结果汇总
    lines.append("## 1. 平均结果汇总 (Mean ± Std)\n")

    avg_results = calculate_average_metrics(file_results, dataset)

    for snr_group in snr_groups:
        if snr_group in snr_display:
            lines.append(f"\n### {snr_display[snr_group]}\n")
            lines.append(format_average_table(avg_results, snr_group, dataset))
            lines.append("")

    # 2. 各 Seed 详细结果
    lines.append("\n---\n")
    lines.append("## 2. 各 Seed 详细结果\n")

    for seed in seeds:
        lines.append(f"\n### Seed {seed}\n")

        if seed not in file_results or not file_results[seed]:
            lines.append("*无数据*\n")
            continue

        results = file_results[seed]

        for snr_group in snr_groups:
            if snr_group in snr_display:
                lines.append(f"\n**{snr_display[snr_group]}**\n")
                lines.append(format_seed_table(results, snr_group, dataset))
                lines.append("")

    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"报告已保存至: {output_path}")


def find_dcase_model_path(base_dir, seed, model_name, model_file, experiment_name=None):
    """查找DCASE模型路径 / Find DCASE model path

    DCASE模型目录结构 / DCASE model directory structure:
    - BEATs: experiments_dcase/DCASE_Results/BEATs_Baseline_seed{seed}/best_model.pth
    - 其他 / Others: experiments_dcase/DCASE_Results/{experiment}/seed_{seed}/models/{model}/best_model.pth

    Args:
        base_dir: 基础目录 / base directory
        seed: 随机种子 / random seed
        model_name: 模型类型名称 / model type name
        model_file: 模型文件名 / model file name (best_model.pth)
        experiment_name: 实验名称（用于直接路径查找）/ experiment name for direct path lookup
    """
    base_dir = Path(base_dir)

    # BEATs模型的特殊路径 / Special path for BEATs model
    if model_name == 'beats':
        beats_path = base_dir / f"BEATs_Baseline_seed{seed}" / model_file
        if beats_path.exists():
            return beats_path

    # 直接实验名称路径 / Direct experiment name path
    if experiment_name:
        direct_path = base_dir / experiment_name / model_file
        if direct_path.exists():
            return direct_path

        seed_path = base_dir / f"{experiment_name}_seed{seed}" / model_file
        if seed_path.exists():
            return seed_path

    # 通用路径模式 / Common path patterns
    possible_paths = [
        base_dir / f"seed_{seed}" / 'models' / model_name / model_file,
        base_dir / f"{model_name}_seed{seed}" / model_file,
        base_dir / model_name / f"seed_{seed}" / model_file,
    ]

    for path in possible_paths:
        if path.exists():
            return path

    return None


def main():
    parser = argparse.ArgumentParser(description="批量评估多个seed的模型")
    parser.add_argument('--dataset', type=str, default='whale',
                        choices=['whale', 'dcase'],
                        help='数据集类型 (whale 或 dcase)')
    parser.add_argument('--seeds', type=int, nargs='+', required=True,
                        help='随机种子列表 (如: 63 64 65 66 67 或 42 43 44 45 46)')
    parser.add_argument('--model', type=str, nargs='+', required=True,
                        help='模型类型')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批次大小')
    parser.add_argument('--output', type=str, default=None,
                        help='输出文件前缀')
    parser.add_argument('--base_dir', type=str, default=None,
                        help='模型基础目录 (默认根据数据集自动选择)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.dataset == 'dcase':
        # DCASE数据集评估 / DCASE dataset evaluation
        import config_dcase as cfg
        from Data.dcase_dataset import load_test_data, load_beats_test_data

        base_dir = Path(args.base_dir) if args.base_dir else Path("experiments_dcase/DCASE_Results")
        model_files = ['best_model.pth']  # DCASE通常只保存best_model

        # 检查是否需要评估BEATs模型 / Check if BEATs model evaluation is needed
        has_beats = 'beats' in args.model
        has_other = any(m != 'beats' for m in args.model)

        # 检查可用的seed / Check available seeds
        available_seeds = []
        for seed in args.seeds:
            # 检查BEATs路径 / Check BEATs path
            if 'beats' in args.model:
                beats_path = base_dir / f"BEATs_Baseline_seed{seed}"
                if beats_path.exists():
                    available_seeds.append(seed)
                    continue

            # 检查其他路径 / Check other paths
            seed_dir = base_dir / f"seed_{seed}"
            if seed_dir.exists():
                available_seeds.append(seed)

        if not available_seeds:
            print("错误: 没有可用的seed目录")
            print(f"检查的路径: {base_dir}")
            sys.exit(1)

        print(f"可用的seeds: {available_seeds}")

        # 加载测试数据 / Load test data
        test_loaders = None
        beats_test_loaders = None

        if has_other:
            print(f"加载 DCASE ACT 测试数据 (batch_size={args.batch_size})...")
            test_loaders = load_test_data(args.batch_size)

        if has_beats:
            print(f"加载 DCASE BEATs 测试数据 (batch_size={args.batch_size})...")
            try:
                beats_test_loaders = load_beats_test_data(args.batch_size)
            except Exception as e:
                print(f"错误: 无法加载BEATs测试数据: {e}")
                print("请先运行: python tools/extract_beats_features_v3.py --dataset dcase --all")
                sys.exit(1)

        # 存储所有结果
        all_results = {}

        total_evals = len(available_seeds) * len(model_files) * len(args.model)
        eval_count = 0

        for seed in available_seeds:
            all_results[seed] = {}

            for model_file in model_files:
                all_results[seed][model_file] = {}

                for model_name in args.model:
                    eval_count += 1

                    # 查找模型路径
                    model_path = find_dcase_model_path(base_dir, seed, model_name, model_file)

                    if model_path is None:
                        print(f"[{eval_count}/{total_evals}] 跳过: 找不到 seed={seed}, model={model_name}")
                        continue

                    print(f"[{eval_count}/{total_evals}] 评估: seed={seed}, model={model_name}")
                    print(f"  路径: {model_path}")

                    try:
                        model = load_dcase_model(str(model_path), model_name, device)

                        # 根据模型类型选择测试数据 / Select test data by model type
                        if model_name == 'beats':
                            current_loaders = beats_test_loaders
                            print("  使用 BEATs 特征进行评估")
                        else:
                            current_loaders = test_loaders
                            print("  使用 ACT 特征进行评估")

                        results = evaluate_dcase_model(model, current_loaders, verbose=False)
                        all_results[seed][model_file][model_name] = results

                        del model
                        torch.cuda.empty_cache()

                    except Exception as e:
                        print(f"  错误: {e}")
                        import traceback
                        traceback.print_exc()

        # 生成报告
        output_prefix = args.output if args.output else f"dcase_eval_{'_'.join(args.model)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        for model_file in model_files:
            file_suffix = 'best' if 'best' in model_file else 'last'
            output_path = f"{output_prefix}_{file_suffix}.md"
            generate_model_file_report(all_results, model_file, available_seeds, args.model, output_path, dataset='dcase')

        print(f"\n评估完成! 生成报告: {output_prefix}_best.md")

    else:
        # Whale数据集评估 / Whale dataset evaluation
        import config as cfg
        from Data.audio_dataset import load_test_data, load_beats_snr_data

        base_dir = Path(args.base_dir) if args.base_dir else Path('experiments/Snr_Results/model_comparison')
        model_files = ['best_epoch.pth', 'last_epoch.pth']

        # 检查seed目录 / Check seed directories
        available_seeds = []
        for seed in args.seeds:
            seed_dir = base_dir / f"seed_{seed}"
            if seed_dir.exists():
                available_seeds.append(seed)
            else:
                print(f"警告: 找不到seed目录 {seed_dir}")

        if not available_seeds:
            print("错误: 没有可用的seed目录")
            sys.exit(1)

        # 检查模型类型 / Check model types
        has_beats = 'beats' in args.model
        has_other = any(m != 'beats' for m in args.model)

        txt_folder_path = cfg.PATH_CONFIG['test_path']

        # 加载测试数据 / Load test data
        test_loaders = None
        beats_test_loaders = None

        if has_other:
            print(f"加载 ACT 测试数据 (batch_size={args.batch_size})...")
            test_loaders = load_test_data(args.batch_size)

        if has_beats:
            print(f"加载 BEATs 测试数据 (batch_size={args.batch_size})...")
            beats_test_loaders = load_beats_snr_data('test', args.batch_size)

        # 存储所有结果
        all_results = {}

        total_evals = len(available_seeds) * len(model_files) * len(args.model)
        eval_count = 0

        for seed in available_seeds:
            all_results[seed] = {}
            seed_dir = base_dir / f"seed_{seed}"

            for model_file in model_files:
                all_results[seed][model_file] = {}

                for model_name in args.model:
                    eval_count += 1
                    model_path = seed_dir / 'models' / model_name / model_file

                    if not model_path.exists():
                        print(f"[{eval_count}/{total_evals}] 跳过: {model_path} 不存在")
                        continue

                    print(f"[{eval_count}/{total_evals}] 评估: seed={seed}, model={model_name}, file={model_file}")

                    try:
                        model = load_whale_model(str(model_path), model_name, device)

                        # 根据模型类型选择测试数据 / Select test data by model type
                        if model_name == 'beats':
                            current_loaders = beats_test_loaders
                            print("  使用 BEATs 特征进行评估")
                        else:
                            current_loaders = test_loaders

                        results = evaluate_whale_model(
                            model, current_loaders, txt_folder_path, verbose=False
                        )
                        all_results[seed][model_file][model_name] = results

                        del model
                        torch.cuda.empty_cache()

                    except Exception as e:
                        print(f"  错误: {e}")
                        import traceback
                        traceback.print_exc()

        # 生成报告
        output_prefix = args.output if args.output else f"eval_{'_'.join(args.model)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        for model_file in model_files:
            file_suffix = 'best' if 'best' in model_file else 'last'
            output_path = f"{output_prefix}_{file_suffix}.md"
            generate_model_file_report(all_results, model_file, available_seeds, args.model, output_path, dataset='whale')

        print(f"\n评估完成! 生成报告:")
        print(f"  - {output_prefix}_best.md")
        print(f"  - {output_prefix}_last.md")


if __name__ == "__main__":
    main()
