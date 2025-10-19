"""
eval_single_model.py - 单独评估指定seed和模型的脚本

支持两种数据集:
- whale: 鲸鱼声音事件检测数据集（原始）
- dcase: DCASE2020声音事件检测数据集

用法:
    # 评估whale数据集模型
    python eval_single_model.py --seed 63 --model daap
    python eval_single_model.py --seed 64 --model faf
    python eval_single_model.py --seed 64 --model faf --model_file last_epoch.pth

    # 评估DCASE数据集模型
    python eval_single_model.py --dataset dcase --seed 63 --model daap
    python eval_single_model.py --dataset dcase --model_path experiments_dcase/DCASE_Results/FAF_Heavy_Proj_bs32/best_model.pth --model faf

    # 批量评估DCASE_Results下的多个模型
    python eval_single_model.py --dataset dcase --batch_eval
"""

import torch
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import argparse
import json
from tqdm import tqdm


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


def load_dcase_model(model_path, model_type, device='cuda'):
    """加载DCASE数据集的模型

    Args:
        model_path (str): 模型权重文件路径(.pth)
        model_type (str): 模型类型
        device (str): 使用的设备

    Returns:
        model: 加载了权重的模型
    """
    import config_dcase as cfg
    from Model.DCASE_Model_Attention_MultiScale import DCASE_Model_Attention_MultiScale
    from Model.FAF_Filt import FAF_Filt_Model

    # DCASE模型初始化
    if model_type in ['daap', 'ddsa', 'multiscale']:
        model = DCASE_Model_Attention_MultiScale()
    elif model_type == 'faf':
        # FAF模型支持多种配置，尝试从checkpoint推断配置
        # 默认使用较大的Heavy配置（与FAF_Heavy_Proj_bs32相匹配）
        # projection_target=128 -> 经过pool1后变为64 freq bins给第一个block
        # conv_channels有5个元素 = 4个blocks with pools
        model = FAF_Filt_Model(
            num_classes=cfg.DCASE_MODEL_CONFIG['num_classes'],
            input_freq_bins=cfg.DCASE_AUDIO_CONFIG['freq'],
            conv_channels=[64, 128, 256, 256, 256],  # Heavy配置: 5元素=4个blocks
            gru_hidden=256,
            gru_layers=2,
            use_projection=True,
            projection_method='conv1d',
            projection_target=128  # 128 -> pool -> 64 for first block
        )
    elif model_type == 'conformer':
        # 从model_comparison_dcase中动态导入
        sys.path.insert(0, str(Path(__file__).parent / 'experiments_dcase'))
        from model_comparison_dcase import Conformer
        model = Conformer(cfg.DCASE_MODEL_CONFIG['num_classes'])
    elif model_type == 'crnn':
        sys.path.insert(0, str(Path(__file__).parent / 'experiments_dcase'))
        from model_comparison_dcase import CRNN
        model = CRNN(cfg.DCASE_MODEL_CONFIG['num_classes'])
    elif model_type == 'panns':
        sys.path.insert(0, str(Path(__file__).parent / 'experiments_dcase'))
        from model_comparison_dcase import PANNs
        model = PANNs(cfg.DCASE_MODEL_CONFIG['num_classes'])
    elif model_type == 'cnn_transformer':
        sys.path.insert(0, str(Path(__file__).parent / 'experiments_dcase'))
        from model_comparison_dcase import CNN_Transformer
        model = CNN_Transformer(cfg.DCASE_MODEL_CONFIG['num_classes'])
    elif model_type == 'tdnn_lstm':
        sys.path.insert(0, str(Path(__file__).parent / 'experiments_dcase'))
        from model_comparison_dcase import TDNN_LSTM
        model = TDNN_LSTM(cfg.DCASE_MODEL_CONFIG['num_classes'])
    elif model_type == 'beats':
        from Model.BEATsBaseline import BEATsBaseline
        model = BEATsBaseline(num_classes=cfg.DCASE_MODEL_CONFIG['num_classes'])
    else:
        raise ValueError(f"未知的DCASE模型类型: {model_type}")

    # 加载权重
    state_dict = torch.load(model_path, map_location=device, weights_only=False)

    # 过滤掉FLOPs计算相关的键
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


def evaluate_whale_model(model, test_loaders, txt_folder_path, verbose=True):
    """评估whale数据集模型"""
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
            for batch in tqdm(loader, desc=f"评估 {snr_group}", disable=not verbose):
                torch.cuda.empty_cache()

                audio, label, filenames = batch
                audio = audio.to(device=device, dtype=model_dtype)
                predictions = model(audio)

                batch_scores = predictions.cpu().detach()
                all_preds.append(batch_scores)

                modified_filenames = [f'soundscape_whale_test_{snr_group}_{Path(f).stem}' for f in filenames]
                all_filenames.extend(modified_filenames)

        all_preds = torch.cat(all_preds, dim=0)

        metrics = evaluator.compute_all_metrics(
            all_preds,
            all_filenames,
            ground_truth_dict,
            ground_truth_df,
            audio_durations
        )

        snr_metrics[snr_group] = metrics

        if verbose:
            print(f"\n=== {snr_group} 评估结果 ===")
            print(f"  PSDS: {metrics['psds_score']:.4f}")
            print(f"  pAUC: {metrics['macro_pauc']:.4f}")
            print(f"  Opt-F1: {metrics['optimal_macro_f1']:.4f}")
            print(f"  Seg-F1 (micro): {metrics['segment_based_f1_micro']:.4f}")
            print(f"  Event-F1: {metrics['event_based_f1']:.4f}")

    return snr_metrics


def evaluate_dcase_model(model, test_loaders, verbose=True):
    """评估DCASE数据集模型

    Args:
        model: 待评估的模型
        test_loaders: 测试数据加载器字典
        verbose: 是否打印详细信息

    Returns:
        dict: SNR分组的评估结果
    """
    import config_dcase as cfg
    from evaluation_metrics_dcase import DCASEEventDetectionEvaluator, load_ground_truth_dcase

    evaluator = DCASEEventDetectionEvaluator()
    snr_metrics = {}
    model.eval()
    device = next(model.parameters()).device

    # DCASE数据路径
    base_path = cfg.DCASE_PATH_CONFIG['snr_data_path']

    for snr_group, loader in test_loaders.items():
        # 加载当前SNR组的标签数据
        snr_suffix = snr_group.replace('snr_', '')
        metadata_file = base_path / snr_group / 'test' / f'test_{snr_suffix}.tsv'

        if not metadata_file.exists():
            print(f"警告: 找不到元数据文件 {metadata_file}")
            continue

        ground_truth_df, ground_truth_dict, audio_durations = load_ground_truth_dcase(metadata_file)

        all_preds = []
        all_filenames = []

        with torch.cuda.amp.autocast(enabled=cfg.DCASE_TRAIN_CONFIG.get('mixed_precision', True)), torch.no_grad():
            for batch in tqdm(loader, desc=f"评估 {snr_group}", disable=not verbose):
                torch.cuda.empty_cache()

                features, label, filenames = batch
                features = features.to(device=device, dtype=torch.float32)
                predictions = model(features)

                batch_scores = predictions.cpu().detach()
                all_preds.append(batch_scores)
                all_filenames.extend(filenames)

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

        if verbose:
            print(f"\n=== {snr_group} 评估结果 ===")
            print(f"  PSDS1: {metrics['psds1_score']:.4f}")
            print(f"  PSDS2: {metrics['psds2_score']:.4f}")
            print(f"  pAUC: {metrics['macro_pauc']:.4f}")
            print(f"  Opt-F1: {metrics['optimal_macro_f1']:.4f}")
            print(f"  Seg-F1 (micro): {metrics['segment_based_f1_micro']:.4f}")
            print(f"  Event-F1: {metrics['event_based_f1']:.4f}")

    return snr_metrics


def print_summary(results, model_name, seed, model_file, dataset='whale'):
    """打印评估结果摘要"""
    print("\n" + "=" * 70)
    print(f"评估摘要: {model_name} (seed={seed}, {model_file}, dataset={dataset})")
    print("=" * 70)

    if dataset == 'dcase':
        metrics_to_print = [
            ('psds1_score', 'PSDS1'),
            ('psds2_score', 'PSDS2'),
            ('macro_pauc', 'pAUC'),
            ('optimal_macro_f1', 'Opt-F1'),
            ('segment_based_f1_micro', 'Seg-F1'),
            ('event_based_f1', 'Event-F1')
        ]
    else:
        metrics_to_print = [
            ('psds_score', 'PSDS'),
            ('macro_pauc', 'pAUC'),
            ('optimal_macro_f1', 'Opt-F1'),
            ('segment_based_f1_micro', 'Seg-F1'),
            ('segment_based_er_micro', 'Seg-ER'),
            ('event_based_f1', 'Event-F1')
        ]

    # 打印表头
    header = f"{'SNR组':<15}"
    for _, name in metrics_to_print:
        header += f"{name:<10}"
    print(header)
    print("-" * 70)

    # 打印每个SNR组的结果
    snr_groups = ['snr_high', 'snr_medium', 'snr_low', 'snr_very_low']
    for snr_group in snr_groups:
        if snr_group in results:
            row = f"{snr_group:<15}"
            for metric_key, _ in metrics_to_print:
                value = results[snr_group].get(metric_key, float('nan'))
                row += f"{value:<10.4f}"
            print(row)

    # 计算平均值
    print("-" * 70)
    avg_row = f"{'平均':<15}"
    for metric_key, _ in metrics_to_print:
        values = [results[snr][metric_key] for snr in snr_groups if snr in results and metric_key in results[snr]]
        avg_value = np.mean(values) if values else float('nan')
        avg_row += f"{avg_value:<10.4f}"
    print(avg_row)

    return {metric_key: np.mean([results[snr][metric_key] for snr in snr_groups
                                  if snr in results and metric_key in results[snr]])
            for metric_key, _ in metrics_to_print}


def save_results(results, output_path, model_name, seed, model_file, dataset='whale'):
    """保存评估结果到JSON文件"""
    output_data = {
        'model': model_name,
        'seed': seed,
        'model_file': model_file,
        'dataset': dataset,
        'results': {}
    }

    for snr_group, metrics in results.items():
        output_data['results'][snr_group] = {
            k: float(v) if isinstance(v, (int, float, np.floating)) else v
            for k, v in metrics.items()
            if k != 'class_wise_metrics'
        }
        if 'class_wise_metrics' in metrics:
            output_data['results'][snr_group]['class_wise_metrics'] = {}
            for class_name, class_metrics in metrics['class_wise_metrics'].items():
                output_data['results'][snr_group]['class_wise_metrics'][class_name] = {
                    k: float(v) if isinstance(v, (int, float, np.floating)) and not np.isnan(v) else None
                    for k, v in class_metrics.items()
                }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存至: {output_path}")


def find_dcase_models(base_dir):
    """查找DCASE_Results目录下的所有模型

    Returns:
        list: [(model_path, model_name, seed), ...]
    """
    base_dir = Path(base_dir)
    models = []

    # 查找所有best_model.pth文件
    for pth_file in base_dir.rglob('best_model.pth'):
        parent_name = pth_file.parent.name

        # 解析模型信息
        # 格式1: BEATs_Baseline_seed42/best_model.pth
        # 格式2: FAF_Heavy_Proj_bs32/best_model.pth
        # 格式3: ulf_comparison/seed_63/models/xxx_best.pth

        if 'seed' in parent_name.lower():
            # 从目录名提取seed
            import re
            match = re.search(r'seed[_]?(\d+)', parent_name, re.IGNORECASE)
            seed = int(match.group(1)) if match else None
            model_name = parent_name.split('_seed')[0] if '_seed' in parent_name else parent_name.split('seed')[0].rstrip('_')
        else:
            seed = None
            model_name = parent_name

        models.append({
            'path': pth_file,
            'name': model_name,
            'seed': seed,
            'dir_name': parent_name
        })

    # 也查找其他格式的best模型
    for pth_file in base_dir.rglob('*_best.pth'):
        if 'best_model' not in pth_file.name:
            parent = pth_file.parent

            # 从路径中提取seed
            seed = None
            for part in pth_file.parts:
                if part.startswith('seed_'):
                    try:
                        seed = int(part.split('_')[1])
                    except:
                        pass

            model_name = pth_file.stem.replace('_best', '')

            models.append({
                'path': pth_file,
                'name': model_name,
                'seed': seed,
                'dir_name': parent.name
            })

    return models


def infer_model_type(model_name):
    """从模型名称推断模型类型"""
    model_name_lower = model_name.lower()

    if 'beats' in model_name_lower:
        return 'beats'
    elif 'faf' in model_name_lower:
        return 'faf'
    elif 'conformer' in model_name_lower:
        return 'conformer'
    elif 'crnn' in model_name_lower:
        return 'crnn'
    elif 'panns' in model_name_lower:
        return 'panns'
    elif 'cnn_transformer' in model_name_lower:
        return 'cnn_transformer'
    elif 'tdnn' in model_name_lower:
        return 'tdnn_lstm'
    elif 'bce' in model_name_lower or 'ulf' in model_name_lower or 'focal' in model_name_lower:
        # ULF comparison实验使用的是DAAP模型
        return 'daap'
    else:
        # 默认使用DAAP模型
        return 'daap'


def main():
    parser = argparse.ArgumentParser(description="单独评估指定seed和模型")
    parser.add_argument('--dataset', type=str, default='whale',
                        choices=['whale', 'dcase'],
                        help='数据集类型 (whale 或 dcase)')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子 (如: 63, 64, 65...)')
    parser.add_argument('--model', type=str, nargs='+', default=['daap'],
                        help='模型类型')
    parser.add_argument('--model_file', type=str, default='best_epoch.pth',
                        help='模型文件名 (best_epoch.pth 或 last_epoch.pth)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='直接指定模型路径（覆盖seed和model参数）')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批次大小')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录')
    parser.add_argument('--quiet', action='store_true',
                        help='安静模式，减少输出')
    parser.add_argument('--batch_eval', action='store_true',
                        help='批量评估DCASE_Results下的所有模型')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.dataset == 'dcase':
        import config_dcase as cfg
        from Data.dcase_dataset import load_test_data, load_beats_test_data

        # 检查是否需要评估BEATs模型
        has_beats = 'beats' in args.model
        has_other = any(m != 'beats' for m in args.model)

        # 加载测试数据
        test_loaders = None
        beats_test_loaders = None

        if has_other or not has_beats:
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

        if args.batch_eval:
            # 批量评估模式
            base_dir = Path("experiments_dcase/DCASE_Results")
            models = find_dcase_models(base_dir)

            print(f"\n找到 {len(models)} 个模型:")
            for m in models:
                print(f"  - {m['name']} (seed={m['seed']}): {m['path']}")

            all_results = []

            for model_info in models:
                model_path = model_info['path']
                model_name = model_info['name']
                seed = model_info['seed']
                model_type = infer_model_type(model_name)

                print(f"\n{'='*70}")
                print(f"评估模型: {model_name} (type={model_type}, seed={seed})")
                print(f"模型路径: {model_path}")
                print(f"{'='*70}")

                try:
                    model = load_dcase_model(str(model_path), model_type, device)
                    print(f"模型已加载到 {device}")

                    # 根据模型类型选择测试数据
                    if model_type == 'beats':
                        current_loaders = beats_test_loaders
                        print("使用 BEATs 特征进行评估")
                    else:
                        current_loaders = test_loaders
                        print("使用 ACT 特征进行评估")

                    results = evaluate_dcase_model(model, current_loaders, verbose=not args.quiet)
                    avg_metrics = print_summary(results, model_name, seed, model_path.name, dataset='dcase')

                    # 保存单个模型结果
                    output_dir = Path(args.output_dir) if args.output_dir else model_path.parent / 'evaluation_results'
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_file = output_dir / f"{model_name}_eval.json"
                    save_results(results, output_file, model_name, seed, model_path.name, dataset='dcase')

                    all_results.append({
                        'model': model_name,
                        'seed': seed,
                        'path': str(model_path),
                        **avg_metrics
                    })

                    del model
                    torch.cuda.empty_cache()

                except Exception as e:
                    print(f"错误: 评估 {model_name} 失败: {e}")
                    import traceback
                    traceback.print_exc()

            # 保存汇总结果
            if all_results:
                summary_df = pd.DataFrame(all_results)
                summary_path = base_dir / 'evaluation_summary.csv'
                summary_df.to_csv(summary_path, index=False)
                print(f"\n汇总结果已保存至: {summary_path}")

                # 打印汇总表格
                print("\n" + "=" * 100)
                print("评估汇总")
                print("=" * 100)
                print(summary_df.to_string(index=False))

        else:
            # 单个模型评估模式
            if args.model_path:
                model_path = Path(args.model_path)
                model_name = args.model[0] if args.model else infer_model_type(model_path.stem)
                seed = args.seed
            else:
                if args.seed is None:
                    print("错误: 需要指定 --seed 或 --model_path")
                    sys.exit(1)

                base_dir = Path("experiments_dcase/DCASE_Results")
                # 尝试不同的目录结构
                possible_paths = [
                    base_dir / f"seed_{args.seed}" / 'models' / args.model[0] / args.model_file,
                    base_dir / f"{args.model[0]}_seed{args.seed}" / args.model_file,
                    base_dir / args.model[0] / f"seed_{args.seed}" / args.model_file,
                ]

                model_path = None
                for p in possible_paths:
                    if p.exists():
                        model_path = p
                        break

                if model_path is None:
                    print(f"错误: 找不到模型文件")
                    print(f"尝试的路径:")
                    for p in possible_paths:
                        print(f"  - {p}")
                    sys.exit(1)

                model_name = args.model[0]
                seed = args.seed

            print(f"\n{'='*70}")
            print(f"评估模型: {model_name} (seed={seed})")
            print(f"模型路径: {model_path}")
            print(f"{'='*70}")

            model_type = args.model[0] if args.model else infer_model_type(model_path.stem)
            model = load_dcase_model(str(model_path), model_type, device)
            print(f"模型已加载到 {device}")

            # 根据模型类型选择测试数据
            if model_type == 'beats':
                current_loaders = beats_test_loaders
                print("使用 BEATs 特征进行评估")
            else:
                current_loaders = test_loaders
                print("使用 ACT 特征进行评估")

            results = evaluate_dcase_model(model, current_loaders, verbose=not args.quiet)
            print_summary(results, model_name, seed, model_path.name, dataset='dcase')

            output_dir = Path(args.output_dir) if args.output_dir else model_path.parent / 'evaluation_results'
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{model_name}_{model_path.stem}_eval.json"
            save_results(results, output_file, model_name, seed, model_path.name, dataset='dcase')

            del model
            torch.cuda.empty_cache()

    else:
        # Whale数据集评估（原始逻辑）
        import config as cfg
        from evaluation_metrics import WhaleEventDetectionEvaluator, load_ground_truth
        from experiments.model_comparison import initialize_model
        from Data.audio_dataset import load_test_data, load_beats_snr_data

        if args.seed is None:
            print("错误: whale数据集需要指定 --seed")
            sys.exit(1)

        base_dir = Path("experiments/Snr_Results/model_comparison")
        seed_dir = base_dir / f"seed_{args.seed}"

        if not seed_dir.exists():
            print(f"错误: 找不到seed目录 {seed_dir}")
            print(f"可用的seed目录: {[d.name for d in base_dir.glob('seed_*')]}")
            sys.exit(1)

        txt_folder_path = cfg.PATH_CONFIG['test_path']

        has_beats = 'beats' in args.model
        has_other = any(m != 'beats' for m in args.model)

        test_loaders = None
        beats_test_loaders = None

        if has_other:
            print(f"加载 ACT 测试数据 (batch_size={args.batch_size})...")
            test_loaders = load_test_data(args.batch_size)

        if has_beats:
            print(f"加载 BEATs 测试数据 (batch_size={args.batch_size})...")
            beats_test_loaders = load_beats_snr_data('test', args.batch_size)

        for model_name in args.model:
            model_path = seed_dir / 'models' / model_name / args.model_file

            if not model_path.exists():
                print(f"警告: 找不到模型文件 {model_path}")
                model_dir = seed_dir / 'models' / model_name
                if model_dir.exists():
                    available_files = list(model_dir.glob('*.pth'))
                    print(f"  可用的模型文件: {[f.name for f in available_files]}")
                continue

            print(f"\n{'='*60}")
            print(f"评估模型: {model_name} (seed={args.seed}, {args.model_file})")
            print(f"模型路径: {model_path}")
            print(f"{'='*60}")

            model = load_whale_model(str(model_path), model_name, device)
            print(f"模型已加载到 {device}")

            if model_name == 'beats':
                current_loaders = beats_test_loaders
                print("使用 BEATs 特征进行评估")
            else:
                current_loaders = test_loaders
                print("使用 ACT 特征进行评估")

            results = evaluate_whale_model(model, current_loaders, txt_folder_path, verbose=not args.quiet)
            print_summary(results, model_name, args.seed, args.model_file, dataset='whale')

            output_dir = Path(args.output_dir) if args.output_dir else seed_dir / 'evaluation_results'
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{model_name}_{args.model_file.replace('.pth', '')}_eval.json"
            save_results(results, output_file, model_name, args.seed, args.model_file, dataset='whale')

            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
