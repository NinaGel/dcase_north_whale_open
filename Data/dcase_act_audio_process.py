import os
import pandas as pd
import numpy as np
import concurrent.futures
from tqdm import tqdm
import config_dcase as cfg
from Data.ACT import process_audio_to_features
import torch
import math


def time_to_frame(time, sr, hop_length):
    """将时间转换为帧索引"""
    return int(time * sr / hop_length)


def txt_to_dataframe(filepath_txt):
    """读取DCASE txt标注文件，转换为DataFrame
    
    Args:
        filepath_txt: txt文件路径
        
    Returns:
        DataFrame with columns ['Start', 'End', 'EventLabel']
    """
    # 如果文件不存在或为空，直接返回空DataFrame
    if not os.path.exists(filepath_txt) or os.path.getsize(filepath_txt) == 0:
        return pd.DataFrame(columns=['Start', 'End', 'EventLabel'])

    with open(filepath_txt, 'r', encoding='utf-8') as file:
        content = file.readlines()

    events_list = []
    for line in content:
        line = line.strip()  # 移除首尾空格
        if line:  # 检查行是否为空
            parts = line.split('\t')  # DCASE使用制表符分隔
            if len(parts) == 3:  # 确保每行有3个部分: Start, End, EventLabel
                events_list.append(parts)

    # 返回DataFrame
    df_events = pd.DataFrame(events_list, columns=['Start', 'End', 'EventLabel'])
    if not df_events.empty:
        df_events['Start'] = pd.to_numeric(df_events['Start'])
        df_events['End'] = pd.to_numeric(df_events['End'])
    
    return df_events


def get_dcase_label_mapping():
    """获取DCASE2020标签映射
    
    Returns:
        dict: 标签到索引的映射
    """
    # DCASE2020的10个标签类别（与config_dcase保持一致）
    dcase_labels = [
        'Alarm_bell_ringing',
        'Blender', 
        'Cat',
        'Dishes',
        'Dog',
        'Electric_shaver_toothbrush',
        'Frying',
        'Running_water',
        'Speech',
        'Vacuum_cleaner'
    ]
    
    return {label: idx for idx, label in enumerate(dcase_labels)}


def process_single_file(args):
    """处理单个DCASE音频文件的函数
    
    Args:
        args: 包含(i, filepath_wav, filepath_txt, act_frame, num_classes)的元组
        
    Returns:
        (features, labels): ACT特征和帧级标签
    """
    i, filepath_wav, filepath_txt, act_frame, num_classes = args
    
    # 初始化该文件的特征和标签
    act_feature, _, resample_sr = process_audio_to_features(
        filepath_wav, 
        cfg.DCASE_AUDIO_CONFIG['n_fft'], 
        cfg.DCASE_AUDIO_CONFIG['hop_length'], 
        cfg.DCASE_AUDIO_CONFIG['window']
    )
    
    # 将PyTorch张量转换为NumPy数组
    act_feature = act_feature.numpy()
    
    feature_shape = act_feature.shape
    freq_bins = int(cfg.DCASE_AUDIO_CONFIG['n_fft'] * 2000 / 8000)
    
    # 使用float16来减少内存使用
    features = np.zeros((freq_bins, act_frame), dtype=np.float16)
    features[:, :feature_shape[1]] = act_feature.astype(np.float16)
    
    # 使用bool类型存储标签以节省内存（DCASE有10个类别）
    labels = np.zeros((act_frame, num_classes), dtype=bool)
    
    # 读取标签
    if os.path.exists(filepath_txt):
        df_events = txt_to_dataframe(filepath_txt)
        if not df_events.empty:
            label_dict = get_dcase_label_mapping()
            
            for _, row in df_events.iterrows():
                start_frame = time_to_frame(row['Start'], resample_sr, cfg.DCASE_AUDIO_CONFIG['hop_length'])
                end_frame = time_to_frame(row['End'], resample_sr, cfg.DCASE_AUDIO_CONFIG['hop_length'])
                
                # 确保帧索引在有效范围内
                start_frame = max(0, min(start_frame, act_frame - 1))
                end_frame = max(0, min(end_frame, act_frame))
                
                # 获取事件标签索引
                event_label = row['EventLabel']
                if event_label in label_dict:
                    label_index = label_dict[event_label]
                    labels[start_frame:end_frame, label_index] = True
            
    return features, labels


def generate_data(audio_path, data_name):
    """生成DCASE ACT特征数据
    
    Args:
        audio_path: 音频数据路径（包含wav和txt子目录）
        data_name: 数据集名称（用于保存文件）
    """
    txt_path = os.path.join(audio_path, 'txt')
    wav_path = os.path.join(audio_path, 'wav')
    
    # 获取所有wav文件
    filenames_wav = sorted([f for f in os.listdir(wav_path) if f.endswith('.wav')])
    filepath_wav = [os.path.join(wav_path, d) for d in filenames_wav]
    filepath_txt = [os.path.join(txt_path, f.replace('.wav', '.txt')) for f in filenames_wav]
    
    # 计算常量
    act_frame = int((cfg.DCASE_AUDIO_CONFIG['sr'] * 10 - cfg.DCASE_AUDIO_CONFIG['n_fft']) / 
                    cfg.DCASE_AUDIO_CONFIG['hop_length']) + 1
    num_classes = cfg.DCASE_MODEL_CONFIG['num_classes']  # DCASE有10个类别
    freq_bins = int(cfg.DCASE_AUDIO_CONFIG['n_fft'] * 2000 / 8000)
    
    # 计算最优的批处理大小
    n_files = len(filepath_wav)
    optimal_batch_size = calculate_optimal_batch_size(freq_bins, act_frame, num_classes)
    
    print(f"\n{'='*60}")
    print(f"处理 {data_name} 数据集")
    print(f"{'='*60}")
    print(f"总文件数: {n_files}")
    print(f"ACT 帧数: {act_frame}")
    print(f"频率 bins: {freq_bins}")
    print(f"类别数: {num_classes}")
    print(f"批处理大小: {optimal_batch_size}")
    print(f"{'='*60}\n")
    
    # 预分配数组
    act_features = np.zeros((n_files, freq_bins, act_frame), dtype=np.float16)
    labels = np.zeros((n_files, act_frame, num_classes), dtype=bool)
    
    # 分批处理文件
    for batch_start in range(0, n_files, optimal_batch_size):
        batch_end = min(batch_start + optimal_batch_size, n_files)
        batch_args = [
            (i, wav, txt, act_frame, num_classes) 
            for i, (wav, txt) in enumerate(zip(
                filepath_wav[batch_start:batch_end], 
                filepath_txt[batch_start:batch_end]
            ), start=batch_start)
        ]
        
        # 并行处理当前批次
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = list(tqdm(
                executor.map(process_single_file, batch_args),
                total=len(batch_args),
                desc=f"处理 {data_name} 批次 {batch_start//optimal_batch_size + 1}/{math.ceil(n_files/optimal_batch_size)}"
            ))
        
        # 收集批次结果
        for i, (feature, label) in enumerate(results, start=batch_start):
            act_features[i] = feature
            labels[i] = label
    
    print(f"\n{data_name} ACT特征形状: {act_features.shape}")
    print(f"{data_name} 标签形状: {labels.shape}")
    
    # 统计标签分布
    label_counts = labels.sum(axis=(0, 1))
    dcase_labels = list(get_dcase_label_mapping().keys())
    print(f"\n{data_name} 标签分布:")
    for label, count in zip(dcase_labels, label_counts):
        print(f"  {label}: {int(count)} 帧")
    
    # 直接在数据目录下保存
    save_path = os.path.join(audio_path, f'act_dcase_{data_name}.npz')
    print(f"\n保存特征到: {save_path}")
    np.savez_compressed(
        save_path,
        features=act_features,
        labels=labels.astype(np.uint8),  # 将bool转换为uint8以节省空间
        filenames=filenames_wav
    )
    print(f"✅ 特征已成功保存到 {save_path}\n")


def calculate_optimal_batch_size(freq_bins, act_frame, num_classes):
    """计算最优批处理大小
    
    Args:
        freq_bins: 频率bins数
        act_frame: ACT帧数
        num_classes: 类别数
        
    Returns:
        optimal_batch_size: 最优批处理大小
    """
    # RTX 4090 有24GB显存
    gpu_memory = 24 * 1024 * 1024 * 1024  # 转换为字节
    
    # 估算单个样本的内存占用（特征 + 标签）
    # 特征: freq_bins * act_frame * 2 (float16)
    # 标签: act_frame * num_classes * 1 (bool/uint8)
    sample_size = freq_bins * act_frame * 2 + act_frame * num_classes
    
    # 预留50%显存给模型和其他开销
    available_memory = gpu_memory * 0.5
    
    # 计算理论最大批大小
    max_batch_size = int(available_memory / sample_size)
    
    # 确保批大小是8的倍数（有利于GPU优化）
    optimal_batch_size = max(8, math.floor(max_batch_size / 8) * 8)
    
    return min(optimal_batch_size, 256)  # 限制最大批大小为256


if __name__ == "__main__":
    # 设置NumPy使用float16
    np.set_printoptions(precision=3, suppress=True)
    
    # DCASE SNR合成数据集的基础路径
    base_path = 'Data/dcase_synthetic_10k/dcase_snr_10k_8000HZ'
    
    # 定义要处理的所有数据集
    # SNR级别: very_low, low, medium, high
    # 数据分割: train, test, validation
    snr_levels = ['snr_very_low', 'snr_low', 'snr_medium', 'snr_high']
    data_splits = ['train', 'test', 'validation']
    
    datasets = []
    for snr_level in snr_levels:
        for split in data_splits:
            # 构建路径
            audio_path = os.path.join(base_path, snr_level, split)
            # 构建数据集名称，例如: train_very_low, test_high, validation_medium
            data_name = f"{split}_{snr_level.replace('snr_', '')}"
            datasets.append((audio_path, data_name))
    
    print(f"\n{'='*60}")
    print(f"DCASE ACT 特征提取")
    print(f"{'='*60}")
    print(f"将处理 {len(datasets)} 个数据集:")
    for audio_path, data_name in datasets:
        print(f"  - {data_name}: {audio_path}")
    print(f"{'='*60}\n")
    
    # 处理所有数据集
    for audio_path, data_name in datasets:
        if os.path.exists(audio_path):
            print(f"\n{'#'*60}")
            print(f"开始处理: {data_name}")
            print(f"{'#'*60}")
            generate_data(audio_path, data_name)
        else:
            print(f"\n⚠️  路径不存在，跳过: {audio_path}")
    
    print(f"\n{'='*60}")
    print(f"✅ 所有数据集处理完成！")
    print(f"{'='*60}\n")
