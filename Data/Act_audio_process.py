import os
import pandas as pd
import numpy as np
import concurrent.futures
from tqdm import tqdm
import config as cfg
from Data.ACT import process_audio_to_features
import torch
from torch.utils.data import DataLoader
import math


def time_to_frame(time, sr, hop_length):
    return int(time * sr / hop_length)


def txt_to_dataframe(filepath_txt):
    # 如果文件不存在或为空，直接返回空DataFrame
    if not os.path.exists(filepath_txt) or os.path.getsize(filepath_txt) == 0:
        return pd.DataFrame(columns=['Start', 'End', 'CallType'])

    with open(filepath_txt, 'r', encoding='utf-8') as file:
        content = file.readlines()  # 读取每一行

    events_list = []
    for line in content:
        line = line.strip()  # 移除首尾空格
        if line:  # 检查行是否为空
            parts = line.split('\t')  # 假设列之间是用制表符分隔的
            if len(parts) == 3:  # 确保每行有3个部分
                events_list.append(parts)

    # 直接返回DataFrame，不需要额外的警告信息
    df_events = pd.DataFrame(events_list, columns=['Start', 'End', 'CallType'])
    if not df_events.empty:
        df_events['Start'] = pd.to_numeric(df_events['Start'])
        df_events['End'] = pd.to_numeric(df_events['End'])
    
    return df_events


def process_single_file(args):
    """处理单个音频文件的函数"""
    i, filepath_wav, filepath_txt, act_frame, labels_name_list = args
    
    # 初始化该文件的特征和标签
    act_feature, _, resample_sr = process_audio_to_features(
        filepath_wav, 
        cfg.AUDIO_CONFIG['n_fft'], 
        cfg.AUDIO_CONFIG['hop_length'], 
        cfg.AUDIO_CONFIG['window']
    )
    
    # 将PyTorch张量转换为NumPy数组
    act_feature = act_feature.numpy()
    
    feature_shape = act_feature.shape
    freq_bins = int(cfg.AUDIO_CONFIG['n_fft'] * 2000 / 8000)
    
    # 使用float16来减少内存使用
    features = np.zeros((freq_bins, act_frame), dtype=np.float16)
    features[:, :feature_shape[1]] = act_feature.astype(np.float16)
    
    # 使用bool类型存储标签以节省内存
    labels = np.zeros((act_frame, 4), dtype=bool)
    
    if os.path.exists(filepath_txt):
        df_events = txt_to_dataframe(filepath_txt)
        if not df_events.empty:
            label_dict = {label: i for i, label in enumerate(labels_name_list)}
            
            for _, row in df_events.iterrows():
                start_frame = time_to_frame(row['Start'], resample_sr, cfg.AUDIO_CONFIG['hop_length'])
                end_frame = time_to_frame(row['End'], resample_sr, cfg.AUDIO_CONFIG['hop_length'])
                label_index = label_dict[row['CallType']]
                labels[start_frame:end_frame, label_index] = True
            
    return features, labels


def generate_data(audio_path, data_name):
    txt_path = os.path.join(audio_path, 'txt')
    wav_path = os.path.join(audio_path, 'wav')
    
    filenames_wav = sorted([f for f in os.listdir(wav_path) if f.endswith('wav')])
    filepath_wav = [os.path.join(wav_path, d) for d in filenames_wav]
    filepath_txt = [os.path.join(txt_path, f.replace('.wav', '.txt')) for f in filenames_wav]
    
    # 计算常量
    act_frame = int((cfg.AUDIO_CONFIG['sr'] * 10 - cfg.AUDIO_CONFIG['n_fft']) / cfg.AUDIO_CONFIG['hop_length']) + 1
    labels_name_list = ['upcall', 'gunshot', 'scream', 'moancall']
    freq_bins = int(cfg.AUDIO_CONFIG['n_fft'] * 2000 / 8000)
    
    # 计算最优的批处理大小
    n_files = len(filepath_wav)
    optimal_batch_size = calculate_optimal_batch_size(freq_bins, act_frame)
    
    # 预分配数组
    act_features = np.zeros((n_files, freq_bins, act_frame), dtype=np.float16)
    labels = np.zeros((n_files, act_frame, 4), dtype=bool)
    
    # 分批处理文件
    for batch_start in range(0, n_files, optimal_batch_size):
        batch_end = min(batch_start + optimal_batch_size, n_files)
        batch_args = [
            (i, wav, txt, act_frame, labels_name_list) 
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
                desc=f"Processing {data_name} batch {batch_start//optimal_batch_size + 1}"
            ))
        
        # 收集批次结果
        for i, (feature, label) in enumerate(results, start=batch_start):
            act_features[i] = feature
            labels[i] = label
    
    print(f"{data_name} features shape:", act_features.shape)
    print(f"{data_name} labels shape:", labels.shape)
    
    # 直接在测试数据目录下保存
    save_path = os.path.join(audio_path, f'act_whale_dataset_{data_name}.npz')
    print(f"Saving features to: {save_path}")
    np.savez_compressed(
        save_path,
        features=act_features,
        labels=labels.astype(np.uint8),  # 将bool转换为uint8以节省空间
        filenames=filenames_wav
    )
    print(f"Features saved successfully to {save_path}")


def calculate_optimal_batch_size(freq_bins, act_frame):
    """计算最优批处理大小"""
    # RTX 4090 有24GB显存
    gpu_memory = 24 * 1024 * 1024 * 1024  # 转换为字节
    
    # 估算单个样本的内存占用（特征 + 标签）
    sample_size = (freq_bins * act_frame * 2 + act_frame * 4) * 4  # float32占4字节
    
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
    
    # 定义数据集路径
    datasets = [
        ('scaper_audio/snr_scaper_audio/test/snr_high', 'test_high'),
        ('scaper_audio/snr_scaper_audio/test/snr_medium', 'test_medium'),
        ('scaper_audio/snr_scaper_audio/test/snr_low', 'test_low'),
        ('scaper_audio/snr_scaper_audio/test/snr_very_low', 'test_very_low')
    ]
    
    for audio_path, data_name in datasets:
        print(f"\nProcessing {data_name} dataset...")
        generate_data(audio_path, data_name)
