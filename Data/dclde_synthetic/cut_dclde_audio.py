import pandas as pd
import librosa
import numpy as np
import os
import soundfile as sf
from pathlib import Path


def load_annotations(csv_path):
    """
    加载并处理CSV格式的注释文件
    参数:
        csv_path: CSV文件的路径
    返回:
        处理后的DataFrame
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    # 确保所需列名存在
    required_columns = ['start_sec', 'end_sec', 'buoy_id', 'call_type']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV文件缺少必要的列: {required_columns}")
    
    # 按照buoy_id和时间戳排序
    df = df.sort_values(['buoy_id', 'start_sec'])
    return df


def cut_audio(audio_path, start_time, end_time, target_duration=10.0):
    """
    切割音频文件
    参数:
        audio_path: 音频文件路径
        start_time: 开始时间(秒)
        end_time: 结束时间(秒)
        target_duration: 目标持续时间(秒)
    返回:
        切割后的音频数据和采样率
    """
    # 加载音频文件
    y, sr = librosa.load(audio_path, sr=None)
    
    # 计算call持续时间
    call_duration = end_time - start_time
    
    # 计算需要在call前后添加的时间
    padding_time = (target_duration - call_duration) / 2
    
    # 计算切割点
    cut_start = max(0, int((start_time - padding_time) * sr))
    cut_end = min(len(y), int((end_time + padding_time) * sr))
    
    # 切割音频
    audio_cut = y[cut_start:cut_end]
    
    return audio_cut, sr

def get_audio_files_map(audio_dir):
    """
    获取音频文件夹中的所有音频文件,创建buoy_id到文件名的映射
    参数:
        audio_dir: 音频文件目录
    返回:
        dict: buoy_id到音频文件名的映射
    """
    audio_files = {}
    
    for file in os.listdir(audio_dir):
        if file.endswith('.wav'):
            # 按照_分割文件名
            parts = file.split('_')
            for part in parts:
                # 查找包含'D'的部分
                if part.startswith('D'):
                    # 去掉'D'获取buoy_id
                    try:
                        buoy_id = int(part.replace('D', ''))
                        audio_files[buoy_id] = file
                        print(f"找到音频文件: {file} -> buoy_id: {buoy_id}")
                        break
                    except ValueError:
                        continue
    
    return audio_files

def process_dclde_dataset(audio_dir, csv_path, output_dir):
    """
    处理DCLDE数据集
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载注释文件
    annotations = pd.read_csv(csv_path)
    
    # 获取所有唯一的buoy_id并排序
    unique_buoy_ids = sorted(annotations['buoy_id'].unique())
    print(f"发现的buoy_id列表: {unique_buoy_ids}")
    
    # 获取音频文件映射
    audio_files_map = get_audio_files_map(audio_dir)
    if not audio_files_map:
        raise ValueError("未找到任何音频文件")
    
    # 检查是否所有buoy_id都有对应的音频文件
    missing_buoys = [bid for bid in unique_buoy_ids if bid not in audio_files_map]
    if missing_buoys:
        print(f"警告: 以下buoy_id没有对应的音频文件: {missing_buoys}")
    
    # 按buoy_id处理数据
    for buoy_id in unique_buoy_ids:
        if buoy_id not in audio_files_map:
            print(f"跳过buoy_id {buoy_id}: 找不到对应的音频文件")
            continue
        
        # 获取当前buoy_id的所有注释
        buoy_annotations = annotations[annotations['buoy_id'] == buoy_id]
        audio_filename = audio_files_map[buoy_id]
        audio_path = os.path.join(audio_dir, audio_filename)
        
        print(f"处理 buoy_id {buoy_id} 的音频文件: {audio_filename}")
        print(f"找到 {len(buoy_annotations)} 条注释")
        
        # 处理每个注释
        for idx, row in buoy_annotations.iterrows():
            start_time = float(row['start_sec'])
            end_time = float(row['end_sec'])
            call_type = row['call_type']
            
            try:
                # 切割音频
                audio_cut, sr = cut_audio(audio_path, start_time, end_time)
                
                # 构建输出文件名
                score_info = f"_score{row['score']}" if 'score' in row and not pd.isna(row['score']) else ""
                output_filename = f"dclde_{call_type}_{int(buoy_id)}_{idx}{score_info}.wav"
                output_path = os.path.join(output_dir, output_filename)
                
                # 使用soundfile保存音频
                sf.write(output_path, audio_cut, sr)
                print(f"已保存: {output_filename}")
                
            except Exception as e:
                print(f"处理音频时出错 {audio_filename}, 时间: {start_time}-{end_time}: {str(e)}")

if __name__ == "__main__":
    # 设置路径
    audio_dir = "C:/Users/luo_o/Desktop/learnning/program/North_Whale_Sed/Data/scaper_audio/dclde_update_data/July31"
    csv_path = "C:/Users/luo_o/Desktop/learnning/program/North_Whale_Sed/Data/scaper_audio/dclde_update_data/doc/annotations_corrected_7.31.csv"
    output_dir = "dclde_update_data/cut_audio_July31"
    
    # 处理数据集
    process_dclde_dataset(audio_dir, csv_path, output_dir)
