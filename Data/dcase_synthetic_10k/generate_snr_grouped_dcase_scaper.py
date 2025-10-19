#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DCASE数据集SNR分级生成脚本 (直接scaper版本)

这个版本直接使用scaper库，不依赖desed包，
解决Python 3.12+版本的兼容性问题。

基于soundscaper_whale.py的SNR分级经验，
生成具有四个SNR级别的DCASE数据集。
"""

import os
import json
import numpy as np

# NumPy 2.0 兼容性补丁
# scaper使用了np.Inf等在NumPy 2.0中已移除的别名
if not hasattr(np, 'Inf'):
    np.Inf = np.inf
    np.NaN = np.nan
    np.Infinity = np.inf
    np.NAN = np.nan
    print("已应用NumPy 2.0兼容性补丁")

import scaper
import jams
import soundfile as sf  # 用于快速获取音频信息
from pathlib import Path
from typing import Dict, List, Tuple
import glob
from tqdm import tqdm
import warnings


class DirectSoundscapeGenerator:
    """直接使用scaper的音景生成器，不依赖desed"""
    
    def __init__(self, fg_folder, bg_folder, output_base,
                 event_occurences_file, snr_group='all', snr_range=None,
                 samplerate=16000, duration=10.0, ref_db=-55,
                 polyphony_weights=None, reverb_range=None,
                 split_paths=None):
        """
        Args:
            fg_folder: 前景音频文件夹路径（默认/后备路径）
            bg_folder: 背景音频文件夹路径（默认/后备路径）
            output_base: 输出基础路径
            event_occurences_file: 事件共现配置文件路径
            snr_group: SNR分组名称
            snr_range: SNR范围 (min, max)，如果提供则使用此值
            samplerate: 目标采样率，默认16000（DESED标准）
            duration: 音频时长（秒），默认10.0
            ref_db: 参考分贝，默认-50（DESED推荐-55）
            polyphony_weights: 复音度分布权重字典，如 {'1-2_events': 0.35, '3-4_events': 0.45, '5_events': 0.20}
            reverb_range: 混响范围元组，如 (0.1, 0.3)
            split_paths: 按split分组的soundbank路径字典，格式:
                         {'train': {'fg_folder': ..., 'bg_folder': ...},
                          'val': {'fg_folder': ..., 'bg_folder': ...},
                          'test': {'fg_folder': ..., 'bg_folder': ...}}
        """
        self.fg_folder = fg_folder
        self.bg_folder = bg_folder
        self.output_base = output_base
        self.event_occurences_file = event_occurences_file
        self.snr_group = snr_group
        self.split_paths = split_paths  # 保存按split分组的路径

        # SNR范围（可从外部传入）
        if snr_range:
            self.snr_range = snr_range
        else:
            # 默认值（与whale数据集对齐，范围-10~10 dB）
            default_ranges = {
                'very_low': (-10, -5),
                'low': (-5, 0),
                'medium': (0, 5),
                'high': (5, 10)
            }
            self.snr_range = default_ranges.get(snr_group, (-10, 10))

        # 基本参数（可从外部传入）
        self.duration = duration
        self.ref_db = ref_db
        self.samplerate = samplerate

        # 复音度分布策略
        self.polyphony_weights = polyphony_weights or {'1-2_events': 0.35, '3-4_events': 0.45, '5_events': 0.20}

        # 混响范围
        self.reverb_range = reverb_range or (0.1, 0.1)  # 默认固定0.1
        
        # 设置输出路径
        if snr_group != 'all':
            self.output_base = os.path.join(output_base, f'snr_{snr_group}')
        
        # 加载事件共现配置
        with open(event_occurences_file, 'r') as f:
            self.co_occur_dict = json.load(f)
            
        # 加载音频文件列表
        self._load_audio_files()
    
    def _load_audio_files(self):
        """加载前景和背景音频文件列表"""
        # 检查路径存在性
        if not os.path.exists(self.fg_folder):
            raise FileNotFoundError(f"前景音频文件夹不存在: {self.fg_folder}")
        if not os.path.exists(self.bg_folder):
            raise FileNotFoundError(f"背景音频文件夹不存在: {self.bg_folder}")
        
        # 加载前景音频文件
        # 关键：按照官方实现，我们需要为每个基础类别创建一个包含所有变体文件的列表
        # 这样在选择文件时，label使用基础类别名，但文件可以来自任何变体文件夹
        self.foreground_files = {}
        self.base_classes = []  # 基础类别列表（不含_nOn等后缀）
        
        # 首先收集所有文件夹
        all_folders = {}
        for class_dir in os.listdir(self.fg_folder):
            class_path = os.path.join(self.fg_folder, class_dir)
            if os.path.isdir(class_path):
                wav_files = glob.glob(os.path.join(class_path, "*.wav"))
                if wav_files:
                    all_folders[class_dir] = wav_files
        
        # 按基础类别组织文件
        base_class_map = {}  # 基础类别 -> [所有变体的文件列表]
        for folder_name, files in all_folders.items():
            # 提取基础类别名
            if '_nOn_nOff' in folder_name:
                base_class = folder_name.replace('_nOn_nOff', '')
            elif '_nOn' in folder_name:
                base_class = folder_name.replace('_nOn', '')
            elif '_nOff' in folder_name:
                base_class = folder_name.replace('_nOff', '')
            else:
                base_class = folder_name
            
            if base_class not in base_class_map:
                base_class_map[base_class] = []
            base_class_map[base_class].extend(files)
        
        # 将所有文件按基础类别存储
        self.foreground_files = base_class_map
        self.base_classes = list(base_class_map.keys())
        
        # 加载背景音频文件
        # 首先尝试直接在根目录查找
        self.background_files = glob.glob(os.path.join(self.bg_folder, "*.wav"))
        
        # 如果根目录没有文件，尝试在子文件夹中查找（DESED的背景文件在sins/tut等子文件夹）
        if not self.background_files:
            print("根目录未找到背景音频，搜索子文件夹...")
            for subdir in os.listdir(self.bg_folder):
                subpath = os.path.join(self.bg_folder, subdir)
                if os.path.isdir(subpath):
                    bg_files = glob.glob(os.path.join(subpath, "*.wav"))
                    if bg_files:
                        print(f"  从子文件夹加载: {subdir} ({len(bg_files)}个文件)")
                        self.background_files.extend(bg_files)
        
        print(f"\n音频文件加载统计:")
        print(f"  基础类别: {self.base_classes}")
        print(f"  前景音频文件总数: {sum(len(files) for files in self.foreground_files.values())}个")
        print(f"  背景音频文件总数: {len(self.background_files)}个")

        # 错误检查
        if not self.foreground_files:
            raise ValueError(f"未找到任何前景音频文件，请检查路径: {self.fg_folder}")
        if not self.background_files:
            raise ValueError(f"未找到任何背景音频文件！\n"
                           f"已搜索路径: {self.bg_folder}\n"
                           f"请确认soundbank已完整下载")

    def _switch_soundbank(self, split):
        """根据split切换soundbank路径并重新加载音频文件

        Args:
            split: 数据集划分名称 ('train', 'val', 'test')
        """
        if self.split_paths is None or split not in self.split_paths:
            return

        split_config = self.split_paths[split]
        new_fg_folder = split_config.get('fg_folder', self.fg_folder)
        new_bg_folder = split_config.get('bg_folder', self.bg_folder)

        # 检查路径是否有变化
        if new_fg_folder == self.fg_folder and new_bg_folder == self.bg_folder:
            return

        print(f"\n切换到 {split} soundbank:")
        print(f"  前景: {new_fg_folder}")
        print(f"  背景: {new_bg_folder}")

        # 更新路径
        self.fg_folder = new_fg_folder
        self.bg_folder = new_bg_folder

        # 重新加载音频文件
        self._load_audio_files()

    def get_snr_range(self):
        """获取当前SNR分组的范围"""
        return self.snr_range
    
    def _setup_folders(self, split):
        """创建输出文件夹结构（与soundscaper_whale.py保持一致）"""
        folders = {
            'audio': os.path.join(self.output_base, split, 'wav'),
            'jams': os.path.join(self.output_base, split, 'jams'),
            'txt': os.path.join(self.output_base, split, 'txt')
        }
        
        for folder in folders.values():
            os.makedirs(folder, exist_ok=True)
            
        return folders
    
    def _sample_num_events_with_weights(self):
        """根据复音度权重采样事件数量"""
        ranges = []
        weights = []

        for key, weight in self.polyphony_weights.items():
            if '1-2_events' in key:
                ranges.append((1, 2))
            elif '3-4_events' in key:
                ranges.append((3, 4))
            elif '5_events' in key or '5+_events' in key:
                ranges.append((5, 5))
            weights.append(weight)

        # 归一化权重
        weights = np.array(weights) / np.sum(weights)

        # 选择范围
        selected_range = ranges[np.random.choice(len(ranges), p=weights)]

        # 在范围内随机选择
        return np.random.randint(selected_range[0], selected_range[1] + 1)

    def _generate_events_from_occurences(self, sc, target_num_events=None):
        """根据事件共现配置生成事件（兼容DESED格式）"""
        # DESED格式：每个主类别有proba和co-occurences
        # 首先根据概率选择一个主类别
        main_classes = []
        main_probas = []

        for event_class, class_config in self.co_occur_dict.items():
            # 检查这个基础类别是否有对应的音频文件
            if event_class in self.foreground_files:
                main_classes.append(event_class)
                main_probas.append(class_config.get('proba', 0.1))

        if not main_classes:
            return

        # 归一化概率
        main_probas = np.array(main_probas)
        main_probas = main_probas / np.sum(main_probas)

        # 选择主类别
        main_class = np.random.choice(main_classes, p=main_probas)
        co_occur_config = self.co_occur_dict[main_class]['co-occurences']

        # 获取共现事件的数量
        if target_num_events is not None:
            # 使用指定的事件数量（来自复音度分布策略）
            num_events = target_num_events
        else:
            # 使用原始逻辑
            mean_events = co_occur_config.get('mean_events', 3)
            max_events = co_occur_config.get('max_events', 5)
            num_events = min(np.random.poisson(mean_events) + 1, max_events)
        
        # 获取可能的共现类别
        co_classes = co_occur_config.get('classes', [main_class])
        co_probas = co_occur_config.get('probas', [1.0] * len(co_classes))
        
        # 过滤出有音频文件的类别
        valid_classes = []
        valid_probas = []
        for cls, prob in zip(co_classes, co_probas):
            if cls in self.foreground_files:
                valid_classes.append(cls)
                valid_probas.append(prob)
        
        if not valid_classes:
            return
        
        # 归一化概率
        valid_probas = np.array(valid_probas)
        valid_probas = valid_probas / np.sum(valid_probas)
        
        # 获取SNR范围
        snr_min, snr_max = self.get_snr_range()
        
        # 生成多个事件
        DEBUG_MODE = os.environ.get('DEBUG_SCAPER', '0') == '1'
        
        for event_idx in range(num_events):
            # 选择事件类别（基础类别）
            event_class = np.random.choice(valid_classes, p=valid_probas)
            
            # 从该基础类别的所有文件中随机选择一个
            # 注意：self.foreground_files[event_class] 包含了该类所有变体的文件
            audio_file = np.random.choice(self.foreground_files[event_class])
            
            # 获取文件所在的实际文件夹名（可能包含_nOn等后缀）
            # 这是scaper验证label时需要匹配的
            file_parent_folder = os.path.basename(os.path.dirname(audio_file))
            
            if DEBUG_MODE:
                print(f"\n[事件{event_idx+1}调试]")
                print(f"  基础类别: {event_class}")
                print(f"  选择的文件: {audio_file}")
                print(f"  文件父文件夹: {file_parent_folder}")
                print(f"  fg_folder根目录: {self.fg_folder}")
            
            # 获取音频时长（使用soundfile更快，不需要解码整个文件）
            try:
                info = sf.info(audio_file)
                audio_duration = info.duration
            except Exception as e:
                print(f"警告: 无法读取音频文件 {audio_file}: {e}")
                continue
            
            # 限制事件时长不超过soundscape时长
            max_event_duration = min(audio_duration, self.duration)
            
            # 计算最晚开始时间
            latest_start = max(0, self.duration - max_event_duration)
            
            # 添加事件
            # 关键：label 必须是文件父文件夹的实际名称（包含_nOn等后缀）
            # 这样scaper验证时 file_parent_folder == label 才会通过
            if DEBUG_MODE:
                print(f"  准备添加事件:")
                print(f"    label: {file_parent_folder}")
                print(f"    source_file: {audio_file}")
                print(f"    event_duration: {max_event_duration}")
                print(f"    snr_range: ({snr_min}, {snr_max})")
            
            try:
                sc.add_event(
                    label=('const', file_parent_folder),
                    source_file=('const', audio_file),
                    source_time=('const', 0),
                    event_time=('uniform', 0, latest_start) if latest_start > 0 else ('const', 0),
                    event_duration=('const', max_event_duration),
                    snr=('uniform', snr_min, snr_max),
                    pitch_shift=('uniform', -2, 2),
                    time_stretch=('uniform', 0.9, 1.0)  # 缩小范围避免超出时长警告
                )
                if DEBUG_MODE:
                    print(f"  ✓ 事件添加成功")
            except Exception as e:
                if DEBUG_MODE:
                    print(f"  ✗ 事件添加失败: {e}")
                raise
    
    def generate_dataset(self, num_train=2000, num_val=500, num_test=0,
                        remove_high_polyphony=True, max_polyphony=3):
        """生成训练集、验证集和测试集"""
        
        # 过滤scaper的time_stretch警告（避免输出混乱）
        warnings.filterwarnings('ignore', message='.*event duration.*stretch factor.*')
        
        # 构建split列表（跳过样本数为0的split）
        splits = []
        if num_train > 0:
            splits.append(('train', num_train))
        if num_val > 0:
            splits.append(('val', num_val))
        if num_test > 0:
            splits.append(('test', num_test))
        
        for split, num_samples in splits:
            print(f"\n=== 生成{split}数据集 (SNR组: {self.snr_group}) ===")
            print(f"目标样本数: {num_samples}")

            # 切换到对应split的soundbank（防止数据泄露）
            self._switch_soundbank(split)

            # 设置输出文件夹
            folders = self._setup_folders(split)
            
            # 获取SNR范围
            snr_min, snr_max = self.get_snr_range()
            print(f"使用SNR范围: {snr_min} ~ {snr_max} dB")
            
            generated_count = 0
            attempt_count = 0
            
            # 创建进度条
            pbar = tqdm(total=num_samples, desc=f"{split}集生成", unit="样本")
            
            while generated_count < num_samples and attempt_count < num_samples * 2:
                attempt_count += 1
                
                try:
                    # 创建scaper对象
                    DEBUG_MODE = os.environ.get('DEBUG_SCAPER', '0') == '1'
                    if DEBUG_MODE and attempt_count == 1:
                        print(f"\n[Scaper初始化调试]")
                        print(f"  fg_folder: {self.fg_folder}")
                        print(f"  bg_folder: {self.bg_folder}")
                        print(f"  duration: {self.duration}")
                        print(f"  ref_db: {self.ref_db}")
                    
                    sc = scaper.Scaper(
                        self.duration, 
                        self.fg_folder, 
                        self.bg_folder
                    )
                    sc.sr = self.samplerate  # 设置采样率作为对象属性
                    sc.protected_labels = []
                    sc.ref_db = self.ref_db
                    
                    # 添加背景
                    if self.background_files:
                        bg_file = np.random.choice(self.background_files)
                        # 获取背景文件的父文件夹名（scaper要求label与父文件夹名匹配）
                        bg_parent_folder = os.path.basename(os.path.dirname(bg_file))
                        sc.add_background(
                            label=('const', bg_parent_folder),
                            source_file=('const', bg_file),
                            source_time=('const', 0)
                        )
                    
                    # 根据复音度分布策略采样事件数量
                    target_num_events = self._sample_num_events_with_weights()

                    # 根据事件共现配置添加前景事件
                    self._generate_events_from_occurences(sc, target_num_events=target_num_events)

                    # 设置输出文件路径
                    base_filename = f"soundscape_dcase_{split}_{generated_count}"
                    audiofile = os.path.join(folders['audio'], f"{base_filename}.wav")
                    jamsfile = os.path.join(folders['jams'], f"{base_filename}.jams")
                    txtfile = os.path.join(folders['txt'], f"{base_filename}.txt")

                    # 采样混响值
                    reverb_val = np.random.uniform(self.reverb_range[0], self.reverb_range[1])

                    # 生成音频（Scaper已经按目标采样率生成，无需后续重采样）
                    sc.generate(
                        audiofile,
                        jamsfile,
                        allow_repeated_label=True,
                        allow_repeated_source=True,
                        reverb=reverb_val,
                        disable_sox_warnings=True,
                        no_audio=False,
                        txt_path=txtfile
                    )
                    
                    # 检查复音度（如果需要）
                    if remove_high_polyphony:
                        polyphony = self._get_polyphony_from_jams(jamsfile)
                        if polyphony > max_polyphony:
                            # 删除生成的文件
                            for filepath in [audiofile, jamsfile, txtfile]:
                                if os.path.exists(filepath):
                                    os.remove(filepath)
                            continue
                    
                    generated_count += 1
                    pbar.update(1)  # 更新进度条
                        
                except Exception as e:
                    if DEBUG_MODE:
                        print(f"\n=== 调试信息 (第{attempt_count}次尝试) ===")
                        print(f"错误: {e}")
                        print(f"错误类型: {type(e).__name__}")
                        import traceback
                        traceback.print_exc()
                        print("=" * 50)
                    continue
            
            pbar.close()  # 关闭进度条
            
            if generated_count < num_samples:
                print(f"警告: 只生成了 {generated_count}/{num_samples} 个有效文件")
            
            # 生成TSV标签文件
            self._create_tsv_labels(folders, split)
            
            print(f"完成{split}数据集生成")
    
    def _get_polyphony_from_jams(self, jams_file):
        """从JAMS文件获取复音度"""
        try:
            jam = jams.load(jams_file)
            
            # 找到所有前景事件的时间范围
            events = []
            for ann in jam.annotations:
                if ann.namespace == 'scaper':
                    for obs in ann.data:
                        if obs.value['role'] == 'foreground':
                            start_time = obs.time
                            end_time = obs.time + obs.duration
                            events.append((start_time, end_time))
            
            if not events:
                return 0
            
            # 计算最大重叠数（简化版本）
            max_polyphony = 0
            time_points = []
            for start, end in events:
                time_points.append((start, 1))  # 开始事件
                time_points.append((end, -1))   # 结束事件
            
            time_points.sort()
            current_polyphony = 0
            
            for _, delta in time_points:
                current_polyphony += delta
                max_polyphony = max(max_polyphony, current_polyphony)
            
            return max_polyphony
            
        except Exception as e:
            print(f"计算复音度时出错: {e}")
            return 0
    
    def _create_tsv_labels(self, folders, split):
        """创建TSV格式的标签文件"""
        # TSV文件放在split目录下（与wav/jams/txt同级）
        tsv_file = os.path.join(self.output_base, split, f'{split}_{self.snr_group}.tsv')
        
        with open(tsv_file, 'w') as f:
            f.write("filename\tonset\toffset\tevent_label\n")
            
            # 处理所有JAMS文件
            jams_files = glob.glob(os.path.join(folders['jams'], "*.jams"))
            
            for jams_file in sorted(jams_files):
                try:
                    jam = jams.load(jams_file)
                    basename = os.path.splitext(os.path.basename(jams_file))[0]
                    
                    for ann in jam.annotations:
                        if ann.namespace == 'scaper':
                            for obs in ann.data:
                                if obs.value['role'] == 'foreground':
                                    onset = obs.time
                                    offset = obs.time + obs.duration
                                    label = obs.value['label']
                                    # 标签去除_nOn/_nOff/_nOn_nOff后缀，保留基础类别名
                                    base_label = label.replace('_nOn_nOff', '').replace('_nOn', '').replace('_nOff', '')
                                    f.write(f"{basename}.wav\t{onset:.3f}\t{offset:.3f}\t{base_label}\n")
                                    
                except Exception as e:
                    print(f"处理JAMS文件 {jams_file} 时出错: {e}")
                    
        print(f"TSV标签文件已生成: {tsv_file}")
    
    def validate_snr_distribution(self, split='val', num_samples=50):
        """验证生成数据的SNR分布"""
        
        jams_folder = os.path.join(self.output_base, split, 'jams')
        if not os.path.exists(jams_folder):
            print(f"验证目录不存在: {jams_folder}")
            return []
            
        jams_files = glob.glob(os.path.join(jams_folder, "*.jams"))[:num_samples]
        
        snr_values = []
        for jams_file in jams_files:
            try:
                jam = jams.load(jams_file)
                
                for ann in jam.annotations:
                    if ann.namespace == 'scaper':
                        for obs in ann.data:
                            if obs.value['role'] == 'foreground':
                                snr_values.append(float(obs.value['snr']))
            except Exception as e:
                print(f"处理JAMS文件 {jams_file} 时出错: {e}")
                continue
        
        if snr_values:
            print(f"\nSNR验证结果（样本数：{len(snr_values)}）:")
            print(f"  平均值: {np.mean(snr_values):.2f} dB")
            print(f"  标准差: {np.std(snr_values):.2f} dB") 
            print(f"  范围: {np.min(snr_values):.2f} ~ {np.max(snr_values):.2f} dB")
            
            # 检查是否在预期范围内
            expected_min, expected_max = self.get_snr_range()
            in_range = all(expected_min <= snr <= expected_max for snr in snr_values)
            print(f"  是否在预期范围内: {'是' if in_range else '否'}")
            
            # 保存统计信息
            stats = {
                'snr_group': self.snr_group,
                'expected_range': [expected_min, expected_max],
                'actual_stats': {
                    'mean': float(np.mean(snr_values)),
                    'std': float(np.std(snr_values)),
                    'min': float(np.min(snr_values)),
                    'max': float(np.max(snr_values)),
                    'count': len(snr_values)
                },
                'in_range': in_range,
                'all_values': snr_values
            }
            
            stats_file = os.path.join(self.output_base, split, f'snr_validation_{self.snr_group}.json')
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"  验证统计已保存至: {stats_file}")
        else:
            print("未找到有效的SNR值进行验证")
        
        return snr_values


def create_default_event_occurences():
    """创建默认的事件共现配置"""
    default_config = {
        "0": {
            "number": 1,
            "events": []
        },
        "1": {
            "number": 10,
            "events": [
                {"event_class": "Alarm_bell_ringing", "event_time": 2.0, "event_duration": 3.0}
            ]
        },
        "2": {
            "number": 10,
            "events": [
                {"event_class": "Blender", "event_time": 1.0, "event_duration": 4.0}
            ]
        },
        "3": {
            "number": 8,
            "events": [
                {"event_class": "Cat", "event_time": 2.0, "event_duration": 2.0},
                {"event_class": "Dishes", "event_time": 5.0, "event_duration": 3.0}
            ]
        }
    }
    
    config_file = "default_event_occurences.json"
    with open(config_file, 'w') as f:
        json.dump(default_config, f, indent=2)
    
    print(f"默认事件共现配置已创建: {config_file}")
    return config_file


def main(config_path="dcase_snr_test_config.json"):
    """主函数：生成所有SNR组的数据集"""
    
    print("=== DCASE SNR分级数据集生成器 (直接scaper版本) ===")
    print("此版本直接使用scaper，兼容Python 3.12+\n")
    
    # 加载配置
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        return
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    paths = config["paths"]
    snr_groups = config["snr_groups"] 
    gen_params = config["generation_params"]
    
    # 加载音频参数（如果配置文件中有）
    audio_params = config.get("audio_params", {
        "samplerate": 8000,
        "duration": 10.0,
        "ref_db": -50
    })
    
    print(f"\n音频参数配置:")
    print(f"  采样率: {audio_params['samplerate']} Hz")
    print(f"  音频时长: {audio_params['duration']} 秒")
    print(f"  参考分贝: {audio_params['ref_db']} dB")
    if audio_params.get('description'):
        print(f"  说明: {audio_params['description']}")
    
    # 检查事件共现配置文件
    if not os.path.exists(paths["event_occurences_file"]):
        print(f"事件共现配置文件不存在: {paths['event_occurences_file']}")
        print("创建默认配置...")
        paths["event_occurences_file"] = create_default_event_occurences()

    # 解析路径配置（按split分组，防止数据泄露）
    if 'train' not in paths or not isinstance(paths['train'], dict):
        print("错误: 配置文件格式不正确，需要按split分组的路径配置")
        print("正确格式示例:")
        print('  "paths": {')
        print('    "train": {"fg_folder": "...", "bg_folder": "..."},')
        print('    "val": {"fg_folder": "...", "bg_folder": "..."},')
        print('    "test": {"fg_folder": "...", "bg_folder": "..."}')
        print('  }')
        return

    split_paths = {
        'train': paths.get('train', {}),
        'val': paths.get('val', {}),
        'test': paths.get('test', {})
    }
    # 使用train路径作为默认路径（用于初始化）
    default_fg_folder = paths['train']['fg_folder']
    default_bg_folder = paths['train']['bg_folder']

    print("\nSoundbank路径配置（按split分组，防止数据泄露）:")
    for split_name, split_config in split_paths.items():
        if split_config:
            print(f"  {split_name}: {split_config.get('fg_folder', 'N/A')}")

    # 验证路径
    required_paths = []
    for split_name, split_config in split_paths.items():
        if split_config:
            required_paths.append(split_config.get('fg_folder'))
            required_paths.append(split_config.get('bg_folder'))
    required_paths = [p for p in required_paths if p]  # 过滤None
    missing_paths = [p for p in required_paths if not os.path.exists(p)]

    if missing_paths:
        print("错误：以下必需路径不存在:")
        for path in missing_paths:
            print(f"  - {path}")
        print("\n请先下载soundbank数据")
        return
    
    # 计算总样本数
    total_train = sum(group.get("train_samples", 0) for group in snr_groups.values())
    total_val = sum(group.get("validation_samples", 0) for group in snr_groups.values())
    total_test = sum(group.get("test_samples", 0) for group in snr_groups.values())
    
    print(f"总训练样本: {total_train}")
    print(f"总验证样本: {total_val}")
    print(f"总测试样本: {total_test}")
    print(f"总计: {total_train + total_val + total_test}")
    print(f"输出目录: {paths['output_base']}")
    
    # 为每个SNR组生成数据
    all_stats = {}
    
    for snr_group, group_config in snr_groups.items():
        print(f"\n{'='*60}")
        print(f"开始生成 {snr_group} SNR组")
        print(f"SNR范围: {group_config['range'][0]} ~ {group_config['range'][1]} dB")
        print(f"训练样本: {group_config['train_samples']}, 验证样本: {group_config['validation_samples']}")
        print(f"{'='*60}")
        
        # 读取复音度权重（如果配置中有）
        polyphony_weights = gen_params.get('polyphony_weights', None)
        # 读取混响范围（如果配置中有）
        reverb_range = gen_params.get('reverb_range', None)

        generator = DirectSoundscapeGenerator(
            fg_folder=default_fg_folder,
            bg_folder=default_bg_folder,
            output_base=paths["output_base"],
            event_occurences_file=paths["event_occurences_file"],
            snr_group=snr_group,
            snr_range=tuple(group_config['range']),  # 从配置文件传入SNR范围
            samplerate=audio_params['samplerate'],   # 传入采样率
            duration=audio_params['duration'],       # 传入时长
            ref_db=audio_params['ref_db'],           # 传入参考分贝
            polyphony_weights=polyphony_weights,     # 传入复音度权重
            reverb_range=reverb_range,               # 传入混响范围
            split_paths=split_paths                  # 按split分组的soundbank路径
        )
        
        try:
            # 生成数据集
            generator.generate_dataset(
                num_train=group_config.get("train_samples", 0),
                num_val=group_config.get("validation_samples", 0),
                num_test=group_config.get("test_samples", 0),
                remove_high_polyphony=gen_params["remove_high_polyphony"],
                max_polyphony=gen_params["max_polyphony"]
            )
            
            # 验证SNR分布
            print(f"\n验证{snr_group}组的SNR分布...")
            snr_values = generator.validate_snr_distribution(
                num_samples=gen_params["validation_sample_size"]
            )
            
            all_stats[snr_group] = {
                'generated': True,
                'train_samples': group_config.get("train_samples", 0),
                'validation_samples': group_config.get("validation_samples", 0),
                'test_samples': group_config.get("test_samples", 0),
                'snr_validation_count': len(snr_values)
            }
            
            print(f"\n{snr_group} SNR组生成完成！")
            
        except Exception as e:
            print(f"\n生成{snr_group} SNR组时出错: {e}")
            all_stats[snr_group] = {
                'generated': False,
                'error': str(e)
            }
            continue
    
    # 生成总体报告
    report_path = os.path.join(paths["output_base"], "generation_report_scaper.json")
    report = {
        'version': 'direct_scaper',
        'config': config,
        'generation_stats': all_stats,
        'total_samples': {
            'train': total_train,
            'val': total_val,
            'test': total_test,
            'total': total_train + total_val + total_test
        }
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("数据集生成完成！")
    print(f"生成报告已保存至: {report_path}")
    
    # 输出统计
    successful_groups = [group for group, stats in all_stats.items() if stats.get('generated', False)]
    failed_groups = [group for group, stats in all_stats.items() if not stats.get('generated', False)]
    
    if successful_groups:
        print(f"\n成功生成的SNR组: {', '.join(successful_groups)}")
    if failed_groups:
        print(f"失败的SNR组: {', '.join(failed_groups)}")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    import sys
    
    # 支持命令行参数指定配置文件
    config_file = sys.argv[1] if len(sys.argv) > 1 else "Data\dcase_synthetic_10k\dcase_snr_desed_standard.json"
    main(config_file)
