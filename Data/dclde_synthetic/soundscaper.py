import scaper
import os
import json
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple
import jams


class AudioAnalyzer:
    """音频分析器类，用于分析生成的音频特征"""
    
    def __init__(self, metadata_path: str):
        """初始化音频分析器
        
        Args:
            metadata_path: metadata文件夹路径
        """
        self.metadata_path = metadata_path
        self.reference_stats = self._load_reference_stats()
        
        # SNR范围定义
        self.snr_ranges = {
            'very_low': (-10, -5),
            'low': (-5, 0),
            'medium': (0, 5),
            'high': (5, 10)
        }
        
    def _load_reference_stats(self) -> Dict:
        """加载参考数据集的统计特征"""
        stats = {}
        # 加载前景音频统计特征
        for audio_type in ['upcall', 'gunshot', 'scream', 'moancall']:
            with open(os.path.join(self.metadata_path, f'{audio_type}_analysis.json'), 'r') as f:
                stats[audio_type] = json.load(f)
        
        # 加载背景音频统计特征
        with open(os.path.join(self.metadata_path, 'background_analysis.json'), 'r') as f:
            stats['background'] = json.load(f)
            
        return stats
    
    def analyze_audio(self, audio_path: str) -> Dict:
        """分析单个音频文件的特征
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            音频特征字典
        """
        y, sr = librosa.load(audio_path, sr=None)
        
        # 计算基本特征
        duration = librosa.get_duration(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)[0]
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        
        return {
            'duration': float(duration),
            'rms_mean': float(np.mean(rms)),
            'rms_std': float(np.std(rms)),
            'spectral_centroid_mean': float(np.mean(spectral_centroid)),
            'spectral_centroid_std': float(np.std(spectral_centroid))
        }
    
    def analyze_snr_from_jams(self, jams_file: str) -> Dict:
        """从JAMS文件中分析SNR值
        
        Args:
            jams_file: JAMS文件路径
            
        Returns:
            SNR统计信息
        """
        jam = jams.load(jams_file)
        snr_values = []
        
        for ann in jam.annotations:
            if ann.namespace == 'scaper':
                for obs in ann.data:
                    if obs.value['role'] == 'foreground':
                        snr_values.append(float(obs.value['snr']))
        
        if not snr_values:
            return None
            
        return {
            'mean': float(np.mean(snr_values)),
            'std': float(np.std(snr_values)),
            'min': float(np.min(snr_values)),
            'max': float(np.max(snr_values)),
            'values': snr_values
        }
    
    def validate_snr(self, jams_file: str, snr_group: str) -> Tuple[bool, Dict]:
        """验证JAMS文件中的SNR值是否在指定范围内
        
        Args:
            jams_file: JAMS文件路径
            snr_group: SNR分组名称
            
        Returns:
            (是否有效, SNR统计信息)
        """
        snr_stats = self.analyze_snr_from_jams(jams_file)
        if not snr_stats:
            return False, {'error': 'No SNR values found'}
            
        snr_range = self.snr_ranges[snr_group]
        is_valid = all(snr_range[0] <= snr <= snr_range[1] for snr in snr_stats['values'])
        
        return is_valid, snr_stats


class SoundscapeGenerator:
    def __init__(self, foreground_path, background_path, output_path, metadata_path, mode='train', snr_group='all'):
        """初始化声景生成器
        
        Args:
            foreground_path: 前景音频路径
            background_path: 背景音频路径
            output_path: 输出路径
            metadata_path: metadata文件夹路径
            mode: 模式('train' or 'val')
            snr_group: SNR分组('very_low', 'low', 'medium', 'high', 'all')
        """
        self.foreground_folder = foreground_path
        self.background_folder = background_path
        self.output_base = output_path
        self.metadata_path = metadata_path
        self.mode = mode
        self.snr_group = snr_group
        
        # 创建音频分析器
        self.analyzer = AudioAnalyzer(metadata_path)
        
        # SNR分组配置
        self.snr_ranges = {
            'very_low': (-10, -5),  # 极低SNR：-10 ~ -5 dB
            'low': (-5, 0),         # 低SNR：-5 ~ 0 dB
            'medium': (0, 5),       # 中SNR：0 ~ 5 dB
            'high': (5, 10)         # 高SNR：5 ~ 10 dB
        }
        
        # 基本配置
        self.soundscape_duration = 10.0
        self.negative_sample_ratio = 0
        self.target_sr = 8000
        
        # 声音事件概率配置
        self.call_type_probabilities = {
            'upcall': [1, 1, 2],    # 100%几率出现1到2次
            'gunshot': [1, 1, 2],   # 100%几率出现1到2次
            'scream': [0.85, 1, 2],    # 100%几率出现1次
            'moancall': [0.75, 1, 1]   # 100%几率出现1次
        }

        # 初始化
        self._setup_folders()
        self._load_audio_files()

    def _setup_folders(self):
        """创建必要的输出文件夹结构"""
        # 如果指定了SNR分组，在输出路径中添加分组标识
        if self.snr_group != 'all':
            self.output_base = os.path.join(self.output_base, f'snr_{self.snr_group}')
            
        self.folders = {
            'wav': os.path.join(self.output_base, self.mode, 'wav'),
            'jams': os.path.join(self.output_base, self.mode, 'jams'),
            'txt': os.path.join(self.output_base, self.mode, 'txt')
        }
        for folder in self.folders.values():
            os.makedirs(folder, exist_ok=True)

    def _load_audio_files(self):
        """加载音频文件列表"""
        # 获取所有声音类型
        self.call_types = [d for d in os.listdir(self.foreground_folder)
                          if os.path.isdir(os.path.join(self.foreground_folder, d))]

        # 获取每种声音类型的文件列表
        self.call_files = {}
        for call_type in self.call_types:
            self.call_files[call_type] = [
                os.path.join(self.foreground_folder, call_type, f)
                for f in os.listdir(os.path.join(self.foreground_folder, call_type))
                if f.endswith('.wav')
            ]

    def _get_snr_range(self):
        """获取当前SNR分组的范围"""
        if self.snr_group == 'all':
            return -10, 10  # 默认范围
        return self.snr_ranges[self.snr_group]

    def _add_sound_event(self, sc, call_type, call_files):
        """添加单个声音事件"""
        file_path = np.random.choice(call_files)
        duration = librosa.get_duration(path=file_path)
        stretch_factor = np.random.uniform(0.8, 1.2)
        adjusted_duration = min(duration * stretch_factor, self.soundscape_duration)
        latest_start_time = self.soundscape_duration - adjusted_duration
        
        # 获取当前SNR分组的范围
        snr_min, snr_max = self._get_snr_range()

        sc.add_event(
            label=('const', call_type),
            source_file=('const', file_path),
            source_time=('const', 0),
            event_time=('uniform', 0, latest_start_time),
            event_duration=('const', duration),
            snr=('uniform', snr_min, snr_max),  # 使用指定的SNR范围
            pitch_shift=('uniform', -2, 2),
            time_stretch=('const', stretch_factor)
        )

    def _create_soundscape(self, sc, is_negative_sample=False):
        """创建单个音景"""
        # 添加背景声
        sc.add_background(
            label=('choose', []),
            source_file=('choose', []),
            source_time=('const', 0)
        )
        
        # 如果是负样本，直接返回
        if is_negative_sample:
            return
        
        # 生成声音事件
        for call_type in self.call_types:
            prob, min_events, max_events = self.call_type_probabilities[call_type]
            if np.random.rand() <= prob:
                num_events = np.random.randint(min_events, max_events + 1)
                for _ in range(num_events):
                    self._add_sound_event(sc, call_type, self.call_files[call_type])

    def _resample_audio(self, input_file, output_file):
        """重采样音频文件"""
        y, _ = librosa.load(input_file, sr=self.target_sr)
        sf.write(output_file, y, self.target_sr)

    def generate(self, num_soundscapes):
        """生成指定数量的合成音频"""
        num_negative_samples = int(num_soundscapes * self.negative_sample_ratio)
        
        for i in range(num_soundscapes):
            print(f'生成{self.mode}合成音频 (SNR组: {self.snr_group}): {i + 1}/{num_soundscapes}')
            
            # 创建scaper对象
            sc = scaper.Scaper(
                self.soundscape_duration, 
                self.foreground_folder, 
                self.background_folder
            )
            sc.protected_labels = []
            sc.ref_db = -50
            
            # 决定是否生成负样本
            is_negative = i < num_negative_samples
            
            # 创建音景
            self._create_soundscape(sc, is_negative)
            
            # 设置输出文件路径
            base_filename = f"soundscape_whale_{self.mode}_{i}"
            audiofile = os.path.join(self.folders['wav'], f"{base_filename}.wav")
            jamsfile = os.path.join(self.folders['jams'], f"{base_filename}.jams")
            txtfile = os.path.join(self.folders['txt'], f"{base_filename}.txt")
            
            # 生成音频文件
            sc.generate(
                audiofile, 
                jamsfile,
                allow_repeated_label=True,
                allow_repeated_source=True,
                reverb=0.1,
                disable_sox_warnings=True,
                no_audio=False,
                txt_path=txtfile
            )
            
            # 重采样
            self._resample_audio(audiofile, audiofile)

    def generate_and_validate(self, num_soundscapes: int, validation_size: int = 5) -> bool:
        """生成并验证合成音频数据集
        
        Args:
            num_soundscapes: 要生成的总音频数量
            validation_size: 用于验证的小批量数据集大小
            
        Returns:
            是否验证通过
        """
        print(f"\n=== 开始生成{self.snr_group} SNR组的验证数据集 ===")
        # 先生成小批量数据进行验证
        self.generate(validation_size)
        
        # 验证生成的数据
        validation_stats = {
            'total_files': 0,
            'valid_files': 0,
            'snr_stats': {
                'values': [],
                'out_of_range': []
            }
        }
        
        # 验证JAMS文件中的SNR值
        jams_folder = os.path.join(self.folders['jams'])
        for jams_file in os.listdir(jams_folder):
            if not jams_file.endswith('.jams'):
                continue
            
            validation_stats['total_files'] += 1
            jams_path = os.path.join(jams_folder, jams_file)
            
            is_valid, snr_stats = self.analyzer.validate_snr(jams_path, self.snr_group)
            validation_stats['snr_stats']['values'].extend(snr_stats.get('values', []))
            
            if is_valid:
                validation_stats['valid_files'] += 1
            else:
                validation_stats['snr_stats']['out_of_range'].append({
                    'file': jams_file,
                    'snr_stats': snr_stats
                })
        
        # 计算总体统计信息
        if validation_stats['snr_stats']['values']:
            validation_stats['snr_stats'].update({
                'mean': float(np.mean(validation_stats['snr_stats']['values'])),
                'std': float(np.std(validation_stats['snr_stats']['values'])),
                'min': float(np.min(validation_stats['snr_stats']['values'])),
                'max': float(np.max(validation_stats['snr_stats']['values']))
            })
        
        # 输出验证结果
        print("\nSNR验证结果:")
        print(f"总文件数: {validation_stats['total_files']}")
        print(f"有效文件数: {validation_stats['valid_files']}")
        print(f"有效率: {validation_stats['valid_files']/validation_stats['total_files']*100:.2f}%")
        
        if validation_stats['snr_stats'].get('mean'):
            print("\nSNR统计:")
            print(f"  - 平均值: {validation_stats['snr_stats']['mean']:.2f} dB")
            print(f"  - 标准差: {validation_stats['snr_stats']['std']:.2f} dB")
            print(f"  - 最小值: {validation_stats['snr_stats']['min']:.2f} dB")
            print(f"  - 最大值: {validation_stats['snr_stats']['max']:.2f} dB")
        
        if validation_stats['snr_stats']['out_of_range']:
            print("\n超出范围的文件:")
            for item in validation_stats['snr_stats']['out_of_range']:
                print(f"  - {item['file']}")
        
        # 判断是否验证通过
        is_valid = (validation_stats['valid_files'] == validation_stats['total_files'])
        
        if not is_valid:
            print(f"\n验证失败！{self.snr_group} SNR组的数据集不符合要求")
            return False
            
        print(f"\n验证通过！继续生成剩余的{num_soundscapes - validation_size}个音频文件")
        
        # 生成剩余的数据
        if num_soundscapes > validation_size:
            self.generate(num_soundscapes - validation_size)
        
        # 保存统计报告
        report_path = os.path.join(self.output_base, f'stats_{self.snr_group}.json')
        with open(report_path, 'w') as f:
            json.dump(validation_stats, f, indent=2)
            
        print(f"\n数据集生成完成！统计报告已保存到: {report_path}")
        return True


def main():
    # 设置路径
    foreground_path = 'scaper_k_fold/test/foreground'
    background_path = 'scaper_k_fold/test/background'
    output_base = 'scaper_k_fold/test'
    metadata_path = '../metadata'
    
    # 设置各SNR组的生成数量（总数1500，比例2:3:3:2）
    snr_group_counts = {
        'very_low': 320,  # 2
        'low': 470,       # 3
        'medium': 470,    # 3
        'high': 320       # 2
    }
    
    # 验证样本数（每组使用20个样本进行验证）
    validation_size = 20
    
    print(f"\n=== 开始生成验证集数据 ===")
    print(f"总样本数: 1500")
    print("各SNR组样本数:")
    for group, count in snr_group_counts.items():
        print(f"  - {group}: {count}")
    print("\n")
    
    # 为每个SNR分组生成数据
    for snr_group, num_samples in snr_group_counts.items():
        print(f"\n{'='*50}")
        print(f"开始生成 {snr_group} SNR组的数据 (目标数量: {num_samples})")
        print(f"{'='*50}")
        
        generator = SoundscapeGenerator(
            foreground_path=foreground_path,
            background_path=background_path,
            output_path=output_base,
            metadata_path=metadata_path,
            mode='test',
            snr_group=snr_group
        )
        
        # 生成并验证数据集
        if generator.generate_and_validate(num_soundscapes=num_samples, validation_size=validation_size):
            print(f"\n{snr_group} SNR组的数据集生成成功！")
            print(f"{'='*50}\n")
        else:
            print(f"\n{snr_group} SNR组的数据集生成失败，请检查参数配置。")
            print(f"{'='*50}\n")
            break


if __name__ == "__main__":
    main()
