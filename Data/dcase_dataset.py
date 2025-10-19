import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys

# 获取项目根目录
ROOT_DIR = Path(__file__).parent.parent

# 添加项目根目录到 Python 路径
sys.path.append(str(ROOT_DIR))

import config_dcase as cfg
from .augmentation.dcase_audio_augmentor import DCASEAudioAugmentor


class DCASEDataset(Dataset):
    """DCASE数据集类（ACT特征版本）"""
    
    def __init__(self, features, labels, filenames, augment=False):
        # 保持原始数据维度，使用float32存储原始数据
        self.features = torch.tensor(features, dtype=torch.float32)  # [N, freq_bins, act_frame]
        self.labels = torch.tensor(labels, dtype=torch.float32)      # [N, act_frame, num_classes]
        self.filenames = filenames
        self.augment = augment
        
        if self.augment:
            self.augmentor = DCASEAudioAugmentor()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]  # [freq_bins, act_frame]
        label = self.labels[idx]      # [act_frame, num_classes]
        filename = self.filenames[idx]

        # 确保特征维度正确 [channel, freq_bins, act_frame]
        if feature.dim() == 2:  # [freq_bins, act_frame]
            feature = feature.unsqueeze(0)  # [1, freq_bins, act_frame]

        # 应用数据增强
        if self.augment:
            feature = feature.float()  # 确保是float32用于数据增强
            feature, label = self.augmentor.augment(feature, label)
        else:
            # 确保返回float32（不要在数据集层面转换为float16）
            feature = feature.float()

        return feature, label, filename


def create_dcase_data_loader(features, labels, filenames, batch_size, augment=False, shuffle=True):
    """创建DCASE数据加载器
    
    Args:
        features: shape [N, freq_bins, act_frame]
        labels: shape [N, act_frame, num_classes]
        filenames: list of str
        batch_size: int
        augment: bool, 是否使用数据增强
        shuffle: bool, 是否打乱数据
    """
    dataset = DCASEDataset(features, labels, filenames, augment=augment)

    num_workers = cfg.DCASE_GPU_CONFIG['num_workers']
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=cfg.DCASE_GPU_CONFIG['pin_memory'],
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=shuffle  # 训练时drop_last=True，评估时False
    )


def load_snr_data(mode='train', batch_size=None):
    """加载不同SNR级别的DCASE数据

    Args:
        mode: str, 'train', 'val' 或 'test'
        batch_size: int, optional

    Returns:
        如果是训练模式:
            DataLoader: 包含所有SNR级别数据的加载器
        如果是验证或测试模式:
            dict: 每个SNR组对应一个数据加载器
    """
    batch_size = batch_size or cfg.DCASE_TRAIN_CONFIG['batch_size']

    # 从配置文件读取数据集路径
    base_path = cfg.DCASE_PATH_CONFIG['snr_data_path']

    # 自动检测可用的 SNR 级别
    # 支持两种数据集格式：
    # 1. dcase_snr_10k_8000HZ: snr_very_low, snr_low, snr_medium, snr_high
    # 2. dcase_snr_desed_standard: snr_low, snr_medium, snr_high
    available_snr_groups = []
    for snr_dir in ['snr_very_low', 'snr_low', 'snr_medium', 'snr_high']:
        if (base_path / snr_dir).exists():
            available_snr_groups.append(snr_dir)

    # 构建 SNR 组和文件名后缀的映射
    snr_groups = {snr: snr.replace('snr_', '') for snr in available_snr_groups}

    if mode == 'train':
        # 训练模式：合并所有数据
        features_list, labels_list, filenames_list = [], [], []
        
        for snr_group, suffix in snr_groups.items():
            # 构建数据路径：dcase_snr_10k_8000HZ/snr_high/train/act_dcase_train_high.npz
            data_path = base_path / snr_group / mode / f'act_dcase_{mode}_{suffix}.npz'
            if data_path.exists():
                data = np.load(str(data_path))
                features_list.append(data['features'])  # [N, freq_bins, act_frame]
                labels_list.append(data['labels'])      # [N, act_frame, num_classes]
                filenames_list.append(data['filenames'])

        if not features_list:
            expected_paths = [f"{base_path}/{snr}/{mode}/act_dcase_{mode}_{suffix}.npz"
                            for snr, suffix in snr_groups.items()]
            raise ValueError(f"未在 {base_path} 找到任何数据。\n"
                           f"检测到的 SNR 级别: {list(snr_groups.keys())}\n"
                           f"期望的文件路径：\n" + "\n".join(expected_paths))

        # 合并数据，保持维度不变
        features = np.concatenate(features_list, axis=0)  # [N_total, freq_bins, act_frame]
        labels = np.concatenate(labels_list, axis=0)      # [N_total, act_frame, num_classes]
        filenames = np.concatenate(filenames_list)

        return create_dcase_data_loader(features, labels, filenames, batch_size, augment=True)
    
    else:
        # 验证/测试模式：为每个SNR组创建单独的加载器
        loaders = {}
        for snr_group, suffix in snr_groups.items():
            data_path = base_path / snr_group / mode / f'act_dcase_{mode}_{suffix}.npz'
            if data_path.exists():
                data = np.load(str(data_path))
                loaders[snr_group] = create_dcase_data_loader(
                    data['features'],
                    data['labels'],
                    data['filenames'],
                    batch_size,
                    augment=False,
                    shuffle=False
                )
        
        if not loaders:
            expected_paths = [f"{base_path}/{snr}/{mode}/act_dcase_{mode}_{suffix}.npz"
                            for snr, suffix in snr_groups.items()]
            raise ValueError(f"未在 {base_path} 找到任何数据。\n"
                           f"检测到的 SNR 级别: {list(snr_groups.keys())}\n"
                           f"期望的文件路径：\n" + "\n".join(expected_paths))
        
        return loaders


def load_test_data(batch_size=None):
    """加载测试数据"""
    return load_snr_data('test', batch_size)


def load_train_val_data(batch_size=None):
    """加载训练和验证数据

    Returns:
        tuple: (train_loader, val_loaders)
            - train_loader: DataLoader, 包含所有SNR级别的训练数据
            - val_loaders: dict, 每个SNR组对应一个验证数据加载器
    """
    train_loader = load_snr_data('train', batch_size)
    val_loaders = load_snr_data('val', batch_size)
    return train_loader, val_loaders

