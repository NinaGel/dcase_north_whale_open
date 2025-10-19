import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys

# 获取项目根目录
ROOT_DIR = Path(__file__).parent.parent

# 添加项目根目录到 Python 路径
sys.path.append(str(ROOT_DIR))

import config as cfg
from .augmentation.audio_augmentor import AudioAugmentor


class WhaleDataset(Dataset):
    """鲸鱼声音数据集类"""

    def __init__(self, features, labels, filenames, augment=False):
        # 保持原始数据维度，使用float32存储原始数据
        self.features = torch.tensor(features, dtype=torch.float32)  # [N, freq, time]
        self.labels = torch.tensor(labels, dtype=torch.float32)  # [N, num_classes]
        self.filenames = filenames
        self.augment = augment

        if self.augment:
            self.augmentor = AudioAugmentor()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]  # [freq, time]
        label = self.labels[idx]  # [num_classes]
        filename = self.filenames[idx]

        # 确保特征维度正确 [channel, freq, time]
        if feature.dim() == 2:  # [freq, time]
            feature = feature.unsqueeze(0)  # [1, freq, time]

        # 应用数据增强
        if self.augment:
            feature = feature.float()  # 确保是float32用于数据增强
            feature, label = self.augmentor.augment(feature, label)

        # 转换为float16用于训练
        if cfg.GPU_CONFIG['precision'] == 'float16':
            feature = feature.half()

        return feature, label, filename


def create_data_loader(features, labels, filenames, batch_size, augment=False, shuffle=True):
    """创建数据加载器

    Args:
        features: shape [N, freq, time]
        labels: shape [N, num_classes]
        filenames: list of str
        batch_size: int
        augment: bool, 是否使用数据增强
        shuffle: bool, 是否打乱数据
    """
    dataset = WhaleDataset(features, labels, filenames, augment=augment)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.GPU_CONFIG['num_workers'],
        pin_memory=cfg.GPU_CONFIG['pin_memory'],
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=shuffle  # 训练时drop_last=True，评估时False
    )


def load_snr_data(mode='train', batch_size=None):
    """加载不同SNR级别的数据

    Args:
        mode: str, 'train', 'val' 或 'test'
        batch_size: int, optional

    Returns:
        如果是训练模式:
            DataLoader: 包含所有SNR级别数据的加载器
        如果是验证或测试模式:
            dict: 每个SNR组对应一个数据加载器
    """
    batch_size = batch_size or cfg.TRAIN_CONFIG['batch_size']
    base_path = cfg.PATH_CONFIG['snr_data_path']
    mode_path = base_path / mode

    # SNR组和对应的文件名后缀
    snr_groups = {
        'snr_high': 'high',
        'snr_medium': 'medium',
        'snr_low': 'low',
        'snr_very_low': 'very_low'
    }

    if mode == 'train':
        # 训练模式：合并所有数据
        features_list, labels_list, filenames_list = [], [], []

        for snr_group, suffix in snr_groups.items():
            data_path = mode_path / snr_group / f'act_whale_dataset_{mode}_{suffix}.npz'
            if data_path.exists():
                data = np.load(str(data_path))
                features_list.append(data['features'])  # [N, freq, time]
                labels_list.append(data['labels'])  # [N, num_classes]
                filenames_list.append(data['filenames'])

        if not features_list:
            raise ValueError(f"未在 {mode_path} 找到数据。请确保数据目录结构正确：\n"
                             f"期望的目录结构：\n"
                             f"{base_path}/train/snr_high/act_whale_dataset_train_high.npz\n"
                             f"{base_path}/train/snr_medium/act_whale_dataset_train_medium.npz\n"
                             f"{base_path}/train/snr_low/act_whale_dataset_train_low.npz\n"
                             f"{base_path}/train/snr_very_low/act_whale_dataset_train_very_low.npz")

        # 合并数据，保持维度不变
        features = np.concatenate(features_list, axis=0)  # [N_total, freq, time]
        labels = np.concatenate(labels_list, axis=0)  # [N_total, num_classes]
        filenames = np.concatenate(filenames_list)

        return create_data_loader(features, labels, filenames, batch_size, augment=True)

    else:
        # 验证/测试模式：为每个SNR组创建单独的加载器
        loaders = {}
        for snr_group, suffix in snr_groups.items():
            data_path = mode_path / snr_group / f'act_whale_dataset_{mode}_{suffix}.npz'
            if data_path.exists():
                data = np.load(str(data_path))
                loaders[snr_group] = create_data_loader(
                    data['features'],
                    data['labels'],
                    data['filenames'],
                    batch_size,
                    augment=False,
                    shuffle=False
                )

        if not loaders:
            raise ValueError(f"未在 {mode_path} 找到数据。请确保数据目录结构正确：\n"
                             f"期望的目录结构：\n"
                             f"{base_path}/{mode}/snr_high/act_whale_dataset_{mode}_high.npz\n"
                             f"{base_path}/{mode}/snr_medium/act_whale_dataset_{mode}_medium.npz\n"
                             f"{base_path}/{mode}/snr_low/act_whale_dataset_{mode}_low.npz\n"
                             f"{base_path}/{mode}/snr_very_low/act_whale_dataset_{mode}_very_low.npz")

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
