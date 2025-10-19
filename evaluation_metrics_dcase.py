"""
evaluation_metrics_dcase.py - DCASE2020声音事件检测评估模块

本模块实现了DCASE2020数据集的声音事件检测(SED)评估器。
主要功能包括:

1. 预测解码:
   - 将模型输出的预测概率转换为具体的事件检测结果
   - 支持多阈值评估
   - 包含中值滤波后处理

2. 评估指标:
   - 基于片段的评估(Segment-based):
     * 微观F1分数
     * 宏观F1分数
     * 错误率
   - 基于事件的评估(Event-based):
     * 使用时间容限的F1分数
     * 支持onset/offset评估
   - PSDS(Polyphonic Sound Detection Score):
     * 基于交叉检测的评分
     * 考虑检测容限和替代成本
   - pAUC(部分曲线下面积):
     * 在低假阳性率区间的性能评估
   - 最优F1:
     * 每个类别的最优阈值F1分数

使用方法:
    evaluator = DCASEEventDetectionEvaluator()
    metrics = evaluator.compute_all_metrics(
        strong_preds,  # 模型预测输出 [batch_size, time_steps, num_classes]
        filenames,     # 音频文件名列表
        ground_truth_df,  # 真实标签DataFrame
        audio_durations   # 音频持续时间字典
    )

依赖库:
    - numpy: 数值计算
    - pandas: 数据处理
    - sed_eval: 声音事件检测评估
    - sed_scores_eval: 评分计算
    - scipy: 信号处理
    - torch: 深度学习支持

配置:
    - 采样率(sample_rate): 音频采样率,默认8000Hz
    - 跳跃长度(hop_length): STFT帧移,默认256
    - 标签列表(labels): 事件类别列表

注意事项:
    1. 输入预测需要是[batch_size, time_steps, num_classes]格式的张量
    2. 真实标签需要包含onset/offset时间和事件类别
    3. 评估时使用1秒的时间分辨率
    4. PSDS评估使用DCASE任务标准参数设置
"""

import os
import numpy as np
import pandas as pd
import torch
import sed_eval
import sed_scores_eval
from pathlib import Path
import scipy.ndimage as ndimage
from sed_scores_eval.base_modules.scores import create_score_dataframe
import config_dcase as cfg


class DCASEEventDetectionEvaluator:
    """DCASE2020声音事件检测评估器

    该类实现了一个完整的声音事件检测评估系统,包含多种评估指标的计算。

    属性:
        sample_rate (int): 音频采样率,默认8000Hz
        hop_length (int): STFT帧移,默认256
        labels (list): 事件类别标签列表

    数据格式要求:
    1. 模型输出格式:
       - strong_preds: [batch_size, time_steps, num_classes] 的预测张量
       - 值域为[0,1]的概率值

    2. 标签格式:
       - ground_truth_dict: 字典格式
         {audio_id: [(onset,offset,event_label),...]}
       - ground_truth_df: DataFrame格式
         columns=[filename,event_label,onset,offset]

    3. 音频信息:
       - audio_durations: 字典格式
         {audio_id: duration_in_seconds}
       - filenames: 列表格式
         [audio_file1.wav, audio_file2.wav,...]
    """

    def __init__(self, sample_rate=cfg.DCASE_AUDIO_CONFIG['sr'],
                 hop_length=cfg.DCASE_AUDIO_CONFIG['hop_length'],
                 labels=cfg.DCASE_MODEL_CONFIG['class_names']):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.labels = labels

    def _frame_to_time(self, frame_idx):
        """将帧索引转换为时间(秒)

        将模型输出的帧索引转换为实际的时间戳。

        Args:
            frame_idx (int): 帧索引

        Returns:
            float: 对应的时间戳(秒)
        """
        return frame_idx * self.hop_length / self.sample_rate

    def decode_predictions(self, strong_preds, filenames, thresholds=[0.5], median_filter=7):
        """解码模型预测输出

        将模型的原始预测输出转换为可评估的格式,包括:
        1. 将预测概率转换为二值预测
        2. 应用中值滤波进行后处理
        3. 提取连续预测区间作为事件

        Args:
            strong_preds (torch.Tensor): [batch_size, time_steps, num_classes]格式的预测张量
            filenames (list): 音频文件名列表
            thresholds (list): 评估阈值列表,默认[0.5]
            median_filter (int): 中值滤波器窗口大小,默认7

        Returns:
            tuple: 包含三个元素:
                - scores_raw (dict): 原始预测分数,键为音频ID,值为DataFrame
                - scores_postprocessed (dict): 后处理分数,键为音频ID,值为DataFrame
                - prediction_dfs (dict): 不同阈值的预测结果,键为阈值,值为DataFrame
        """
        scores_raw = {}
        scores_postprocessed = {}
        # 使用列表收集每个阈值的预测结果，避免频繁concat空DataFrame
        prediction_lists = {th: [] for th in thresholds}

        # 遍历每个样本
        for i in range(strong_preds.shape[0]):
            audio_id = Path(filenames[i]).stem
            filename = f"{audio_id}.wav"

            # 获取当前样本的预测分数，并转换为float32类型
            scores = strong_preds[i].cpu().float().numpy()  # [time_steps, num_classes]

            # 存储原始分数
            scores_raw[audio_id] = create_score_dataframe(
                scores=scores,
                timestamps=self._frame_to_time(np.arange(len(scores) + 1)),
                event_classes=self.labels
            )

            # 应用中值滤波
            if median_filter:
                # 确保数据类型为float32
                scores = scores.astype(np.float32)
                scores = ndimage.median_filter(scores, size=(median_filter, 1))

            # 存储后处理分数
            scores_postprocessed[audio_id] = create_score_dataframe(
                scores=scores,
                timestamps=self._frame_to_time(np.arange(len(scores) + 1)),
                event_classes=self.labels
            )

            # 对每个阈值进行解码
            for threshold in thresholds:
                # 获取高于阈值的预测
                pred_binary = scores > threshold

                # 直接从二值预测矩阵中提取事件
                events = []
                for class_idx in range(pred_binary.shape[1]):
                    # 找到连续预测为1的区间
                    changes = np.diff(pred_binary[:, class_idx].astype(int), prepend=0, append=0)
                    starts = np.where(changes == 1)[0]
                    ends = np.where(changes == -1)[0]

                    for start, end in zip(starts, ends):
                        events.append({
                            'event_label': self.labels[class_idx],
                            'onset': self._frame_to_time(start),
                            'offset': self._frame_to_time(end)
                        })

                # 只有当检测到事件时才创建DataFrame并添加到列表
                if events:
                    pred_df = pd.DataFrame(events)
                    pred_df["filename"] = filename
                    prediction_lists[threshold].append(pred_df)

        # 一次性合并所有非空的预测结果
        prediction_dfs = {}
        for threshold in thresholds:
            if prediction_lists[threshold]:
                prediction_dfs[threshold] = pd.concat(prediction_lists[threshold], ignore_index=True)
            else:
                # 如果所有样本都没有检测到事件，创建空DataFrame
                prediction_dfs[threshold] = pd.DataFrame(columns=["event_label", "onset", "offset", "filename"])

        return scores_raw, scores_postprocessed, prediction_dfs

    def compute_segment_metrics(self, predictions_df, ground_truth_df, time_resolution=1.0):
        """计算基于片段的评估指标

        使用固定时间分辨率(默认1秒)将音频分段,计算片段级别的评估指标。

        Args:
            predictions_df (pd.DataFrame): 预测结果DataFrame,包含columns: [filename, event_label, onset, offset]
            ground_truth_df (pd.DataFrame): 真实标签DataFrame,格式同predictions_df
            time_resolution (float): 时间分辨率,默认1.0秒

        Returns:
            dict: 包含以下指标:
                - segment_based_f1_micro: 微观F1分数
                - segment_based_er_micro: 微观错误率
                - segment_based_f1_macro: 宏观F1分数
                - class_wise: 每个类别的详细评估指标
        """
        all_classes = sorted(list(set(ground_truth_df.event_label.unique()) |
                                set(predictions_df.event_label.unique())))

        segment_metrics = sed_eval.sound_event.SegmentBasedMetrics(
            event_label_list=all_classes,
            time_resolution=time_resolution
        )

        for filename in ground_truth_df.filename.unique():
            gt_file = ground_truth_df[ground_truth_df.filename == filename]
            pred_file = predictions_df[predictions_df.filename == filename]

            segment_metrics.evaluate(
                reference_event_list=gt_file.to_dict('records'),
                estimated_event_list=pred_file.to_dict('records')
            )

        results = segment_metrics.results()

        # 添加每个类别的指标
        class_wise_results = {}
        for event_label in all_classes:
            if event_label in results['class_wise']:
                class_metrics = results['class_wise'][event_label]
                class_wise_results[event_label] = {
                    'f_measure': class_metrics['f_measure'].get('f_measure', float('nan')),
                    'precision': class_metrics['f_measure'].get('precision', float('nan')),
                    'recall': class_metrics['f_measure'].get('recall', float('nan')),
                    'error_rate': class_metrics['error_rate'].get('error_rate', float('nan')),
                    'deletion_rate': class_metrics['error_rate'].get('deletion_rate', float('nan')),
                    'insertion_rate': class_metrics['error_rate'].get('insertion_rate', float('nan'))
                }
                # 只有在存在substitution_rate时才添加
                if 'substitution_rate' in class_metrics['error_rate']:
                    class_wise_results[event_label]['substitution_rate'] = class_metrics['error_rate']['substitution_rate']

        return {
            'segment_based_f1_micro': results['overall']['f_measure']['f_measure'],
            'segment_based_er_micro': results['overall']['error_rate']['error_rate'],
            'segment_based_f1_macro': results['class_wise_average']['f_measure']['f_measure'],
            'segment_based_class_wise': class_wise_results
        }

    def compute_pauc_metrics(self, scores, ground_truth, audio_durations, max_fpr=0.1):
        """计算部分AUC(pAUC)分数

        在低假阳性率区间计算ROC曲线下面积,评估模型在高精确率要求下的性能。

        Args:
            scores (dict): 预测分数字典,键为音频ID,值为DataFrame
            ground_truth (dict): 真实标签字典,键为音频ID,值为事件列表
            audio_durations (dict): 音频持续时间字典,键为音频ID,值为持续时间
            max_fpr (float): 最大假阳性率,默认0.1

        Returns:
            float: 宏观pAUC分数
        """
        # 使用segment_based.auroc计算pAUC
        return sed_scores_eval.segment_based.auroc(
            scores,
            ground_truth,
            audio_durations,
            segment_length=1.0,  # 使用1秒的片段长度
            max_fpr=max_fpr  # 设置最大假阳性率
        )[0]["mean"]  # 返回平均AUC

    def compute_optimal_f1(self, scores, ground_truth, audio_durations):
        """计算最优F1分数

        为每个类别寻找最优阈值,计算对应的宏观F1分数。

        Args:
            scores (dict): 预测分数字典,键为音频ID,值为DataFrame
            ground_truth (dict): 真实标签字典,键为音频ID,值为事件列表
            audio_durations (dict): 音频持续时间字典,键为音频ID,值为持续时间

        Returns:
            float: 最优宏观F1分数
        """
        # 使用segment_based.best_fscore计算最优F1
        return sed_scores_eval.segment_based.best_fscore(
            scores,
            ground_truth,
            audio_durations,
            segment_length=1.0  # 使用1秒的片段长度
        )[0]["macro_average"]  # 返回宏观平均F1分数

    def compute_psds_metrics(self, scores, ground_truth, audio_durations, scenario=1):
        """计算PSDS(Polyphonic Sound Detection Score)评分

        PSDS是一个综合性的评估指标,考虑了:
        1. 检测容限
        2. 跨类别触发
        3. 替代成本

        该方法支持DCASE2021 Task4的两种评估场景:
        - Scenario 1 (PSDS1): 严格的检测容限,侧重于准确的事件检测
        - Scenario 2 (PSDS2): 宽松的容限,考虑跨类别触发的情况

        Args:
            scores (dict): 预测分数字典,键为音频ID,值为DataFrame
            ground_truth (dict): 真实标签字典,键为音频ID,值为事件列表
            audio_durations (dict): 音频持续时间字典,键为音频ID,值为持续时间
            scenario (int): 评估场景,1或2,默认为1

        Returns:
            float: PSDS分数

        Raises:
            ValueError: 如果scenario不是1或2
        """
        if scenario == 1:
            # PSDS1: 严格场景
            # dtc/gtc=0.7 表示检测需要与真实事件有70%的重叠
            # cttc=None 表示不考虑跨类别触发
            # alpha_ct=0 表示跨触发不计入成本
            # alpha_st=1 表示全权重考虑时间偏移
            dtc_threshold = 0.7
            gtc_threshold = 0.7
            cttc_threshold = None
            alpha_ct = 0
            alpha_st = 1
        elif scenario == 2:
            # PSDS2: 宽松场景
            # dtc/gtc=0.1 表示只需10%的重叠即可
            # cttc=0.3 表示30%重叠视为跨类别触发
            # alpha_ct=0.5 表示跨触发占50%权重
            # alpha_st=1 表示全权重考虑时间偏移
            dtc_threshold = 0.1
            gtc_threshold = 0.1
            cttc_threshold = 0.3
            alpha_ct = 0.5
            alpha_st = 1
        else:
            raise ValueError(f"Invalid scenario: {scenario}. Must be 1 or 2.")

        # 使用intersection_based.psds计算PSDS
        return sed_scores_eval.intersection_based.psds(
            scores=scores,
            ground_truth=ground_truth,
            audio_durations=audio_durations,
            dtc_threshold=dtc_threshold,
            gtc_threshold=gtc_threshold,
            cttc_threshold=cttc_threshold,
            alpha_ct=alpha_ct,
            alpha_st=alpha_st,
            max_efpr=100,  # 最大事件假阳性率(每小时)
            num_jobs=4  # 并行作业数,与参考实现一致
        )[0]  # 返回PSDS分数

    def compute_event_based_metrics(self, predictions_df, ground_truth_df):
        """计算基于事件的评估指标

        使用时间容限评估事件检测的准确性,包括:
        1. 事件匹配使用时间容限(±200ms)
        2. 偏移时间评估使用相对容限(20%)

        Args:
            predictions_df (pd.DataFrame): 预测结果DataFrame
            ground_truth_df (pd.DataFrame): 真实标签DataFrame

        Returns:
            dict: 包含以下指标:
                - event_based_f1: 基于事件的F1分数
                - event_based_class_wise: 每个类别的详细评估指标
        """
        all_classes = sorted(list(set(ground_truth_df.event_label.unique()) |
                                set(predictions_df.event_label.unique())))

        event_metrics = sed_eval.sound_event.EventBasedMetrics(
            event_label_list=all_classes,
            t_collar=0.200,
            percentage_of_length=0.2
        )

        for filename in ground_truth_df.filename.unique():
            gt_file = ground_truth_df[ground_truth_df.filename == filename]
            pred_file = predictions_df[predictions_df.filename == filename]

            event_metrics.evaluate(
                reference_event_list=gt_file.to_dict('records'),
                estimated_event_list=pred_file.to_dict('records')
            )

        results = event_metrics.results()

        # 添加每个类别的指标
        class_wise_results = {}
        for event_label in all_classes:
            if event_label in results['class_wise']:
                class_metrics = results['class_wise'][event_label]
                class_wise_results[event_label] = {
                    'f_measure': class_metrics['f_measure'].get('f_measure', float('nan')),
                    'precision': class_metrics['f_measure'].get('precision', float('nan')),
                    'recall': class_metrics['f_measure'].get('recall', float('nan')),
                    'error_rate': class_metrics['error_rate'].get('error_rate', float('nan')),
                    'deletion_rate': class_metrics['error_rate'].get('deletion_rate', float('nan')),
                    'insertion_rate': class_metrics['error_rate'].get('insertion_rate', float('nan'))
                }
                # 只有在存在substitution_rate时才添加
                if 'substitution_rate' in class_metrics['error_rate']:
                    class_wise_results[event_label]['substitution_rate'] = class_metrics['error_rate']['substitution_rate']

        return {
            'event_based_f1': results['class_wise_average']['f_measure']['f_measure'],
            'event_based_class_wise': class_wise_results
        }

    def compute_all_metrics(self, strong_preds, filenames, ground_truth_dict, ground_truth_df, audio_durations):
        """计算所有评估指标

        整合所有评估指标的计算,包括:
        1. 预测解码和后处理
        2. 基于片段的评估
        3. 基于事件的评估
        4. PSDS评分(场景1和场景2)
        5. pAUC分数
        6. 最优F1分数

        Args:
            strong_preds (torch.Tensor): 模型预测输出
            filenames (list): 音频文件名列表
            ground_truth_dict (dict): 真实标签字典 {audio_id: [(onset,offset,event_label),...]}
            ground_truth_df (pd.DataFrame): 真实标签DataFrame
            audio_durations (dict): 音频持续时间字典 {audio_id: duration}

        Returns:
            dict: 包含所有评估指标的字典:
                - psds1_score: PSDS场景1评分(严格)
                - psds2_score: PSDS场景2评分(宽松)
                - macro_pauc: 宏观pAUC
                - optimal_macro_f1: 最优宏观F1
                - segment_based_*: 基于片段的指标
                - event_based_*: 基于事件的指标
                - class_wise_metrics: 每个声音事件类别的指标
        """
        # 解码预测
        scores_raw, scores_post, pred_dfs = self.decode_predictions(
            strong_preds, filenames
        )

        # 计算基于片段的指标
        segment_metrics = self.compute_segment_metrics(
            pred_dfs[0.5], ground_truth_df
        )

        # 计算pAUC (使用 dict)
        pauc_score = self.compute_pauc_metrics(
            scores_post, ground_truth_dict, audio_durations
        )

        # 计算最优F1 (使用 dict)
        optimal_f1 = self.compute_optimal_f1(
            scores_post, ground_truth_dict, audio_durations
        )

        # 计算PSDS1和PSDS2 (使用 dict)
        # PSDS1: 严格场景,侧重准确的事件检测
        psds1_score = self.compute_psds_metrics(
            scores_post, ground_truth_dict, audio_durations, scenario=1
        )
        # PSDS2: 宽松场景,考虑跨类别触发
        psds2_score = self.compute_psds_metrics(
            scores_post, ground_truth_dict, audio_durations, scenario=2
        )

        # 计算基于事件的指标 (使用 DataFrame)
        event_metrics = self.compute_event_based_metrics(
            pred_dfs[0.5], ground_truth_df
        )

        # 为每个声音事件类别创建单独的评估指标集合
        class_wise_metrics = {}

        # 从配置中获取类别名称
        class_names = self.labels

        for class_name in class_names:
            # 检查类别是否存在于结果中
            segment_class_metrics = segment_metrics.get('segment_based_class_wise', {}).get(class_name, {})
            event_class_metrics = event_metrics.get('event_based_class_wise', {}).get(class_name, {})

            # 组合该类别的所有指标
            class_wise_metrics[class_name] = {
                # 段级指标
                'segment_f_measure': segment_class_metrics.get('f_measure', float('nan')),
                'segment_precision': segment_class_metrics.get('precision', float('nan')),
                'segment_recall': segment_class_metrics.get('recall', float('nan')),
                'segment_error_rate': segment_class_metrics.get('error_rate', float('nan')),

                # 事件级指标
                'event_f_measure': event_class_metrics.get('f_measure', float('nan')),
                'event_precision': event_class_metrics.get('precision', float('nan')),
                'event_recall': event_class_metrics.get('recall', float('nan')),

                # 额外的错误率组件
                'deletion_rate': event_class_metrics.get('deletion_rate', float('nan')),
                'insertion_rate': event_class_metrics.get('insertion_rate', float('nan')),
                'substitution_rate': event_class_metrics.get('substitution_rate', float('nan'))
            }

        # 合并所有指标
        metrics = {
            'psds1_score': psds1_score,
            'psds2_score': psds2_score,
            'macro_pauc': pauc_score,
            'optimal_macro_f1': optimal_f1,
            **{k: v for k, v in segment_metrics.items() if k != 'segment_based_class_wise'},
            **{k: v for k, v in event_metrics.items() if k != 'event_based_class_wise'},
            'class_wise_metrics': class_wise_metrics
        }

        return metrics


def merge_overlapping_events(events, tolerance=0.001):
    """合并重叠的事件

    Args:
        events (list): [(onset, offset, event_label), ...] 格式的事件列表
        tolerance (float): 判断事件是否连接的时间容差

    Returns:
        list: 合并后的事件列表
    """
    if not events:
        return events

    # 按照开始时间排序
    events = sorted(events, key=lambda x: (x[2], x[0]))  # 先按类别，再按开始时间排序

    merged = []
    current_events = {}  # 每个类别当前正在处理的事件

    for onset, offset, label in events:
        if label not in current_events:
            current_events[label] = (onset, offset)
        else:
            curr_onset, curr_offset = current_events[label]

            # 检查是否重叠或连接
            if onset <= (curr_offset + tolerance):
                # 更新当前事件的结束时间
                current_events[label] = (curr_onset, max(curr_offset, offset))
            else:
                # 保存当前事件，开始新事件
                merged.append((curr_onset, curr_offset, label))
                current_events[label] = (onset, offset)

    # 添加剩余的事件
    for label, (onset, offset) in current_events.items():
        merged.append((onset, offset, label))

    return sorted(merged, key=lambda x: x[0])  # 按开始时间排序


def load_ground_truth_dcase(metadata_file):
    """加载DCASE2020标签文件

    将tsv格式的标签文件转换为评估所需的格式:
    1. ground_truth_df: DataFrame格式,包含[filename,event_label,onset,offset]
    2. ground_truth_dict: 字典格式,用于sed_scores_eval库
    3. audio_durations: 字典格式,记录每个音频文件的持续时间

    Args:
        metadata_file (str): 标签文件路径(.tsv)

    Returns:
        tuple: (ground_truth_df, ground_truth_dict, audio_durations)
    """
    # 读取标签文件
    metadata_file = Path(metadata_file)

    if not metadata_file.exists():
        raise ValueError(f"No ground truth data found at {metadata_file}")

    # 读取tsv文件
    df = pd.read_csv(metadata_file, sep='\t')

    # 初始化数据结构
    ground_truth_dict = {}
    audio_durations = {}

    # 按文件分组
    for filename in df['filename'].unique():
        file_data = df[df['filename'] == filename]
        audio_id = Path(filename).stem

        # 收集事件
        events = []
        max_offset = 0

        for _, row in file_data.iterrows():
            onset = float(row['onset'])
            offset = float(row['offset'])
            event_label = row['event_label']

            max_offset = max(max_offset, offset)
            events.append((onset, offset, event_label))

        if events:
            # 合并重叠事件
            merged_events = merge_overlapping_events(events)

            # 更新字典格式的数据
            ground_truth_dict[audio_id] = merged_events
            audio_durations[audio_id] = max_offset

    # DataFrame格式已经在原始df中，只需要确保格式正确
    ground_truth_df = df.copy()

    return ground_truth_df, ground_truth_dict, audio_durations
