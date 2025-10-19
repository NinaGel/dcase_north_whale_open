"""
评估工具函数
"""

import numpy as np
import torch
from scipy import stats

def time_to_frame(time, sr=22050, hop_length=512):
    """
    将时间转换为帧索引
    """
    return int(time * sr / hop_length)

def frame_to_time(frame, sr=22050, hop_length=512):
    """
    将帧索引转换为时间
    """
    return frame * hop_length / sr

def merge_events(events, min_gap=0.15):
    """
    合并时间间隔小于min_gap的相同类别事件
    """
    if len(events) <= 1:
        return events
        
    merged = []
    current = events[0].copy()
    
    for event in events[1:]:
        if (event['onset'] - current['offset'] <= min_gap and 
            event['event_label'] == current['event_label']):
            current['offset'] = event['offset']
        else:
            merged.append(current)
            current = event.copy()
    
    merged.append(current)
    return merged

def compute_confidence_intervals(metrics, confidence=0.95):
    """
    计算评估指标的置信区间
    """
    intervals = {}
    for metric, values in metrics.items():
        if isinstance(values, (list, np.ndarray)):
            ci = stats.t.interval(confidence, len(values)-1, 
                                loc=np.mean(values), 
                                scale=stats.sem(values))
            intervals[metric] = {
                'mean': np.mean(values),
                'ci_lower': ci[0],
                'ci_upper': ci[1]
            }
    return intervals 