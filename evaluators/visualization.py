import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from evaluators.base_evaluator import BaseEvaluator


class VisualizationEvaluator(BaseEvaluator):
    """声音事件检测可视化评估器
    
    这个类负责生成各种评估可视化结果,包括:
    1. 混淆矩阵
    2. 性能指标图表
    3. 检测结果可视化
    
    继承自BaseEvaluator,使用其提供的基础功能。
    """
    
    def __init__(self, model_type, device, experiment_type=None):
        super().__init__(model_type, device, experiment_type)
    
    def plot_confusion_matrix(self, y_preds, y_labels, model_path):
        """绘制混淆矩阵热图
        
        将模型预测结果与真实标签的对比以热图形式可视化。
        
        Args:
            y_preds (torch.Tensor): 预测结果张量
            y_labels (torch.Tensor): 真实标签张量
            model_path (Path): 模型路径
        """
        fold_name = model_path.stem
        save_path = self.plot_dir / f'confusion_matrix_{fold_name}.png'
        
        # 转换数据格式
        y_pred = y_preds.cpu().numpy().reshape(-1, self.num_classes)
        y_label = y_labels.cpu().numpy().reshape(-1, self.num_classes)
        
        # 计算混淆矩阵
        total_cm = np.zeros((self.num_classes, self.num_classes))
        
        for i in range(self.num_classes):
            true_indices = np.where(y_label[:, i] == 1)[0]
            for index in true_indices:
                for j in range(self.num_classes):
                    if i != j:  # 交叉检测
                        if y_label[index, j] == 0 and y_pred[index, j] == 1:
                            total_cm[i, j] += 1
                    else:  # 正确检测
                        if y_pred[index, i] == 1:
                            total_cm[i, i] += 1
        
        # 计算百分比
        row_sums = total_cm.sum(axis=1)
        cm_percentage = np.zeros_like(total_cm, dtype=float)
        for i in range(self.num_classes):
            if row_sums[i] > 0:
                cm_percentage[i] = total_cm[i] / row_sums[i]
        
        # 绘制热图
        plt.figure(figsize=(4.5, 3.2))
        sns.heatmap(
            cm_percentage, 
            annot=True,  # 显示数值
            fmt='.2f',   # 数值格式
            cmap='Purples',  # 配色方案
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted classes')
        plt.ylabel('True classes')
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(
            save_path,
            bbox_inches='tight',
            pad_inches=0.1,
            dpi=300
        )
        plt.close()
    
    def plot_metrics_by_class(self, metrics, fold_name):
        """绘制每个类别的性能指标柱状图
        
        Args:
            metrics (dict): 评估指标字典
            fold_name (str): 模型名称
        """
        save_path = self.plot_dir / f'metrics_by_class_{fold_name}.png'
        
        metrics_to_plot = ['precision', 'recall', 'f1']
        n_metrics = len(metrics_to_plot)
        
        plt.figure(figsize=(12, 4))
        x = np.arange(self.num_classes)
        width = 0.25
        
        for i, metric in enumerate(metrics_to_plot):
            values = metrics[metric]
            plt.bar(x + i*width, values, width, label=metric.capitalize())
        
        plt.xlabel('Classes')
        plt.ylabel('Score')
        plt.title('Performance Metrics by Class')
        plt.xticks(x + width, self.class_names, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def plot_detection_timeline(self, predictions, ground_truth, audio_name):
        """绘制检测结果时间线
        
        Args:
            predictions (pd.DataFrame): 预测结果数据框
            ground_truth (pd.DataFrame): 真实标签数据框
            audio_name (str): 音频文件名
        """
        save_path = self.plot_dir / f'timeline_{audio_name}.png'
        
        plt.figure(figsize=(15, 5))
        
        # 筛选当前音频的数据
        pred_events = predictions[predictions['filename'] == audio_name]
        true_events = ground_truth[ground_truth['filename'] == audio_name]
        
        # 绘制预测结果
        for _, event in pred_events.iterrows():
            plt.barh(
                y=self.class_names.index(event['event_label']),
                width=event['offset'] - event['onset'],
                left=event['onset'],
                alpha=0.5,
                label='Prediction'
            )
        
        # 绘制真实标签
        for _, event in true_events.iterrows():
            plt.barh(
                y=self.class_names.index(event['event_label']),
                width=event['offset'] - event['onset'],
                left=event['onset'],
                alpha=0.5,
                label='Ground Truth',
                color='red'
            )
        
        plt.yticks(range(len(self.class_names)), self.class_names)
        plt.xlabel('Time (seconds)')
        plt.title(f'Detection Timeline - {audio_name}')
        
        # 去除重复的图例
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()