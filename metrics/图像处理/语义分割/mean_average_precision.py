from sklearn.metrics import average_precision_score


class MeanAveragePrecision:
    """
    计算平均精度均值（mAP）的指标类

    参数：
    average : str, 默认为'macro'
        平均方式，可选 'micro', 'macro', 'samples', 'weighted'

    示例：
    >>> metric = MeanAveragePrecision(average='macro')
    >>> y_true = [1, 0, 1, 0]
    >>> y_pred = [0.9, 0.1, 0.8, 0.2]
    >>> print(f"mAP: {metric(y_true, y_pred):.4f}")
    """

    def __init__(self, average='macro'):
        self.average = average

    def __call__(self, y_true, y_pred):
        """
        计算指标值

        参数：
        y_true : array-like
            真实标签列表
        y_pred : array-like
            预测概率列表

        返回：
        float: 计算得到的mAP值
        """
        return average_precision_score(
            y_true,
            y_pred,
            average=self.average
        )


class IntersectionOverUnion:
    """
    计算两个边界框交并比（IoU）的指标类

    示例：
    >>> metric = IntersectionOverUnion()
    >>> box1 = [0, 0, 10, 10]
    >>> box2 = [5, 5, 15, 15]
    >>> print(f"IoU: {metric(box1, box2):.4f}")
    """

    def __call__(self, box1, box2):
        """
        计算交并比

        参数：
        box1 : list
            第一个边界框坐标 [x1, y1, x2, y2]
        box2 : list
            第二个边界框坐标 [x1, y1, x2, y2]

        返回：
        float: IoU值（0.0~1.0）
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area != 0 else 0.0


# 使用示例
if __name__ == "__main__":
    # 创建指标实例
    map_metric = MeanAveragePrecision(average='macro')
    iou_metric = IntersectionOverUnion()

    # 测试数据
    y_true = [1, 0, 1, 0]
    y_pred = [0.9, 0.1, 0.8, 0.2]
    box1 = [0, 0, 10, 10]
    box2 = [5, 5, 15, 15]

    # 计算指标
    print(f"mAP: {map_metric(y_true, y_pred):.4f}")  # 输出: mAP: 1.0000
    print(f"IoU: {iou_metric(box1, box2):.4f}")  # 输出: IoU: 0.1429