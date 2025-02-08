from sklearn.metrics import average_precision_score

class MeanAveragePrecision:
    """
    封装平均精度 (mAP) 为类
    """
    def __init__(self, average='macro'):
        """
        初始化 mAP 计算器
        :param average: 计算方式，默认为 'macro'
        """
        self.average = average

    def calculate(self, y_true, y_pred):
        """
        计算平均精度 (mAP)
        :param y_true: 真实标签
        :param y_pred: 预测概率
        :return: mAP 值
        """
        return average_precision_score(y_true, y_pred, average=self.average)

class IoU:
    """
    封装交并比 (IoU) 为类
    """
    @staticmethod
    def calculate(box1, box2):
        """
        计算两个边界框的交并比 (IoU)
        :param box1: 第一个边界框，格式为 [x1, y1, x2, y2]
        :param box2: 第二个边界框，格式为 [x1, y1, x2, y2]
        :return: IoU 值
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area


if __name__ == "__main__":
    # 示例数据
    y_true = [1, 0, 1, 0]
    y_pred = [0.9, 0.1, 0.8, 0.2]
    box1 = [0, 0, 10, 10]
    box2 = [5, 5, 15, 15]

    # 使用 MeanAveragePrecision 类计算 mAP
    map_calculator = MeanAveragePrecision(average='macro')
    mAP = map_calculator.calculate(y_true, y_pred)

    # 使用 IoU 类计算 IoU
    iou_calculator = IoU()
    iou_value = iou_calculator.calculate(box1, box2)

    # 打印结果
    print(f"mAP: {mAP:.4f}")
    print(f"IoU: {iou_value:.4f}")





