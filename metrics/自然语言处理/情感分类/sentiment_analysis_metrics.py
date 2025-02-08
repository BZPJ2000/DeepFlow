import torch

def accuracy(y_true, y_pred):
    """
    计算准确率

    参数:
        y_true (torch.Tensor): 真实标签，形状为 (batch_size,)
        y_pred (torch.Tensor): 预测标签，形状为 (batch_size,)

    返回:
        acc (float): 准确率
    """
    correct = (y_true == y_pred).sum().item()
    total = y_true.size(0)
    acc = correct / total
    return acc

def precision(y_true, y_pred, num_classes):
    """
    计算精确率

    参数:
        y_true (torch.Tensor): 真实标签，形状为 (batch_size,)
        y_pred (torch.Tensor): 预测标签，形状为 (batch_size,)
        num_classes (int): 类别数量

    返回:
        prec (float): 精确率
    """
    prec = 0.0
    for c in range(num_classes):
        true_positives = ((y_true == c) & (y_pred == c)).sum().item()
        predicted_positives = (y_pred == c).sum().item()
        if predicted_positives > 0:
            prec += true_positives / predicted_positives
    prec /= num_classes
    return prec

def recall(y_true, y_pred, num_classes):
    """
    计算召回率

    参数:
        y_true (torch.Tensor): 真实标签，形状为 (batch_size,)
        y_pred (torch.Tensor): 预测标签，形状为 (batch_size,)
        num_classes (int): 类别数量

    返回:
        rec (float): 召回率
    """
    rec = 0.0
    for c in range(num_classes):
        true_positives = ((y_true == c) & (y_pred == c)).sum().item()
        actual_positives = (y_true == c).sum().item()
        if actual_positives > 0:
            rec += true_positives / actual_positives
    rec /= num_classes
    return rec

def f1_score(y_true, y_pred, num_classes):
    """
    计算 F1 分数

    参数:
        y_true (torch.Tensor): 真实标签，形状为 (batch_size,)
        y_pred (torch.Tensor): 预测标签，形状为 (batch_size,)
        num_classes (int): 类别数量

    返回:
        f1 (float): F1 分数
    """
    prec = precision(y_true, y_pred, num_classes)
    rec = recall(y_true, y_pred, num_classes)
    if (prec + rec) > 0:
        f1 = 2 * (prec * rec) / (prec + rec)
    else:
        f1 = 0.0
    return f1

# 示例调用
if __name__ == "__main__":
    # 假设是一个 3 分类任务
    num_classes = 3
    batch_size = 10

    # 真实标签
    y_true = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

    # 预测标签
    y_pred = torch.tensor([0, 1, 1, 0, 2, 2, 0, 1, 2, 0])

    # 计算评价指标
    acc = accuracy(y_true, y_pred)
    prec = precision(y_true, y_pred, num_classes)
    rec = recall(y_true, y_pred, num_classes)
    f1 = f1_score(y_true, y_pred, num_classes)

    # 打印结果
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")