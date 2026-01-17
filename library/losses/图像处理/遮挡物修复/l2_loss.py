import torch
import torch.nn as nn

class L2Loss:
    """
    封装 L2 损失（均方误差损失）为类
    """
    def __init__(self, reduction='mean', weight=None):
        """
        初始化 L2 损失
        :param reduction: 损失计算模式，支持 'mean'、'sum' 和 'none'，默认为 'mean'
        :param weight: 样本权重，形状为 (batch_size,)，默认为 None
        """
        self.reduction = reduction
        self.weight = weight
        self.criterion = nn.MSELoss(reduction=reduction)

    def calculate(self, predictions, targets):
        """
        计算 L2 损失
        :param predictions: 预测值，形状为 (batch_size,)
        :param targets: 目标值，形状为 (batch_size,)
        :return: L2 损失值
        """
        if self.weight is not None:
            # 如果提供了权重，则对损失进行加权
            squared_diff = (predictions - targets) ** 2
            weighted_loss = self.weight * squared_diff
            if self.reduction == 'mean':
                return torch.mean(weighted_loss)
            elif self.reduction == 'sum':
                return torch.sum(weighted_loss)
            elif self.reduction == 'none':
                return weighted_loss
            else:
                raise ValueError(f"Unsupported reduction mode: {self.reduction}")
        else:
            # 如果没有提供权重，直接使用 PyTorch 的 MSELoss
            return self.criterion(predictions, targets)

if __name__ == "__main__":
    # 示例数据
    batch_size = 4
    predictions = torch.tensor([0.5, 0.7, 0.9, 0.3], requires_grad=True)
    targets = torch.tensor([0.6, 0.8, 1.0, 0.4])
    weights = torch.tensor([1.0, 2.0, 1.0, 0.5])  # 样本权重

    # 实例化 L2Loss 类
    l2_loss_mean = L2Loss(reduction='mean')  # 默认 reduction 为 'mean'
    l2_loss_sum = L2Loss(reduction='sum')
    l2_loss_weighted = L2Loss(reduction='mean', weight=weights)

    # 计算损失
    loss_mean = l2_loss_mean.calculate(predictions, targets)
    loss_sum = l2_loss_sum.calculate(predictions, targets)
    loss_weighted = l2_loss_weighted.calculate(predictions, targets)

    # 打印结果
    print(f"L2 Loss (mean): {loss_mean.item()}")
    print(f"L2 Loss (sum): {loss_sum.item()}")
    print(f"L2 Loss (weighted mean): {loss_weighted.item()}")