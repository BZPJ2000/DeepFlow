import os
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"
import torch
from torch import nn


class CrossEntropyLoss(nn.Module):
    """
    交叉熵损失函数封装类（PyTorch实现）

    参数:
        weight (torch.Tensor, optional): 各类别的权重，默认为None
        ignore_index (int, optional): 要忽略的目标类别索引，默认为-100
        reduction (str, optional): 损失缩减方式，可选 'none' | 'mean' | 'sum'，默认为'mean'

    示例:
        >>> criterion = CrossEntropyLoss()
        >>> outputs = torch.randn(3, 5)  # (batch_size, num_classes)
        >>> labels = torch.tensor([1, 0, 4])  # (batch_size,)
        >>> loss = criterion(outputs, labels)
    """

    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index,
            reduction=reduction
        )

    def forward(self, model_output, true_labels):
        """
        前向计算损失

        参数:
            model_output (torch.Tensor): 模型原始输出（未经过softmax），形状为 (N, C)
            true_labels (torch.Tensor): 真实标签（类别索引），形状为 (N,)

        返回:
            torch.Tensor: 计算得到的损失值
        """
        # 自动处理维度不匹配的情况
        if true_labels.dim() == 2:
            true_labels = true_labels.squeeze(1)

        return self.criterion(model_output, true_labels)

    def extra_repr(self):
        """显示初始化参数信息"""
        return f'weight={self.criterion.weight}, ignore_index={self.criterion.ignore_index}, reduction={self.criterion.reduction}'


# 使用示例
if __name__ == "__main__":
    # 创建损失函数实例（可配置参数）
    criterion = CrossEntropyLoss()

    # 模拟输入数据
    batch_size = 4
    num_classes = 3
    outputs = torch.randn(batch_size, num_classes)  # 模型原始输出
    labels = torch.randint(0, num_classes, (batch_size,))  # 真实标签

    # 计算损失
    loss = criterion(outputs, labels)
    print(f"Cross Entropy Loss: {loss.item():.4f}")