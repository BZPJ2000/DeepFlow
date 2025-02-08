import torch
import torch.nn as nn

def binary_cross_entropy_loss(predictions, targets):
    """
    计算二元交叉熵损失

    参数:
        predictions (torch.Tensor): 预测概率，形状为 (batch_size,)
        targets (torch.Tensor): 目标值，形状为 (batch_size,)

    返回:
        loss (torch.Tensor): 二元交叉熵损失值
    """
    criterion = nn.BCELoss()
    loss = criterion(predictions, targets)
    return loss

# 示例调用
if __name__ == "__main__":
    batch_size = 4
    predictions = torch.tensor([0.5, 0.7, 0.9, 0.3], requires_grad=True)
    targets = torch.tensor([0.6, 0.8, 1.0, 0.4])

    loss = binary_cross_entropy_loss(predictions, targets)
    print(f"Binary Cross-Entropy Loss: {loss.item()}")