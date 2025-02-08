import torch
import torch.nn as nn

def smooth_l1_loss(predictions, targets):
    """
    计算平滑 L1 损失

    参数:
        predictions (torch.Tensor): 预测值，形状为 (batch_size,)
        targets (torch.Tensor): 目标值，形状为 (batch_size,)

    返回:
        loss (torch.Tensor): 平滑 L1 损失值
    """
    criterion = nn.SmoothL1Loss()
    loss = criterion(predictions, targets)
    return loss

# 示例调用
if __name__ == "__main__":
    batch_size = 4
    predictions = torch.tensor([0.5, 0.7, 0.9, 0.3], requires_grad=True)
    targets = torch.tensor([0.6, 0.8, 1.0, 0.4])

    loss = smooth_l1_loss(predictions, targets)
    print(f"Smooth L1 Loss: {loss.item()}")