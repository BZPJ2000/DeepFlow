import torch
import torch.nn as nn

def advantage_loss(predicted_values, target_values):
    """
    计算优势损失

    参数:
        predicted_values (torch.Tensor): 预测值，形状为 (batch_size,)
        target_values (torch.Tensor): 目标值，形状为 (batch_size,)

    返回:
        loss (torch.Tensor): 优势损失值
    """
    criterion = nn.MSELoss()
    loss = criterion(predicted_values, target_values)
    return loss

# 示例调用
if __name__ == "__main__":
    batch_size = 4
    predicted_values = torch.tensor([0.5, 0.7, 0.9, 0.3], requires_grad=True)
    target_values = torch.tensor([0.6, 0.8, 1.0, 0.4])

    loss = advantage_loss(predicted_values, target_values)
    print(f"Advantage Loss: {loss.item()}")