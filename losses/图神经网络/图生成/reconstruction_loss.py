import torch
import torch.nn as nn

def reconstruction_loss(inputs, reconstructions):
    """
    计算重构损失

    参数:
        inputs (torch.Tensor): 输入数据，形状为 (batch_size, ...)
        reconstructions (torch.Tensor): 重构数据，形状为 (batch_size, ...)

    返回:
        loss (torch.Tensor): 重构损失值
    """
    criterion = nn.MSELoss()
    loss = criterion(reconstructions, inputs)
    return loss

# 示例调用
if __name__ == "__main__":
    batch_size = 4
    inputs = torch.randn(batch_size, 3, 64, 64)
    reconstructions = torch.randn(batch_size, 3, 64, 64)

    loss = reconstruction_loss(inputs, reconstructions)
    print(f"Reconstruction Loss: {loss.item()}")