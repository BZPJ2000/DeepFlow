import torch

def iou_loss(predictions, targets, smooth=1e-6):
    """
    计算 IoU Loss

    参数:
        predictions (torch.Tensor): 预测概率，形状为 (batch_size,)
        targets (torch.Tensor): 目标值，形状为 (batch_size,)
        smooth (float): 平滑因子

    返回:
        loss (torch.Tensor): IoU Loss 值
    """
    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum() - intersection
    loss = 1 - (intersection + smooth) / (union + smooth)
    return loss

# 示例调用
if __name__ == "__main__":
    batch_size = 4
    predictions = torch.tensor([0.5, 0.7, 0.9, 0.3], requires_grad=True)
    targets = torch.tensor([0.6, 0.8, 1.0, 0.4])

    loss = iou_loss(predictions, targets)
    print(f"IoU Loss: {loss.item()}")