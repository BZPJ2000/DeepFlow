import torch
import torch.nn.functional as F

def focal_loss(predictions, targets, alpha=0.25, gamma=2.0):
    """
    计算 Focal Loss

    参数:
        predictions (torch.Tensor): 预测概率，形状为 (batch_size, num_classes)
        targets (torch.Tensor): 目标类别索引，形状为 (batch_size,)
        alpha (float): 平衡因子
        gamma (float): 调节因子

    返回:
        loss (torch.Tensor): Focal Loss 值
    """
    ce_loss = F.cross_entropy(predictions, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    loss = alpha * (1 - pt) ** gamma * ce_loss
    return loss.mean()

# 示例调用
if __name__ == "__main__":
    batch_size = 4
    num_classes = 3
    predictions = torch.randn(batch_size, num_classes)
    targets = torch.tensor([0, 2, 1, 0])

    loss = focal_loss(predictions, targets)
    print(f"Focal Loss: {loss.item()}")