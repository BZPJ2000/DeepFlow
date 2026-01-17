import torch
import torch.nn.functional as F

def nll_loss(log_probs, targets):
    """
    计算负对数似然损失

    参数:
        log_probs (torch.Tensor): 对数概率，形状为 (batch_size, num_classes)
        targets (torch.Tensor): 目标类别索引，形状为 (batch_size,)

    返回:
        loss (torch.Tensor): 负对数似然损失值
    """
    loss = F.nll_loss(log_probs, targets)
    return loss

# 示例调用
if __name__ == "__main__":
    batch_size = 4
    num_classes = 3
    log_probs = torch.randn(batch_size, num_classes)
    targets = torch.tensor([0, 2, 1, 0])

    loss = nll_loss(log_probs, targets)
    print(f"NLL Loss: {loss.item()}")