import torch
import torch.nn as nn

def cross_loss_pytorch(model_output, true_labels):
    """
    计算交叉熵损失（PyTorch 实现）

    参数:
        model_output (torch.Tensor): 模型的原始输出（未经过 Softmax），形状为 (batch_size, num_classes)
        true_labels (torch.Tensor): 真实标签（类别索引），形状为 (batch_size,)

    返回:
        losses (torch.Tensor): 交叉熵损失值
    """
    criterion = nn.CrossEntropyLoss()
    loss = criterion(model_output, true_labels)
    return loss


