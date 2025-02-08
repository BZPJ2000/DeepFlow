import torch
import torch.nn as nn

def q_learning_mse_loss(q_values, target_q_values):
    """
    计算 Q-Learning 的 MSE 损失

    参数:
        q_values (torch.Tensor): 预测的 Q 值，形状为 (batch_size,)
        target_q_values (torch.Tensor): 目标 Q 值，形状为 (batch_size,)

    返回:
        losses (torch.Tensor): MSE 损失值
    """
    criterion = nn.MSELoss()
    loss = criterion(q_values, target_q_values)
    return loss
