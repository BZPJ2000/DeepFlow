import torch
import torch.nn as nn

def dqn_huber_loss(q_values, target_q_values, delta=1.0):
    """
    计算 DQN 的 Huber 损失

    参数:
        q_values (torch.Tensor): 预测的 Q 值，形状为 (batch_size,)
        target_q_values (torch.Tensor): 目标 Q 值，形状为 (batch_size,)
        delta (float): Huber 损失的阈值

    返回:
        losses (torch.Tensor): Huber 损失值
    """
    criterion = nn.HuberLoss(delta=delta)
    loss = criterion(q_values, target_q_values)
    return loss
