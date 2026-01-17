import torch
import torch.nn as nn

def value_function_loss(predicted_values, target_values):
    """
    计算值函数损失

    参数:
        predicted_values (torch.Tensor): Critic 网络预测的值，形状为 (batch_size,)
        target_values (torch.Tensor): 目标值（如回报或目标 Q 值），形状为 (batch_size,)

    返回:
        loss (torch.Tensor): 值函数损失值
    """
    # 使用均方误差（MSE）作为损失函数
    criterion = nn.MSELoss()
    loss = criterion(predicted_values, target_values)
    return loss

# 示例调用
if __name__ == "__main__":
    # 假设 batch_size = 4
    batch_size = 4

    # Critic 网络预测的值（例如，状态值函数 V(s) 或 Q 值）
    predicted_values = torch.tensor([0.5, 0.7, 0.9, 0.3], requires_grad=True)

    # 目标值（例如，回报或目标 Q 值）
    # 在强化学习中，目标值通常通过 Bellman 方程计算：
    # target_values = rewards + gamma * next_predicted_values * (1 - dones)
    target_values = torch.tensor([0.6, 0.8, 1.0, 0.4])

    # 计算值函数损失
    loss = value_function_loss(predicted_values, target_values)

    # 打印损失值
    print(f"Value Function Loss: {loss.item()}")