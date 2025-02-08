import torch

def policy_gradient_loss(log_probs, advantages):
    """
    计算策略梯度损失

    参数:
        log_probs (torch.Tensor): 动作的对数概率，形状为 (batch_size,)
        advantages (torch.Tensor): 优势值，形状为 (batch_size,)

    返回:
        loss (torch.Tensor): 策略梯度损失值
    """
    loss = -torch.mean(log_probs * advantages)
    return loss

# 示例调用
if __name__ == "__main__":
    batch_size = 4
    log_probs = torch.tensor([-0.5, -0.7, -0.9, -0.3], requires_grad=True)
    advantages = torch.tensor([0.6, 0.8, 1.0, 0.4])

    loss = policy_gradient_loss(log_probs, advantages)
    print(f"Policy Gradient Loss: {loss.item()}")