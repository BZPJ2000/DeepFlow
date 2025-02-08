import torch
import torch.nn as nn

def adversarial_loss(discriminator_output, is_real=True):
    """
    计算对抗损失

    参数:
        discriminator_output (torch.Tensor): 判别器的输出，形状为 (batch_size,)
        is_real (bool): 是否为真实样本

    返回:
        loss (torch.Tensor): 对抗损失值
    """
    criterion = nn.BCEWithLogitsLoss()
    targets = torch.ones_like(discriminator_output) if is_real else torch.zeros_like(discriminator_output)
    loss = criterion(discriminator_output, targets)
    return loss

# 示例调用
if __name__ == "__main__":
    batch_size = 4
    discriminator_output = torch.randn(batch_size)

    # 计算真实样本的对抗损失
    real_loss = adversarial_loss(discriminator_output, is_real=True)
    print(f"Adversarial Loss (Real): {real_loss.item()}")

    # 计算生成样本的对抗损失
    fake_loss = adversarial_loss(discriminator_output, is_real=False)
    print(f"Adversarial Loss (Fake): {fake_loss.item()}")