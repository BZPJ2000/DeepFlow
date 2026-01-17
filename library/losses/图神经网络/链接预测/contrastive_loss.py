import torch

def contrastive_loss(embedding1, embedding2, label, margin=1.0):
    """
    计算对比损失

    参数:
        embedding1 (torch.Tensor): 样本 1 的嵌入向量，形状为 (batch_size, embedding_dim)
        embedding2 (torch.Tensor): 样本 2 的嵌入向量，形状为 (batch_size, embedding_dim)
        label (torch.Tensor): 样本对的标签（1 表示相似，0 表示不相似），形状为 (batch_size,)
        margin (float): 间隔参数

    返回:
        loss (torch.Tensor): 对比损失值
    """
    distance = torch.norm(embedding1 - embedding2, dim=1)
    loss = (label * torch.square(distance) + (1 - label) * torch.square(torch.clamp(margin - distance, min=0.0))).mean()
    return loss

# 示例调用
if __name__ == "__main__":
    batch_size = 4
    embedding_dim = 16
    embedding1 = torch.randn(batch_size, embedding_dim)
    embedding2 = torch.randn(batch_size, embedding_dim)
    label = torch.tensor([1, 0, 1, 0])

    loss = contrastive_loss(embedding1, embedding2, label)
    print(f"Contrastive Loss: {loss.item()}")