import torch
import torch.nn as nn

class PerceptualLoss:
    """
    封装感知损失为类
    """
    def __init__(self, reduction='mean', feature_extractor=None):
        """
        初始化感知损失
        :param reduction: 损失计算模式，支持 'mean'、'sum' 和 'none'，默认为 'mean'
        :param feature_extractor: 特征提取器（如预训练的 VGG 网络），默认为 None
        """
        self.reduction = reduction
        self.feature_extractor = feature_extractor
        self.criterion = nn.L1Loss(reduction=reduction)

    def calculate(self, features1, features2):
        """
        计算感知损失
        :param features1: 特征图 1，形状为 (batch_size, channels, height, width)
        :param features2: 特征图 2，形状为 (batch_size, channels, height, width)
        :return: 感知损失值
        """
        if self.feature_extractor is not None:
            # 如果提供了特征提取器，则先提取特征
            features1 = self.feature_extractor(features1)
            features2 = self.feature_extractor(features2)

        # 计算 L1 损失
        loss = self.criterion(features1, features2)
        return loss

if __name__ == "__main__":
    # 示例数据
    batch_size = 4
    channels = 3
    height = 64
    width = 64
    features1 = torch.randn(batch_size, channels, height, width)
    features2 = torch.randn(batch_size, channels, height, width)

    # 实例化 PerceptualLoss 类
    perceptual_loss_mean = PerceptualLoss(reduction='mean')  # 默认 reduction 为 'mean'
    perceptual_loss_sum = PerceptualLoss(reduction='sum')

    # 计算损失
    loss_mean = perceptual_loss_mean.calculate(features1, features2)
    loss_sum = perceptual_loss_sum.calculate(features1, features2)

    # 打印结果
    print(f"Perceptual Loss (mean): {loss_mean.item()}")
    print(f"Perceptual Loss (sum): {loss_sum.item()}")