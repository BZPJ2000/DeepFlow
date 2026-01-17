"""示例 ResNet 模型

演示如何创建符合 DeepFlow 规范的模型。
"""

import torch
import torch.nn as nn
from deepflow.components.base_model import BaseModel, ComponentMetadata


class SimpleResNet(BaseModel):
    """简单的 ResNet 模型示例

    用于图像分类任务的简化版 ResNet。
    """

    @classmethod
    def get_metadata(cls) -> ComponentMetadata:
        """返回模型元数据"""
        return ComponentMetadata(
            name="SimpleResNet",
            category="vision",
            subcategory="classification",
            description="简化版 ResNet 模型，适用于图像分类任务",
            author="DeepFlow Team",
            version="1.0.0",
            tags=["classification", "resnet", "example"]
        )

    @classmethod
    def get_required_params(cls):
        """返回必需参数"""
        return {
            'num_classes': int,
        }

    @classmethod
    def get_optional_params(cls):
        """返回可选参数"""
        return {
            'in_channels': 3,
            'dropout': 0.5,
        }

    def __init__(self, num_classes: int, in_channels: int = 3, dropout: float = 0.5):
        """初始化模型

        Args:
            num_classes: 分类数量
            in_channels: 输入通道数
            dropout: Dropout 比率
        """
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            dropout=dropout
        )

        # 特征提取层
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # 简化的残差块
            self._make_layer(64, 128, 2),
            self._make_layer(128, 256, 2),
            self._make_layer(256, 512, 2),
        )

        # 分类层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def _make_layer(self, in_channels, out_channels, num_blocks):
        """创建残差层"""
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_blocks - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        """前向传播

        Args:
            x: 输入张量 [B, C, H, W]

        Returns:
            输出张量 [B, num_classes]
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
