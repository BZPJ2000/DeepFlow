"""损失函数基类模块

定义损失函数的基础接口。
"""

import torch.nn as nn
from abc import abstractmethod
from .base_component import BaseComponent, ComponentMetadata


class BaseLoss(nn.Module, BaseComponent):
    """损失函数基类

    所有损失函数的抽象基类。
    """

    def __init__(self, **kwargs):
        """初始化损失函数

        Args:
            **kwargs: 损失函数参数
        """
        super().__init__()
        self.config = kwargs

    @abstractmethod
    def forward(self, pred, target):
        """计算损失

        Args:
            pred: 预测值
            target: 目标值

        Returns:
            损失值
        """
        pass
