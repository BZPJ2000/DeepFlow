"""评估指标基类模块

定义评估指标的基础接口。
"""

from abc import ABC, abstractmethod
from .base_component import BaseComponent, ComponentMetadata


class BaseMetric(BaseComponent, ABC):
    """评估指标基类

    所有评估指标的抽象基类。
    """

    def __init__(self, **kwargs):
        """初始化评估指标

        Args:
            **kwargs: 指标参数
        """
        self.config = kwargs
        self.reset()

    @abstractmethod
    def update(self, pred, target):
        """更新指标状态

        Args:
            pred: 预测值
            target: 目标值
        """
        pass

    @abstractmethod
    def compute(self):
        """计算指标值

        Returns:
            指标值
        """
        pass

    @abstractmethod
    def reset(self):
        """重置指标状态"""
        pass
