"""模型基类模块

定义深度学习模型的基础接口。
"""

import torch.nn as nn
from abc import abstractmethod
from typing import Dict, Any
from .base_component import BaseComponent, ComponentMetadata


class BaseModel(nn.Module, BaseComponent):
    """模型基类

    所有深度学习模型的抽象基类，继承自 PyTorch nn.Module。
    """

    def __init__(self, **kwargs):
        """初始化模型

        Args:
            **kwargs: 模型参数
        """
        super().__init__()
        self.config = kwargs

    @abstractmethod
    def forward(self, x):
        """前向传播

        Args:
            x: 输入张量

        Returns:
            输出张量
        """
        pass

    def get_num_parameters(self) -> int:
        """计算模型参数量

        Returns:
            int: 总参数数量
        """
        return sum(p.numel() for p in self.parameters())

    def get_model_size(self) -> float:
        """计算模型大小

        Returns:
            float: 模型大小 (MB)
        """
        param_size = sum(p.nelement() * p.element_size()
                        for p in self.parameters())
        buffer_size = sum(b.nelement() * b.element_size()
                         for b in self.buffers())
        return (param_size + buffer_size) / 1024 / 1024
