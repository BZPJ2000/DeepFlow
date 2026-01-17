"""组件基类模块

定义所有组件的基础接口和元数据结构。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


@dataclass
class ComponentMetadata:
    """组件元数据

    Attributes:
        name: 组件名称
        category: 类别 (nlp/vision/graph/rl)
        subcategory: 子类别
        description: 组件描述
        author: 作者
        version: 版本号
        tags: 标签列表
    """
    name: str
    category: str
    subcategory: str
    description: str
    author: Optional[str] = None
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)


class BaseComponent(ABC):
    """组件基类

    所有组件（模型、损失函数、指标、优化器）的抽象基类。
    """

    @classmethod
    @abstractmethod
    def get_metadata(cls) -> ComponentMetadata:
        """返回组件元数据

        Returns:
            ComponentMetadata: 组件元数据对象
        """
        pass

    @classmethod
    @abstractmethod
    def get_required_params(cls) -> Dict[str, type]:
        """返回必需参数及其类型

        Returns:
            Dict[str, type]: 参数名到类型的映射
        """
        pass

    @classmethod
    def get_optional_params(cls) -> Dict[str, Any]:
        """返回可选参数及其默认值

        Returns:
            Dict[str, Any]: 参数名到默认值的映射
        """
        return {}
