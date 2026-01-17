"""组件注册中心模块

统一管理所有已发现的组件。
"""

from typing import Dict, List, Optional
from ..components.base_component import ComponentMetadata


class ComponentRegistry:
    """组件注册中心 (单例模式)

    统一管理所有类型的组件（模型、损失函数、指标、优化器）。
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._registry = {
                'models': {},
                'losses': {},
                'metrics': {},
                'optimizers': {},
            }
        return cls._instance

    def register(self, component_type: str, name: str, metadata: ComponentMetadata):
        """注册组件

        Args:
            component_type: 组件类型 (models/losses/metrics/optimizers)
            name: 组件名称
            metadata: 组件元数据

        Raises:
            ValueError: 组件类型不存在时抛出
        """
        if component_type not in self._registry:
            raise ValueError(f"Unknown component type: {component_type}")

        self._registry[component_type][name] = metadata

    def get(self, component_type: str, name: str) -> Optional[ComponentMetadata]:
        """获取组件元数据

        Args:
            component_type: 组件类型
            name: 组件名称

        Returns:
            Optional[ComponentMetadata]: 组件元数据，不存在则返回 None
        """
        return self._registry.get(component_type, {}).get(name)

    def list(self, component_type: str, category: Optional[str] = None) -> List[ComponentMetadata]:
        """列出组件

        Args:
            component_type: 组件类型
            category: 可选的类别筛选

        Returns:
            List[ComponentMetadata]: 组件元数据列表
        """
        components = self._registry.get(component_type, ).values()

        if category:
            components = [c for c in components if c.category == category]

        return list(components)

    def search(self, query: str) -> List[ComponentMetadata]:
        """搜索组件

        Args:
            query: 搜索关键词

        Returns:
            List[ComponentMetadata]: 匹配的组件列表
        """
        results = []
        query_lower = query.lower()

        for comp_type in self._registry.values():
            for metadata in comp_type.values():
                if (query_lower in metadata.name.lower() or
                    query_lower in metadata.description.lower()):
                    results.append(metadata)

        return results

    def clear(self):
        """清空注册中心"""
        for comp_type in self._registry:
            self._registry[comp_type].clear()
