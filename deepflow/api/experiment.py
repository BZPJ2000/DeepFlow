"""实验管理 API 模块

提供实验管理的高级接口。
"""

from typing import Dict, List, Optional
from pathlib import Path
from ..core.registry import ComponentRegistry
from ..core.discovery import ComponentDiscovery
from ..core.loader import ComponentLoader
from ..components.base_component import ComponentMetadata


class ExperimentAPI:
    """实验管理 API

    提供实验创建、配置、执行的统一接口。
    """

    def __init__(self, library_path: str = 'library'):
        """初始化 API

        Args:
            library_path: 组件库路径
        """
        self.library_path = library_path
        self.registry = ComponentRegistry()
        self.loader = ComponentLoader(self.registry)
        self._initialized = False

    def initialize(self):
        """初始化组件发现"""
        if self._initialized:
            return

        discovery = ComponentDiscovery(self.library_path)
        discovered = discovery.discover_all()

        # 注册所有组件
        for comp_type, components in discovered.items():
            for comp in components:
                self.registry.register(comp_type, comp.name, comp)

        self._initialized = True

    def get_available_models(
        self,
        category: Optional[str] = None,
        subcategory: Optional[str] = None
    ) -> List[ComponentMetadata]:
        """获取可用模型列表

        Args:
            category: 类别筛选
            subcategory: 子类别筛选

        Returns:
            List[ComponentMetadata]: 模型列表
        """
        if not self._initialized:
            self.initialize()

        models = self.registry.list('models', category=category)

        if subcategory:
            models = [m for m in models if m.subcategory == subcategory]

        return models

    def get_available_losses(self, category: Optional[str] = None) -> List[ComponentMetadata]:
        """获取可用损失函数列表"""
        if not self._initialized:
            self.initialize()
        return self.registry.list('losses', category=category)

    def get_available_metrics(self, category: Optional[str] = None) -> List[ComponentMetadata]:
        """获取可用评估指标列表"""
        if not self._initialized:
            self.initialize()
        return self.registry.list('metrics', category=category)

    def search_components(self, query: str) -> List[ComponentMetadata]:
        """搜索组件

        Args:
            query: 搜索关键词

        Returns:
            List[ComponentMetadata]: 匹配的组件列表
        """
        if not self._initialized:
            self.initialize()
        return self.registry.search(query)
