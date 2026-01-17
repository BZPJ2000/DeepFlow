"""组件自动发现引擎模块

自动扫描并识别组件库中的所有可用组件。
"""

import os
import ast
import importlib.util
from pathlib import Path
from typing import List, Dict, Optional
from ..components.base_component import ComponentMetadata


class ComponentDiscovery:
    """组件自动发现引擎

    扫描指定目录，自动识别符合规范的组件。
    """

    def __init__(self, base_path: str):
        """初始化发现引擎

        Args:
            base_path: 组件库根目录路径
        """
        self.base_path = Path(base_path)
        self.discovered_components = {}

    def discover_all(self) -> Dict[str, List[ComponentMetadata]]:
        """发现所有组件

        Returns:
            Dict[str, List[ComponentMetadata]]: 按类型分组的组件列表
        """
        results = {
            'models': self.discover_in_path(self.base_path / 'models'),
            'losses': self.discover_in_path(self.base_path / 'losses'),
            'metrics': self.discover_in_path(self.base_path / 'metrics'),
            'optimizers': self.discover_in_path(self.base_path / 'optimizers'),
        }
        return results

    def discover_in_path(self, path: Path) -> List[ComponentMetadata]:
        """在指定路径发现组件

        Args:
            path: 扫描路径

        Returns:
            List[ComponentMetadata]: 发现的组件列表
        """
        components = []

        if not path.exists():
            return components

        for py_file in path.rglob("*.py"):
            if py_file.name.startswith('_'):
                continue

            try:
                metadata_list = self._extract_components(py_file)
                components.extend(metadata_list)
            except Exception as e:
                print(f"Warning: Failed to process {py_file}: {e}")

        return components

    def _extract_components(self, file_path: Path) -> List[ComponentMetadata]:
        """从文件中提取组件

        Args:
            file_path: Python 文件路径

        Returns:
            List[ComponentMetadata]: 提取的组件列表
        """
        components = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
        except SyntaxError:
            return components

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if self._is_component_class(node):
                    metadata = self._extract_metadata(node, file_path)
                    if metadata:
                        components.append(metadata)

        return components

    def _is_component_class(self, node: ast.ClassDef) -> bool:
        """检查是否为组件类

        Args:
            node: AST 类定义节点

        Returns:
            bool: 是否为组件类
        """
        base_classes = ['BaseModel', 'BaseLoss', 'BaseMetric', 'BaseOptimizer']

        for base in node.bases:
            if isinstance(base, ast.Name) and base.id in base_classes:
                return True

        return False

    def _extract_metadata(self, node: ast.ClassDef, file_path: Path) -> Optional[ComponentMetadata]:
        """提取组件元数据

        Args:
            node: AST 类定义节点
            file_path: 文件路径

        Returns:
            Optional[ComponentMetadata]: 组件元数据
        """
        # 提取类别和子类别
        parts = file_path.parts
        try:
            # 找到 models/losses/metrics/optimizers 的索引
            for i, part in enumerate(parts):
                if part in ['models', 'losses', 'metrics', 'optimizers']:
                    category = parts[i + 1] if i + 1 < len(parts) else 'unknown'
                    subcategory = parts[i + 2] if i + 2 < len(parts) else 'unknown'
                    break
            else:
                category = 'unknown'
                subcategory = 'unknown'
        except IndexError:
            category = 'unknown'
            subcategory = 'unknown'

        return ComponentMetadata(
            name=node.name,
            category=category,
            subcategory=subcategory,
            description=ast.get_docstring(node) or "",
        )
