"""组件动态加载器模块

根据配置动态加载和实例化组件。
"""

import importlib
from pathlib import Path
from typing import Any, Dict, Optional
import torch.nn as nn


class ComponentLoader:
    """组件动态加载器

    负责根据组件名称动态加载和实例化组件。
    """

    def __init__(self, registry):
        """初始化加载器

        Args:
            registry: 组件注册中心实例
        """
        self.registry = registry
        self._module_cache = {}
        self._class_cache = {}

    def load_model(self, name: str, **kwargs) -> nn.Module:
        """加载模型

        Args:
            name: 模型名称
            **kwargs: 模型参数

        Returns:
            nn.Module: 模型实例

        Raises:
            ValueError: 模型不存在时抛出
        """
        model_info = self.registry.get('models', name)
        if not model_info:
            raise ValueError(f"Model not found: {name}")

        # 动态导入模块
        module_path = self._get_module_path(model_info)
        module = self._import_module(module_path)

        # 获取类
        model_class = getattr(module, model_info.name)

        # 验证参数
        self._validate_params(model_class, kwargs)

        # 实例化
        return model_class(**kwargs)

    def load_loss(self, name: str, **kwargs) -> nn.Module:
        """加载损失函数

        Args:
            name: 损失函数名称
            **kwargs: 损失函数参数

        Returns:
            nn.Module: 损失函数实例

        Raises:
            ValueError: 损失函数不存在时抛出
        """
        loss_info = self.registry.get('losses', name)
        if not loss_info:
            raise ValueError(f"Loss not found: {name}")

        module_path = self._get_module_path(loss_info)
        module = self._import_module(module_path)
        loss_class = getattr(module, loss_info.name)

        return loss_class(**kwargs)

    def _import_module(self, module_path: str):
        """动态导入模块（带缓存）

        Args:
            module_path: 模块路径

        Returns:
            导入的模块
        """
        if module_path in self._module_cache:
            return self._module_cache[module_path]

        module = importlib.import_module(module_path)
        self._module_cache[module_path] = module

        return module

    def _get_module_path(self, component_info) -> str:
        """获取模块路径

        Args:
            component_info: 组件信息

        Returns:
            str: 模块路径
        """
        # 这里需要根据实际文件结构构建模块路径
        # 示例: library.models.vision.resnet.model
        return f"library.{component_info.category}.{component_info.subcategory}.{component_info.name.lower()}"

    def _validate_params(self, component_class, params: Dict):
        """验证参数

        Args:
            component_class: 组件类
            params: 参数字典

        Raises:
            ValueError: 缺少必需参数时抛出
            TypeError: 参数类型错误时抛出
        """
        required = component_class.get_required_params()

        for param_name, param_type in required.items():
            if param_name not in params:
                raise ValueError(f"Missing required parameter: {param_name}")

            if not isinstance(params[param_name], param_type):
                raise TypeError(
                    f"Parameter {param_name} should be {param_type}, "
                    f"got {type(params[param_name])}"
                )
