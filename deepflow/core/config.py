"""配置管理模块

管理应用配置和实验配置。
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """配置管理器 (单例模式)

    统一管理应用配置。
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._config = {}
            cls._instance._load_default_config()
        return cls._instance

    def _load_default_config(self):
        """加载默认配置"""
        self._config = {
            'app': {
                'name': 'DeepFlow',
                'version': '2.0.0',
                'debug': False
            },
            'paths': {
                'library': 'library',
                'data': 'data',
                'outputs': 'outputs',
                'cache': '.deepflow_cache.json'
            },
            'discovery': {
                'enabled': True,
                'cache_enabled': True,
                'scan_on_startup': True
            },
            'training': {
                'default_epochs': 10,
                'default_batch_size': 32,
                'default_device': 'cuda'
            }
        }

    def load_from_file(self, config_file: str):
        """从文件加载配置

        Args:
            config_file: 配置文件路径
        """
        config_path = Path(config_file)
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f)
                self._config.update(file_config)

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值

        支持点号分隔的嵌套键，如 'training.default_epochs'

        Args:
            key: 配置键
            default: 默认值

        Returns:
            Any: 配置值
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default

        return value if value is not None else default

    def set(self, key: str, value: Any):
        """设置配置值

        Args:
            key: 配置键
            value: 配置值
        """
        keys = key.split('.')
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value
