# DeepFlow 维护与扩展指南

## 第六部分：项目维护与扩展

---

## 6.1 代码规范

### 6.1.1 Python 代码风格

**遵循 PEP 8 规范:**
```python
# 导入顺序
import os                          # 标准库
import sys

import torch                       # 第三方库
import numpy as np

from deepflow.core import loader   # 本地模块
from deepflow.utils import logger

# 类定义
class ModelLoader:
    """模型加载器

    详细描述模型加载器的功能和用途。

    Attributes:
        registry: 组件注册中心
        cache: 缓存字典
    """

    def __init__(self, registry):
        self.registry = registry
        self._cache = {}

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
        pass
```

### 6.1.2 类型注解

**强制使用类型注解:**
```python
from typing import Dict, List, Optional, Union, Tuple

def process_data(
    data: List[Dict[str, any]],
    config: Optional[Dict] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """处理数据"""
    pass

class DataProcessor:
    def __init__(self, config: Dict[str, any]):
        self.config: Dict[str, any] = config
        self.results: List[np.ndarray] = []
```

### 6.1.3 文档字符串

**使用 Google 风格文档字符串:**
```python
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 10,
    device: str = 'cuda'
) -> Dict[str, List[float]]:
    """训练深度学习模型

    使用给定的数据加载器训练模型，支持 GPU 加速。

    Args:
        model: PyTorch 模型实例
        train_loader: 训练数据加载器
        epochs: 训练轮数，默认 10
        device: 训练设备，'cuda' 或 'cpu'

    Returns:
        包含训练历史的字典，格式为:
        {
            'loss': [epoch1_loss, epoch2_loss, ...],
            'accuracy': [epoch1_acc, epoch2_acc, ...]
        }

    Raises:
        RuntimeError: CUDA 不可用但指定了 cuda 设备
        ValueError: epochs 小于 1

    Example:
        >>> model = ResNet50()
        >>> loader = DataLoader(dataset, batch_size=32)
        >>> history = train_model(model, loader, epochs=20)
        >>> print(history['loss'][-1])
        0.234
    """
    pass
```

---

## 6.2 添加新组件

### 6.2.1 添加新模型

**步骤 1: 创建模型文件**
```bash
# 在对应类别下创建目录
mkdir -p library/models/vision/my_new_model
cd library/models/vision/my_new_model
```

**步骤 2: 实现模型类**
```python
# library/models/vision/my_new_model/model.py

import torch.nn as nn
from deepflow.components.base_model import BaseModel, ComponentMetadata

class MyNewModel(BaseModel):
    """我的新模型

    详细描述模型的功能、特点和适用场景。
    """

    @classmethod
    def get_metadata(cls) -> ComponentMetadata:
        """返回模型元数据"""
        return ComponentMetadata(
            name="MyNewModel",
            category="vision",
            subcategory="classification",
            description="一个用于图像分类的新模型",
            author="Your Name",
            version="1.0.0",
            tags=["classification", "lightweight"]
        )

    @classmethod
    def get_required_params(cls) -> Dict[str, type]:
        """返回必需参数"""
        return {
            'num_classes': int,
            'input_channels': int,
        }

    @classmethod
    def get_optional_params(cls) -> Dict[str, any]:
        """返回可选参数及默认值"""
        return {
            'dropout': 0.5,
            'activation': 'relu',
        }

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 3,
        dropout: float = 0.5,
        activation: str = 'relu'
    ):
        super().__init__(
            num_classes=num_classes,
            input_channels=input_channels,
            dropout=dropout,
            activation=activation
        )

        # 定义网络层
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # ... 更多层
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """前向传播"""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
```

**步骤 3: 添加 README**
```markdown
# MyNewModel

## 简介
简要描述模型的功能和特点。

## 参数
- `num_classes`: 分类数量
- `input_channels`: 输入通道数，默认 3
- `dropout`: Dropout 比率，默认 0.5

## 使用示例
\`\`\`python
from deepflow.core.loader import ComponentLoader

loader = ComponentLoader(registry)
model = loader.load_model(
    'MyNewModel',
    num_classes=10,
    input_channels=3
)
\`\`\`

## 性能指标
- 参数量: 2.3M
- 推理速度: 50 FPS (GPU)
- ImageNet Top-1: 75.2%
```

**步骤 4: 自动发现**
```python
# 重启应用，模型会自动被发现
# 或手动触发发现
from deepflow.core.discovery import ComponentDiscovery

discovery = ComponentDiscovery('library')
discovery.discover_all()
```

---

### 6.2.2 添加新损失函数

```python
# library/losses/vision/my_loss.py

import torch
import torch.nn as nn
from deepflow.components.base_loss import BaseLoss, ComponentMetadata

class MyCustomLoss(BaseLoss):
    """自定义损失函数"""

    @classmethod
    def get_metadata(cls) -> ComponentMetadata:
        return ComponentMetadata(
            name="MyCustomLoss",
            category="vision",
            subcategory="classification",
            description="结合交叉熵和焦点损失的自定义损失",
            version="1.0.0"
        )

    @classmethod
    def get_required_params(cls) -> Dict[str, type]:
        return {}

    @classmethod
    def get_optional_params(cls) -> Dict[str, any]:
        return {
            'alpha': 0.25,
            'gamma': 2.0,
        }

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__(alpha=alpha, gamma=gamma)
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        """计算损失"""
        ce = self.ce_loss(pred, target)
        pt = torch.exp(-ce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce
        return focal_loss
```

---

## 6.3 配置管理

### 6.3.1 配置文件结构

```yaml
# configs/default.yaml

# 应用配置
app:
  name: "DeepFlow"
  version: "2.0.0"
  debug: false

# 路径配置
paths:
  library: "library"
  data: "data"
  outputs: "outputs"
  cache: ".deepflow_cache.json"

# 组件发现配置
discovery:
  enabled: true
  cache_enabled: true
  scan_on_startup: true
  excluded_dirs:
    - "__pycache__"
    - ".git"
    - "tests"

# 训练默认配置
training:
  default_epochs: 10
  default_batch_size: 32
  default_device: "cuda"
  checkpoint_interval: 5
  early_stopping:
    enabled: true
    patience: 10
    min_delta: 0.001

# UI 配置
ui:
  theme: "light"
  page_icon: "🚀"
  layout: "wide"
```

### 6.3.2 配置加载

```python
# deepflow/core/config.py

import yaml
from pathlib import Path
from typing import Dict, Any

class Config:
    """配置管理器 (单例)"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """加载配置"""
        config_file = Path("configs/default.yaml")

        with open(config_file, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值

        支持点号分隔的嵌套键，如 'training.default_epochs'
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default

        return value if value is not None else default
```

---

## 6.4 日志管理

### 6.4.1 日志配置

```python
# deepflow/utils/logger.py

import logging
from pathlib import Path
from datetime import datetime

def setup_logger(name: str, log_dir: str = "outputs/logs") -> logging.Logger:
    """设置日志记录器"""

    # 创建日志目录
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 文件处理器
    log_file = log_path / f"{name}_{datetime.now():%Y%m%d}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)

    # 格式化
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
```

下一段将继续说明测试、性能优化和部署相关内容。
