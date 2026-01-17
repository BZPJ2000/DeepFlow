# DeepFlow 开发者指南

## 📖 目录

1. [架构概览](#架构概览)
2. [核心模块](#核心模块)
3. [添加新组件](#添加新组件)
4. [代码规范](#代码规范)
5. [测试指南](#测试指南)

---

## 🏗️ 架构概览

### 设计原则

1. **模块化设计** - 高内聚、低耦合
2. **自动发现** - 零配置的组件加载
3. **类型安全** - 完整的类型注解
4. **易于扩展** - 简单的组件添加流程

### 目录结构

```
DeepFlow/
├── deepflow/              # 核心框架
│   ├── core/             # 核心功能
│   │   ├── discovery.py  # 组件自动发现
│   │   ├── registry.py   # 组件注册中心
│   │   ├── loader.py     # 动态加载器
│   │   └── config.py     # 配置管理
│   ├── components/       # 组件基类
│   │   ├── base_component.py
│   │   ├── base_model.py
│   │   ├── base_loss.py
│   │   └── base_metric.py
│   ├── training/         # 训练模块
│   │   └── trainer.py
│   ├── utils/            # 工具函数
│   └── api/              # API 接口
│       └── experiment.py
├── library/              # 组件库
│   ├── models/          # 模型实现
│   ├── losses/          # 损失函数
│   ├── metrics/         # 评估指标
│   └── optimizers/      # 优化器
├── ui/                  # 用户界面
│   └── pages/          # Streamlit 页面
└── tests/              # 测试代码
```

---

## 🔧 核心模块

### 1. 组件自动发现 (discovery.py)

**功能：** 扫描 library/ 目录，自动识别所有组件

**工作流程：**
```
扫描目录 → 解析 AST → 提取元数据 → 验证接口 → 返回组件列表
```

**关键类：**
```python
class ComponentDiscovery:
    def discover_all() -> Dict[str, List[ComponentMetadata]]
    def discover_in_path(path: Path) -> List[ComponentMetadata]
```

### 2. 组件注册中心 (registry.py)

**功能：** 统一管理所有已发现的组件（单例模式）

**关键方法：**
```python
class ComponentRegistry:
    def register(component_type, name, metadata)
    def get(component_type, name) -> ComponentMetadata
    def list(component_type, category) -> List[ComponentMetadata]
    def search(query) -> List[ComponentMetadata]
```

### 3. 动态加载器 (loader.py)

**功能：** 根据名称动态加载和实例化组件

**关键方法：**
```python
class ComponentLoader:
    def load_model(name, **kwargs) -> nn.Module
    def load_loss(name, **kwargs) -> nn.Module
```

**特点：**
- 模块缓存，提升性能
- 参数验证，避免错误
- 按需加载，节省内存

---

## ➕ 添加新组件

### 添加新模型

**步骤 1：创建模型文件**

在 `library/models/` 下创建文件：
```
library/models/vision/classification/my_model.py
```

**步骤 2：实现模型类**

```python
from deepflow.components.base_model import BaseModel, ComponentMetadata
import torch.nn as nn

class MyModel(BaseModel):
    """我的自定义模型"""

    @classmethod
    def get_metadata(cls) -> ComponentMetadata:
        return ComponentMetadata(
            name="MyModel",
            category="vision",
            subcategory="classification",
            description="我的自定义图像分类模型",
            author="Your Name",
            version="1.0.0",
            tags=["classification", "custom"]
        )

    @classmethod
    def get_required_params(cls):
        return {
            'num_classes': int,
        }

    @classmethod
    def get_optional_params(cls):
        return {
            'dropout': 0.5,
        }

    def __init__(self, num_classes: int, dropout: float = 0.5):
        super().__init__(num_classes=num_classes, dropout=dropout)

        # 定义网络层
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64 * 16 * 16, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
```

**步骤 3：重启应用**

模型会自动被发现和注册，无需手动配置！

### 添加损失函数

```python
from deepflow.components.base_loss import BaseLoss, ComponentMetadata
import torch.nn as nn

class MyLoss(BaseLoss):
    """自定义损失函数"""

    @classmethod
    def get_metadata(cls) -> ComponentMetadata:
        return ComponentMetadata(
            name="MyLoss",
            category="vision",
            subcategory="classification",
            description="自定义损失函数",
            version="1.0.0"
        )

    @classmethod
    def get_required_params(cls):
        return {}

    @classmethod
    def get_optional_params(cls):
        return {'weight': 1.0}

    def __init__(self, weight: float = 1.0):
        super().__init__(weight=weight)
        self.weight = weight

    def forward(self, pred, target):
        # 实现损失计算
        loss = nn.functional.cross_entropy(pred, target)
        return loss * self.weight
```

---

## 📝 代码规范

### Python 风格

遵循 PEP 8 规范：

```python
# 导入顺序
import os                          # 标准库
import sys

import torch                       # 第三方库
import numpy as np

from deepflow.core import loader   # 本地模块

# 命名规范
class MyModel:                     # 类名：PascalCase
    def train_model(self):         # 方法名：snake_case
        MAX_EPOCHS = 100           # 常量：UPPER_CASE
        learning_rate = 0.001      # 变量：snake_case
```

### 类型注解

**强制使用类型注解：**

```python
from typing import Dict, List, Optional, Tuple

def process_data(
    data: List[Dict[str, any]],
    config: Optional[Dict] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """处理数据"""
    pass
```

### 文档字符串

**使用 Google 风格：**

```python
def train_model(
    model: nn.Module,
    epochs: int = 10
) -> Dict[str, List[float]]:
    """训练深度学习模型

    Args:
        model: PyTorch 模型实例
        epochs: 训练轮数，默认 10

    Returns:
        包含训练历史的字典

    Raises:
        RuntimeError: CUDA 不可用时

    Example:
        >>> model = ResNet50()
        >>> history = train_model(model, epochs=20)
    """
    pass
```

---

## 🧪 测试指南

### 运行测试

```bash
# 运行所有测试
python tests/test_core.py

# 运行特定测试
python -m pytest tests/test_discovery.py -v
```

### 编写测试

```python
import unittest
from deepflow.core.discovery import ComponentDiscovery

class TestDiscovery(unittest.TestCase):
    def setUp(self):
        self.discovery = ComponentDiscovery('library')

    def test_discover_models(self):
        models = self.discovery.discover_in_path(
            Path('library/models')
        )
        self.assertGreater(len(models), 0)
```

---

**更多内容请参考用户指南和 API 文档。**
