# DeepFlow 测试与性能优化指南

## 第七部分：测试与性能优化

---

## 7.1 测试策略

### 7.1.1 单元测试

**测试组件发现:**
```python
# tests/test_discovery.py

import unittest
from pathlib import Path
from deepflow.core.discovery import ComponentDiscovery

class TestComponentDiscovery(unittest.TestCase):
    """测试组件发现功能"""

    def setUp(self):
        """测试前准备"""
        self.discovery = ComponentDiscovery('library')

    def test_discover_models(self):
        """测试模型发现"""
        models = self.discovery.discover_in_path(
            Path('library/models')
        )
        self.assertGreater(len(models), 0)
        self.assertTrue(all(m.name for m in models))

    def test_discover_losses(self):
        """测试损失函数发现"""
        losses = self.discovery.discover_in_path(
            Path('library/losses')
        )
        self.assertGreater(len(losses), 0)

    def test_metadata_extraction(self):
        """测试元数据提取"""
        models = self.discovery.discover_in_path(
            Path('library/models')
        )
        for model in models:
            self.assertIsNotNone(model.name)
            self.assertIsNotNone(model.category)
            self.assertIsNotNone(model.description)
```

**测试组件加载:**
```python
# tests/test_loader.py

import unittest
import torch.nn as nn
from deepflow.core.loader import ComponentLoader
from deepflow.core.registry import ComponentRegistry

class TestComponentLoader(unittest.TestCase):
    """测试组件加载功能"""

    def setUp(self):
        """测试前准备"""
        self.registry = ComponentRegistry()
        self.loader = ComponentLoader(self.registry)

    def test_load_model(self):
        """测试模型加载"""
        model = self.loader.load_model(
            'ResNet50',
            num_classes=10
        )
        self.assertIsInstance(model, nn.Module)

    def test_load_with_invalid_params(self):
        """测试无效参数"""
        with self.assertRaises(ValueError):
            self.loader.load_model('ResNet50')  # 缺少必需参数

    def test_load_nonexistent_model(self):
        """测试加载不存在的模型"""
        with self.assertRaises(ValueError):
            self.loader.load_model('NonExistentModel')
```

### 7.1.2 集成测试

```python
# tests/test_integration.py

import unittest
from deepflow.api.experiment import ExperimentAPI

class TestExperimentWorkflow(unittest.TestCase):
    """测试完整实验流程"""

    def test_full_workflow(self):
        """测试完整工作流"""
        api = ExperimentAPI()

        # 1. 获取可用模型
        models = api.get_available_models(
            category='vision',
            subcategory='classification'
        )
        self.assertGreater(len(models), 0)

        # 2. 创建实验配置
        config = {
            'model': 'ResNet50',
            'model_params': {'num_classes': 10},
            'loss': 'CrossEntropyLoss',
            'optimizer': 'Adam',
            'optimizer_params': {'lr': 0.001},
            'epochs': 2,
            'batch_size': 16
        }

        # 3. 初始化实验
        experiment = api.create_experiment(config)
        self.assertIsNotNone(experiment)

        # 4. 验证组件加载
        self.assertIsNotNone(experiment.model)
        self.assertIsNotNone(experiment.loss_fn)
        self.assertIsNotNone(experiment.optimizer)
```

### 7.1.3 运行测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行特定测试
python -m pytest tests/test_discovery.py -v

# 生成覆盖率报告
python -m pytest tests/ --cov=deepflow --cov-report=html

# 查看覆盖率
open htmlcov/index.html
```

---

## 7.2 性能优化

### 7.2.1 组件发现优化

**使用缓存减少扫描:**
```python
# deepflow/core/discovery.py (优化版)

class ComponentDiscovery:
    """组件发现引擎 (优化版)"""

    def __init__(self, base_path: str, use_cache: bool = True):
        self.base_path = Path(base_path)
        self.use_cache = use_cache
        self.cache = DiscoveryCache() if use_cache else None

    def discover_all(self) -> Dict[str, List[ComponentInfo]]:
        """发现所有组件 (带缓存)"""

        # 检查缓存
        if self.use_cache and not self._needs_rescan():
            return self.cache.get_all_components()

        # 执行扫描
        results = self._scan_all_components()

        # 更新缓存
        if self.use_cache:
            self.cache.update_all(results)
            self.cache.save()

        return results

    def _needs_rescan(self) -> bool:
        """检查是否需要重新扫描"""
        for py_file in self.base_path.rglob("*.py"):
            if self.cache.is_file_changed(py_file):
                return True
        return False
```

### 7.2.2 模型加载优化

**延迟加载和模块缓存:**
```python
# deepflow/core/loader.py (优化版)

class ComponentLoader:
    """组件加载器 (优化版)"""

    def __init__(self, registry):
        self.registry = registry
        self._module_cache = {}
        self._class_cache = {}

    def load_model(self, name: str, **kwargs) -> nn.Module:
        """加载模型 (带缓存)"""

        # 检查类缓存
        cache_key = f"model_{name}"
        if cache_key in self._class_cache:
            model_class = self._class_cache[cache_key]
        else:
            # 获取信息并导入
            model_info = self.registry.get('models', name)
            module = self._get_cached_module(model_info.module_path)
            model_class = getattr(module, model_info.name)
            self._class_cache[cache_key] = model_class

        # 实例化
        return model_class(**kwargs)

    def _get_cached_module(self, module_path: str):
        """获取缓存的模块"""
        if module_path not in self._module_cache:
            self._module_cache[module_path] = importlib.import_module(
                module_path
            )
        return self._module_cache[module_path]
```

### 7.2.3 数据加载优化

```python
# deepflow/utils/data_utils.py

from torch.utils.data import DataLoader

def create_optimized_dataloader(
    dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2
) -> DataLoader:
    """创建优化的数据加载器"""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,      # 多进程加载
        pin_memory=pin_memory,        # 固定内存
        prefetch_factor=prefetch_factor,  # 预取因子
        persistent_workers=True       # 保持工作进程
    )
```

---

## 7.3 内存优化

### 7.3.1 梯度累积

```python
# deepflow/training/trainer.py

class Trainer:
    """训练器 (支持梯度累积)"""

    def _train_epoch_with_accumulation(
        self,
        train_loader: DataLoader,
        accumulation_steps: int = 4
    ):
        """使用梯度累积训练"""

        self.model.train()
        self.optimizer.zero_grad()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # 前向传播
            output = self.model(data)
            loss = self.loss_fn(output, target)

            # 归一化损失
            loss = loss / accumulation_steps

            # 反向传播
            loss.backward()

            # 每 N 步更新一次
            if (batch_idx + 1) % accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
```

### 7.3.2 混合精度训练

```python
# deepflow/training/trainer.py

import torch
from torch.cuda.amp import autocast, GradScaler

class MixedPrecisionTrainer(Trainer):
    """混合精度训练器"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaler = GradScaler()

    def _train_epoch(self, train_loader: DataLoader):
        """使用混合精度训练"""

        self.model.train()
        total_loss = 0

        for data, target in train_loader:
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            # 使用自动混合精度
            with autocast():
                output = self.model(data)
                loss = self.loss_fn(output, target)

            # 缩放损失并反向传播
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

        return total_loss / len(train_loader)
```

---

## 7.4 性能监控

### 7.4.1 训练监控

```python
# deepflow/training/callbacks.py

import time
from typing import Dict

class PerformanceMonitor:
    """性能监控器"""

    def __init__(self):
        self.metrics = {
            'epoch_time': [],
            'batch_time': [],
            'data_loading_time': [],
            'forward_time': [],
            'backward_time': []
        }
        self.start_time = None

    def on_epoch_start(self):
        """Epoch 开始"""
        self.start_time = time.time()

    def on_epoch_end(self):
        """Epoch 结束"""
        epoch_time = time.time() - self.start_time
        self.metrics['epoch_time'].append(epoch_time)

    def get_summary(self) -> Dict:
        """获取性能摘要"""
        return {
            'avg_epoch_time': sum(self.metrics['epoch_time']) /
                            len(self.metrics['epoch_time']),
            'total_time': sum(self.metrics['epoch_time'])
        }
```

---

## 7.5 基准测试

```python
# scripts/benchmark.py

import time
import torch
from deepflow.core.loader import ComponentLoader
from deepflow.core.registry import ComponentRegistry

def benchmark_model_loading():
    """基准测试模型加载速度"""

    registry = ComponentRegistry()
    loader = ComponentLoader(registry)

    models = ['ResNet50', 'VGG16', 'MobileNetV2']
    results = {}

    for model_name in models:
        start = time.time()
        model = loader.load_model(model_name, num_classes=10)
        load_time = time.time() - start

        results[model_name] = {
            'load_time': load_time,
            'num_params': sum(p.numel() for p in model.parameters())
        }

    return results

if __name__ == "__main__":
    results = benchmark_model_loading()
    for name, metrics in results.items():
        print(f"{name}:")
        print(f"  加载时间: {metrics['load_time']:.4f}s")
        print(f"  参数量: {metrics['num_params']:,}")
```

---

下一段将创建快速开始指南。
