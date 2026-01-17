# DeepFlow 模型选择与训练流程设计

## 第五部分：模型选择与训练流程

---

## 5.1 整体工作流程

```
用户启动应用
    ↓
1. 任务选择
   - 选择领域 (NLP/CV/GNN/RL)
   - 选择子任务
    ↓
2. 模型选择
   - 浏览可用模型
   - 查看模型详情
   - 选择模型架构
    ↓
3. 数据配置
   - 设置数据路径
   - 配置数据分割
   - 选择数据增强
    ↓
4. 训练配置
   - 选择损失函数
   - 选择评估指标
   - 选择优化器
   - 设置超参数
    ↓
5. 开始训练
   - 初始化组件
   - 执行训练循环
   - 实时监控
    ↓
6. 结果展示
   - 训练曲线
   - 评估指标
   - 模型保存
```

---

## 5.2 模型选择机制

### 5.2.1 模型浏览界面

```python
# ui/pages/2_model_selection.py

import streamlit as st
from deepflow.api.experiment import ExperimentAPI

def render_model_selection():
    """渲染模型选择页面"""

    st.title("🤖 模型选择")

    # 获取当前任务信息
    task_info = st.session_state.get('task_info')
    if not task_info:
        st.warning("请先选择任务")
        return

    # 获取可用模型
    api = ExperimentAPI()
    models = api.get_available_models(
        category=task_info['category'],
        subcategory=task_info['subcategory']
    )

    # 显示模型列表
    st.subheader(f"可用模型 ({len(models)})")

    # 筛选选项
    col1, col2 = st.columns(2)
    with col1:
        sort_by = st.selectbox(
            "排序方式",
            ["名称", "参数量", "推荐度"]
        )
    with col2:
        filter_tags = st.multiselect(
            "标签筛选",
            ["轻量级", "高精度", "实时", "预训练"]
        )

    # 模型卡片展示
    for model in models:
        render_model_card(model)
```

### 5.2.2 模型卡片组件

```python
# ui/components/model_card.py

import streamlit as st
from deepflow.utils.model_utils import ModelAnalyzer

def render_model_card(model_info):
    """渲染模型卡片"""

    with st.expander(f"📦 {model_info.name}", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("参数量", f"{model_info.num_params / 1e6:.2f}M")

        with col2:
            st.metric("模型大小", f"{model_info.size_mb:.2f} MB")

        with col3:
            st.metric("推荐度", "⭐" * model_info.rating)

        # 描述
        st.markdown(f"**描述:** {model_info.description}")

        # 标签
        if model_info.tags:
            st.markdown("**标签:** " + " ".join(
                [f"`{tag}`" for tag in model_info.tags]
            ))

        # 资源需求
        st.markdown("**资源需求:**")
        analyzer = ModelAnalyzer()
        requirements = analyzer.estimate_requirements(model_info)

        st.write(f"- 最小显存: {requirements['min_memory']} GB")
        st.write(f"- 推荐显存: {requirements['recommended_memory']} GB")
        st.write(f"- 训练时间估计: {requirements['training_time']}")

        # 选择按钮
        if st.button(f"选择 {model_info.name}", key=f"select_{model_info.name}"):
            st.session_state['selected_model'] = model_info
            st.success(f"已选择模型: {model_info.name}")
```

---

## 5.3 动态加载机制

### 5.3.1 组件加载器

```python
# deepflow/core/loader.py

import importlib
from pathlib import Path
from typing import Any, Dict, Optional
import torch.nn as nn

class ComponentLoader:
    """组件动态加载器"""

    def __init__(self, registry):
        self.registry = registry
        self._cache = {}

    def load_model(self, name: str, **kwargs) -> nn.Module:
        """加载模型"""

        # 从注册中心获取信息
        model_info = self.registry.get('models', name)
        if not model_info:
            raise ValueError(f"Model not found: {name}")

        # 动态导入模块
        module = self._import_module(model_info.module_path)

        # 获取类
        model_class = getattr(module, model_info.name)

        # 验证参数
        self._validate_params(model_class, kwargs)

        # 实例化
        model = model_class(**kwargs)

        return model

    def load_loss(self, name: str, **kwargs) -> nn.Module:
        """加载损失函数"""
        loss_info = self.registry.get('losses', name)
        if not loss_info:
            raise ValueError(f"Loss not found: {name}")

        module = self._import_module(loss_info.module_path)
        loss_class = getattr(module, loss_info.name)

        return loss_class(**kwargs)

    def load_metric(self, name: str, **kwargs):
        """加载评估指标"""
        metric_info = self.registry.get('metrics', name)
        if not metric_info:
            raise ValueError(f"Metric not found: {name}")

        module = self._import_module(metric_info.module_path)
        metric_class = getattr(module, metric_info.name)

        return metric_class(**kwargs)

    def _import_module(self, module_path: str):
        """动态导入模块"""
        if module_path in self._cache:
            return self._cache[module_path]

        module = importlib.import_module(module_path)
        self._cache[module_path] = module

        return module

    def _validate_params(self, component_class, params: Dict):
        """验证参数"""
        required = component_class.get_required_params()

        for param_name, param_type in required.items():
            if param_name not in params:
                raise ValueError(f"Missing required parameter: {param_name}")

            if not isinstance(params[param_name], param_type):
                raise TypeError(
                    f"Parameter {param_name} should be {param_type}, "
                    f"got {type(params[param_name])}"
                )
```

---

## 5.4 训练流程设计

### 5.4.1 训练器核心

```python
# deepflow/training/trainer.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Callable
from pathlib import Path

class Trainer:
    """训练器"""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: Dict[str, Callable],
        device: str = 'cuda',
        callbacks: Optional[List] = None
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metrics = metrics
        self.device = device
        self.callbacks = callbacks or []

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': {},
            'val_metrics': {}
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        save_dir: Optional[Path] = None
    ):
        """训练模型"""

        for epoch in range(epochs):
            # 训练阶段
            train_loss, train_metrics = self._train_epoch(train_loader)

            # 验证阶段
            if val_loader:
                val_loss, val_metrics = self._validate_epoch(val_loader)
            else:
                val_loss, val_metrics = None, {}

            # 记录历史
            self._update_history(epoch, train_loss, val_loss,
                               train_metrics, val_metrics)

            # 执行回调
            self._execute_callbacks('on_epoch_end', epoch)

            # 保存检查点
            if save_dir and (epoch + 1) % 5 == 0:
                self._save_checkpoint(save_dir, epoch)

    def _train_epoch(self, train_loader: DataLoader):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        metric_values = {name: 0 for name in self.metrics}

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            # 记录损失
            total_loss += loss.item()

            # 计算指标
            for name, metric_fn in self.metrics.items():
                metric_values[name] += metric_fn(output, target).item()

        # 平均值
        avg_loss = total_loss / len(train_loader)
        avg_metrics = {
            name: value / len(train_loader)
            for name, value in metric_values.items()
        }

        return avg_loss, avg_metrics
```

下一段将继续说明验证、回调和配置管理机制。

