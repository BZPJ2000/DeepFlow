# DeepFlow 实现计划

## 第三部分：分阶段实现流程

---

## 3.1 实施路线图

### 总体时间规划

```
阶段 0: 准备工作 (前置)
  ├── 代码备份
  ├── 环境准备
  └── 依赖检查

阶段 1: 项目清理 (基础)
  ├── 删除冗余文件
  ├── 更新 .gitignore
  └── 重组目录结构

阶段 2: 核心框架开发 (关键)
  ├── 自动发现引擎
  ├── 组件注册中心
  ├── 动态加载器
  └── 配置管理

阶段 3: 组件库迁移 (重要)
  ├── 定义基类接口
  ├── 迁移模型库
  ├── 迁移损失函数
  ├── 迁移评估指标
  └── 迁移优化器

阶段 4: UI 重构 (用户体验)
  ├── 重构页面结构
  ├── 集成新框架
  └── 优化交互流程

阶段 5: 测试与优化 (质量保证)
  ├── 单元测试
  ├── 集成测试
  └── 性能优化

阶段 6: 文档与部署 (交付)
  ├── 编写文档
  ├── 示例项目
  └── 部署指南
```

---

## 3.2 阶段 0: 准备工作

### 步骤 0.1: 代码备份

```bash
# 创建备份分支
git checkout -b backup-$(date +%Y%m%d)
git add .
git commit -m "Backup before refactoring - $(date +%Y-%m-%d)"

# 推送备份到远程
git push origin backup-$(date +%Y%m%d)

# 创建重构分支
git checkout -b refactor-v2
```

### 步骤 0.2: 环境准备

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 验证环境
python env_safety_check.py
```

### 步骤 0.3: 依赖检查

创建 `scripts/check_dependencies.py`:
```python
import importlib
import sys

REQUIRED_PACKAGES = [
    'torch', 'streamlit', 'numpy', 'pandas',
    'scikit-learn', 'opencv-python', 'plotly'
]

def check_dependencies():
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg.replace('-', '_'))
            print(f"✓ {pkg}")
        except ImportError:
            missing.append(pkg)
            print(f"✗ {pkg}")

    if missing:
        print(f"\n缺少依赖: {', '.join(missing)}")
        sys.exit(1)
    else:
        print("\n所有依赖已安装")

if __name__ == "__main__":
    check_dependencies()
```

---

## 3.3 阶段 1: 项目清理

### 步骤 1.1: 删除冗余文件

```bash
# 删除临时目录
rm -rf .trae/ try/ tmp/ streamlit_example-main/

# 删除 IDE 配置
rm -rf .idea/

# 删除 Python 缓存
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete

# 删除旧的主入口
rm app.py
```

### 步骤 1.2: 创建新目录结构

```bash
# 创建核心目录
mkdir -p deepflow/{core,components,utils,training,api}
mkdir -p ui/{pages,components,static/{css,images}}
mkdir -p library/{models,losses,metrics,optimizers}
mkdir -p configs/experiments
mkdir -p data/{raw,processed,samples}
mkdir -p outputs/{logs,checkpoints,results,visualizations}
mkdir -p tests
mkdir -p docs
mkdir -p scripts

# 创建 __init__.py 文件
touch deepflow/__init__.py
touch deepflow/core/__init__.py
touch deepflow/components/__init__.py
touch deepflow/utils/__init__.py
touch deepflow/training/__init__.py
touch deepflow/api/__init__.py
touch ui/__init__.py
touch ui/pages/__init__.py
touch ui/components/__init__.py
touch tests/__init__.py
```

### 步骤 1.3: 更新 .gitignore

```bash
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo
.DS_Store

# Project specific
outputs/
data/raw/
data/processed/
*.h5
*.pth
*.ckpt
*.pkl

# Temporary
tmp/
.trae/
try/
cached_projects/

# Logs
*.log
logs/

# Environment
.env
.env.local
EOF
```

---

## 3.4 阶段 2: 核心框架开发

### 步骤 2.1: 组件基类定义

创建 `deepflow/components/base_component.py`:
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ComponentMetadata:
    """组件元数据"""
    name: str
    category: str
    subcategory: str
    description: str
    author: Optional[str] = None
    version: Optional[str] = "1.0.0"
    tags: Optional[list] = None

class BaseComponent(ABC):
    """组件基类"""

    @classmethod
    @abstractmethod
    def get_metadata(cls) -> ComponentMetadata:
        """返回组件元数据"""
        pass

    @classmethod
    @abstractmethod
    def get_required_params(cls) -> Dict[str, type]:
        """返回必需参数"""
        pass

    @classmethod
    def get_optional_params(cls) -> Dict[str, Any]:
        """返回可选参数及默认值"""
        return {}
```

### 步骤 2.2: 模型基类

创建 `deepflow/components/base_model.py`:
```python
import torch.nn as nn
from typing import Dict, Any
from .base_component import BaseComponent, ComponentMetadata

class BaseModel(nn.Module, BaseComponent):
    """模型基类"""

    def __init__(self, **kwargs):
        super().__init__()
        self.config = kwargs

    @abstractmethod
    def forward(self, x):
        """前向传播"""
        pass

    def get_num_parameters(self) -> int:
        """计算参数量"""
        return sum(p.numel() for p in self.parameters())

    def get_model_size(self) -> float:
        """计算模型大小 (MB)"""
        param_size = sum(p.nelement() * p.element_size()
                        for p in self.parameters())
        buffer_size = sum(b.nelement() * b.element_size()
                         for b in self.buffers())
        return (param_size + buffer_size) / 1024 / 1024
```

### 步骤 2.3: 自动发现引擎

创建 `deepflow/core/discovery.py`:
```python
import os
import ast
import importlib.util
from pathlib import Path
from typing import List, Dict, Type
from ..components.base_component import BaseComponent, ComponentMetadata

class ComponentDiscovery:
    """组件自动发现引擎"""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.discovered_components = {}

    def discover_all(self) -> Dict[str, List[ComponentMetadata]]:
        """发现所有组件"""
        results = {
            'models': self.discover_in_path(self.base_path / 'models'),
            'losses': self.discover_in_path(self.base_path / 'losses'),
            'metrics': self.discover_in_path(self.base_path / 'metrics'),
            'optimizers': self.discover_in_path(self.base_path / 'optimizers'),
        }
        return results

    def discover_in_path(self, path: Path) -> List[ComponentMetadata]:
        """在指定路径发现组件"""
        components = []

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
        """从文件中提取组件"""
        components = []

        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # 检查是否继承自 BaseComponent
                if self._is_component_class(node):
                    metadata = self._extract_metadata(node, file_path)
                    if metadata:
                        components.append(metadata)

        return components

    def _is_component_class(self, node: ast.ClassDef) -> bool:
        """检查是否为组件类"""
        for base in node.bases:
            if isinstance(base, ast.Name):
                if base.id in ['BaseModel', 'BaseLoss',
                              'BaseMetric', 'BaseOptimizer']:
                    return True
        return False

    def _extract_metadata(self, node: ast.ClassDef,
                         file_path: Path) -> Optional[ComponentMetadata]:
        """提取组件元数据"""
        # 这里简化处理，实际应该解析 get_metadata 方法
        return ComponentMetadata(
            name=node.name,
            category=file_path.parent.parent.name,
            subcategory=file_path.parent.name,
            description=ast.get_docstring(node) or "",
        )
```

---

## 3.5 阶段 2 续: 注册中心与加载器

### 步骤 2.4: 组件注册中心

创建 `deepflow/core/registry.py`:
```python
from typing import Dict, List, Optional, Type
from ..components.base_component import ComponentMetadata

class ComponentRegistry:
    """组件注册中心 (单例)"""

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

    def register(self, component_type: str,
                name: str, metadata: ComponentMetadata):
        """注册组件"""
        if component_type not in self._registry:
            raise ValueError(f"Unknown component type: {component_type}")

        self._registry[component_type][name] = metadata

    def get(self, component_type: str, name: str) -> Optional[ComponentMetadata]:
        """获取组件"""
        return self._registry.get(component_type, {}).get(name)

    def list(self, component_type: str,
            category: Optional[str] = None) -> List[ComponentMetadata]:
        """列出组件"""
        components = self._registry.get(component_type, {}).values()

        if category:
            components = [c for c in components if c.category == category]

        return list(components)

    def search(self, query: str) -> List[ComponentMetadata]:
        """搜索组件"""
        results = []
        for comp_type in self._registry.values():
            for metadata in comp_type.values():
                if query.lower() in metadata.name.lower() or \
                   query.lower() in metadata.description.lower():
                    results.append(metadata)
        return results
```

下一段将继续实现加载器和配置管理。
