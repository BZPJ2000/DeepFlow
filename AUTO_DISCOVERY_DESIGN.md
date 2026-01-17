# DeepFlow 函数自动感知机制设计

## 第四部分：自动发现机制详细设计

---

## 4.1 设计目标

### 核心需求
1. **零配置** - 添加新组件无需手动注册
2. **智能识别** - 自动识别符合规范的类和函数
3. **元数据提取** - 自动提取组件信息（名称、参数、描述）
4. **类型验证** - 确保组件符合接口规范
5. **性能优化** - 缓存发现结果，避免重复扫描

---

## 4.2 发现流程

```
启动应用
    ↓
扫描组件库目录
    ↓
遍历 Python 文件
    ↓
解析 AST 语法树
    ↓
识别类/函数定义
    ↓
检查基类继承
    ↓
提取元数据
    ↓
验证接口规范
    ↓
注册到中心
    ↓
缓存结果
```

---

## 4.3 核心实现

### 4.3.1 组件扫描器

```python
# deepflow/core/scanner.py

import os
import ast
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class ComponentInfo:
    """组件信息"""
    name: str                    # 类名/函数名
    module_path: str             # 模块路径
    file_path: str               # 文件路径
    component_type: str          # 组件类型
    category: str                # 类别
    subcategory: str             # 子类别
    description: str             # 描述
    parameters: Dict             # 参数定义
    base_classes: List[str]      # 基类列表
    decorators: List[str]        # 装饰器列表

class ComponentScanner:
    """组件扫描器"""

    def __init__(self, library_path: str):
        self.library_path = Path(library_path)
        self.cache_file = Path(".deepflow_cache.json")

    def scan_all(self) -> Dict[str, List[ComponentInfo]]:
        """扫描所有组件"""
        results = {
            'models': self._scan_directory('models'),
            'losses': self._scan_directory('losses'),
            'metrics': self._scan_directory('metrics'),
            'optimizers': self._scan_directory('optimizers'),
        }
        return results

    def _scan_directory(self, component_type: str) -> List[ComponentInfo]:
        """扫描指定类型的组件目录"""
        components = []
        base_path = self.library_path / component_type

        if not base_path.exists():
            return components

        # 遍历所有 Python 文件
        for py_file in base_path.rglob("*.py"):
            if py_file.name.startswith('_'):
                continue

            try:
                file_components = self._parse_file(py_file, component_type)
                components.extend(file_components)
            except Exception as e:
                print(f"Warning: Failed to parse {py_file}: {e}")

        return components

    def _parse_file(self, file_path: Path,
                   component_type: str) -> List[ComponentInfo]:
        """解析单个文件"""
        components = []

        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                tree = ast.parse(f.read(), filename=str(file_path))
            except SyntaxError as e:
                print(f"Syntax error in {file_path}: {e}")
                return components

        # 提取类定义
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                info = self._extract_class_info(node, file_path, component_type)
                if info:
                    components.append(info)
            elif isinstance(node, ast.FunctionDef):
                info = self._extract_function_info(node, file_path, component_type)
                if info:
                    components.append(info)

        return components
```

---

### 4.3.2 AST 解析器

```python
# deepflow/core/ast_parser.py

import ast
from typing import Dict, List, Optional

class ASTParser:
    """AST 语法树解析器"""

    @staticmethod
    def extract_class_info(node: ast.ClassDef) -> Dict:
        """提取类信息"""
        return {
            'name': node.name,
            'docstring': ast.get_docstring(node),
            'base_classes': ASTParser._get_base_classes(node),
            'methods': ASTParser._get_methods(node),
            'decorators': ASTParser._get_decorators(node),
        }

    @staticmethod
    def _get_base_classes(node: ast.ClassDef) -> List[str]:
        """获取基类列表"""
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(f"{base.value.id}.{base.attr}")
        return bases

    @staticmethod
    def _get_methods(node: ast.ClassDef) -> Dict[str, Dict]:
        """获取方法信息"""
        methods = {}
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                methods[item.name] = {
                    'args': ASTParser._get_function_args(item),
                    'returns': ASTParser._get_return_type(item),
                    'docstring': ast.get_docstring(item),
                }
        return methods

    @staticmethod
    def _get_function_args(node: ast.FunctionDef) -> List[Dict]:
        """获取函数参数"""
        args = []
        for arg in node.args.args:
            arg_info = {
                'name': arg.arg,
                'annotation': None,
            }
            if arg.annotation:
                arg_info['annotation'] = ast.unparse(arg.annotation)
            args.append(arg_info)
        return args

    @staticmethod
    def _get_return_type(node: ast.FunctionDef) -> Optional[str]:
        """获取返回类型"""
        if node.returns:
            return ast.unparse(node.returns)
        return None

    @staticmethod
    def _get_decorators(node: ast.ClassDef) -> List[str]:
        """获取装饰器"""
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name):
                    decorators.append(decorator.func.id)
        return decorators
```

---

## 4.4 组件验证

### 4.4.1 接口验证器

```python
# deepflow/core/validator.py

from typing import Type, List, Dict
from ..components.base_component import BaseComponent

class ComponentValidator:
    """组件验证器"""

    # 必需方法定义
    REQUIRED_METHODS = {
        'BaseModel': ['forward', 'get_metadata'],
        'BaseLoss': ['forward', 'get_metadata'],
        'BaseMetric': ['compute', 'get_metadata'],
        'BaseOptimizer': ['step', 'get_metadata'],
    }

    @staticmethod
    def validate_component(component_class: Type,
                          expected_base: str) -> bool:
        """验证组件是否符合规范"""

        # 检查基类继承
        if not ComponentValidator._check_inheritance(
            component_class, expected_base):
            return False

        # 检查必需方法
        if not ComponentValidator._check_required_methods(
            component_class, expected_base):
            return False

        # 检查元数据
        if not ComponentValidator._check_metadata(component_class):
            return False

        return True

    @staticmethod
    def _check_inheritance(component_class: Type,
                          expected_base: str) -> bool:
        """检查继承关系"""
        base_names = [base.__name__ for base in component_class.__mro__]
        return expected_base in base_names

    @staticmethod
    def _check_required_methods(component_class: Type,
                               expected_base: str) -> bool:
        """检查必需方法"""
        required = ComponentValidator.REQUIRED_METHODS.get(expected_base, [])

        for method_name in required:
            if not hasattr(component_class, method_name):
                print(f"Missing required method: {method_name}")
                return False

        return True

    @staticmethod
    def _check_metadata(component_class: Type) -> bool:
        """检查元数据"""
        try:
            metadata = component_class.get_metadata()
            required_fields = ['name', 'category', 'description']

            for field in required_fields:
                if not hasattr(metadata, field):
                    print(f"Missing metadata field: {field}")
                    return False

            return True
        except Exception as e:
            print(f"Failed to get metadata: {e}")
            return False
```

---

## 4.5 缓存机制

### 4.5.1 发现结果缓存

```python
# deepflow/core/cache.py

import json
import hashlib
from pathlib import Path
from typing import Dict, List
from datetime import datetime

class DiscoveryCache:
    """发现结果缓存"""

    def __init__(self, cache_file: str = ".deepflow_cache.json"):
        self.cache_file = Path(cache_file)
        self.cache_data = self._load_cache()

    def _load_cache(self) -> Dict:
        """加载缓存"""
        if self.cache_file.exists():
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {'version': '1.0', 'components': {}, 'checksums': {}}

    def save_cache(self):
        """保存缓存"""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache_data, f, indent=2, ensure_ascii=False)

    def is_file_changed(self, file_path: Path) -> bool:
        """检查文件是否变化"""
        current_checksum = self._calculate_checksum(file_path)
        cached_checksum = self.cache_data['checksums'].get(str(file_path))

        if cached_checksum != current_checksum:
            self.cache_data['checksums'][str(file_path)] = current_checksum
            return True

        return False

    @staticmethod
    def _calculate_checksum(file_path: Path) -> str:
        """计算文件校验和"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def get_cached_components(self, component_type: str) -> List[Dict]:
        """获取缓存的组件"""
        return self.cache_data['components'].get(component_type, [])

    def update_components(self, component_type: str, components: List[Dict]):
        """更新组件缓存"""
        self.cache_data['components'][component_type] = components
        self.cache_data['last_updated'] = datetime.now().isoformat()
```

下一段将继续说明动态加载和实例化机制。
