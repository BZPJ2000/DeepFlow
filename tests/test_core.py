"""测试核心功能

验证组件发现、注册和加载功能。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from deepflow.core.registry import ComponentRegistry
from deepflow.core.discovery import ComponentDiscovery
from deepflow.api.experiment import ExperimentAPI


def test_discovery():
    """测试组件发现"""
    print("=" * 50)
    print("测试组件发现功能")
    print("=" * 50)

    discovery = ComponentDiscovery('library')
    discovered = discovery.discover_all()

    for comp_type, components in discovered.items():
        print(f"\n{comp_type}: 发现 {len(components)} 个组件")
        for comp in components[:3]:  # 只显示前3个
            print(f"  - {comp.name} ({comp.category}/{comp.subcategory})")


def test_registry():
    """测试组件注册"""
    print("\n" + "=" * 50)
    print("测试组件注册功能")
    print("=" * 50)

    registry = ComponentRegistry()
    discovery = ComponentDiscovery('library')
    discovered = discovery.discover_all()

    # 注册所有组件
    for comp_type, components in discovered.items():
        for comp in components:
            registry.register(comp_type, comp.name, comp)

    # 测试查询
    models = registry.list('models')
    print(f"\n注册的模型数量: {len(models)}")


def test_api():
    """测试 API 接口"""
    print("\n" + "=" * 50)
    print("测试 API 接口")
    print("=" * 50)

    api = ExperimentAPI()
    api.initialize()

    # 获取可用模型
    models = api.get_available_models()
    print(f"\n可用模型: {len(models)}")

    # 搜索组件
    results = api.search_components("resnet")
    print(f"搜索 'resnet': {len(results)} 个结果")


if __name__ == "__main__":
    try:
        test_discovery()
        test_registry()
        test_api()
        print("\n" + "=" * 50)
        print("[SUCCESS] All tests passed!")
        print("=" * 50)
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
