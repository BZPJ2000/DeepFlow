import importlib
import inspect
import os
import streamlit as st

def load_optimizer():
    """
    加载优化器类。
    """
    optimizer_path = os.path.join("optimizer", "optimizer.py")

    if not os.path.exists(optimizer_path):
        st.error(f"优化器文件不存在: {optimizer_path}")
        return None

    # 动态加载脚本文件
    spec = importlib.util.spec_from_file_location("optimizer_module", optimizer_path)
    if spec is None:
        st.error(f"无法创建模块规范: {optimizer_path}")
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # 获取模块中所有类
    valid_classes = []
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and not name.startswith('_'):
            valid_classes.append(name)

    if not valid_classes:
        st.error(f"优化器文件中没有可用的类: {optimizer_path}")
        return None

    # 选择类
    selected_class = st.selectbox("选择优化器类", valid_classes, key="optimizer_select_class")

    # 动态加载类
    cls = getattr(module, selected_class)
    return cls