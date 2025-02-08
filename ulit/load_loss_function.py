import inspect
import os
import streamlit as st
import importlib.util

def list_loss_files(base_path):
    """
    列出 base_path 下的所有损失函数文件。
    """
    if not os.path.exists(base_path):
        st.error(f"路径不存在: {base_path}")
        return []
    return [f for f in os.listdir(base_path) if f.endswith(".py") and not f.startswith("_")]

def list_classes_from_file(file_path):
    """
    从 .py 文件中列出所有类。
    """
    if not os.path.exists(file_path):
        st.error(f"文件不存在: {file_path}")
        return []

    # 动态加载脚本文件
    spec = importlib.util.spec_from_file_location("loss_module", file_path)
    if spec is None:
        st.error(f"无法创建模块规范: {file_path}")
        return []
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # 获取模块中所有类
    valid_classes = []
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and not name.startswith('_'):
            valid_classes.append(name)

    return valid_classes

def load_loss_function(prefix=""):
    """
    加载损失函数类。
    prefix: 用于生成唯一的 key，避免 Streamlit 元素 ID 冲突。
    """
    # 基础路径
    base_path = os.path.join("losses", st.session_state.get('selected_category', ''),
                             st.session_state.get('selected_subcategory', ''))

    # 1. 列出所有损失函数文件
    loss_files = list_loss_files(base_path)
    if not loss_files:
        st.error(f"没有可用的损失函数文件: {base_path}")
        return None

    # 2. 选择文件
    selected_file = st.selectbox("选择损失函数文件", loss_files, key=f"{prefix}_select_file")
    file_path = os.path.join(base_path, selected_file)

    # 3. 列出文件中的所有类
    classes = list_classes_from_file(file_path)
    if not classes:
        st.error(f"损失函数文件中没有可用的类: {selected_file}")
        return None

    # 4. 选择类
    selected_class = st.selectbox("选择损失函数类", classes, key=f"{prefix}_select_class")

    # 5. 动态加载类
    spec = importlib.util.spec_from_file_location("loss_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, selected_class)

    return cls