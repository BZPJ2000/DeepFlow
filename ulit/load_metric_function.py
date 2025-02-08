import inspect
import os
import streamlit as st
import importlib.util

def load_metric_function():
    """
    加载评估指标类，支持多选。
    """
    # 基础路径
    base_path = os.path.join("metrics", st.session_state.get('selected_category', ''),
                             st.session_state.get('selected_subcategory', ''))

    # 列出所有评估指标文件
    metric_files = [f for f in os.listdir(base_path) if f.endswith(".py") and not f.startswith("_")]
    if not metric_files:
        st.error(f"没有可用的评估指标文件: {base_path}")
        return None

    # 选择文件
    selected_file = st.selectbox("选择评估指标文件", metric_files, key="metric_select_file")
    file_path = os.path.join(base_path, selected_file)

    # 动态加载脚本文件
    spec = importlib.util.spec_from_file_location("metric_module", file_path)
    if spec is None:
        st.error(f"无法创建模块规范: {file_path}")
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # 获取模块中所有类
    valid_classes = []
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and not name.startswith('_'):
            valid_classes.append(name)

    if not valid_classes:
        st.error(f"评估指标文件中没有可用的类: {selected_file}")
        return None

    # 选择类（支持多选）
    selected_classes = st.multiselect("选择评估指标类", valid_classes, key="metric_select_class")

    # 动态加载类
    classes = []
    for class_name in selected_classes:
        cls = getattr(module, class_name)
        classes.append(cls)

    return classes