import inspect
import os
import streamlit as st
import importlib.util


def load_dataset_class():
    """加载数据集类"""
    # 基础路径：datasets/任务类别/任务子类
    base_path = os.path.join("dataset",
                             st.session_state.get('selected_category', ''),
                             st.session_state.get('selected_subcategory', ''))

    if not os.path.exists(base_path):
        st.error(f"数据集目录不存在: {base_path}")
        return None

    # 列出所有数据集文件
    dataset_files = [f for f in os.listdir(base_path) if f.endswith(".py") and not f.startswith("_")]
    if not dataset_files:
        st.error(f"没有可用的数据集文件: {base_path}")
        return None

    # 选择文件
    selected_file = st.selectbox("选择数据集文件", dataset_files, key="dataset_select_file")
    file_path = os.path.join(base_path, selected_file)

    # 动态加载脚本
    spec = importlib.util.spec_from_file_location("dataset_module", file_path)
    if spec is None:
        st.error(f"无法加载模块: {file_path}")
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # 获取有效类（排除私有类）
    valid_classes = []
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and not name.startswith('_'):
            valid_classes.append(name)

    if not valid_classes:
        st.error(f"数据集中没有找到有效的类: {selected_file}")
        return None

    # 选择类（单选）
    selected_class = st.selectbox("选择数据集类", valid_classes, key="dataset_select_class")

    # 获取类对象
    cls = getattr(module, selected_class)

    # 显示类文档字符串
    if cls.__doc__:
        st.code(f"类说明：\n{cls.__doc__}", language="python")

    return cls
