import os
import importlib.util
import streamlit as st
import inspect


def list_folders(base_path):
    """
    列出 base_path 下的所有文件夹。
    """
    if not os.path.exists(base_path):
        st.error(f"路径不存在: {base_path}")
        return []
    return [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]


def list_py_files(folder_path):
    """
    列出 folder_path 下的所有 .py 文件。
    """
    if not os.path.exists(folder_path):
        st.error(f"路径不存在: {folder_path}")
        return []
    return [f for f in os.listdir(folder_path) if f.endswith(".py")]


def list_classes_from_file(file_path):
    """
    从 .py 文件中列出所有类。
    """
    if not os.path.exists(file_path):
        st.error(f"文件不存在: {file_path}")
        return []

    # 动态加载脚本文件
    spec = importlib.util.spec_from_file_location("module_name", file_path)
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


def load_selected_class():
    """
    分层次选择文件夹、文件和类。
    """
    # 基础路径
    base_path = os.path.join("models", "图像处理", "语义分割")

    # 1. 选择文件夹
    folders = list_folders(base_path)
    if not folders:
        st.error("没有可用的文件夹")
        return None

    selected_folder = st.selectbox("选择文件夹", folders)
    folder_path = os.path.join(base_path, selected_folder)

    # 2. 选择文件
    py_files = list_py_files(folder_path)
    if not py_files:
        st.error("没有可用的 .py 文件")
        return None

    selected_file = st.selectbox("选择文件", py_files)
    file_path = os.path.join(folder_path, selected_file)

    # 3. 选择类
    classes = list_classes_from_file(file_path)
    if not classes:
        st.error("没有可用的类")
        return None

    selected_class = st.selectbox("选择类", classes)

    # 动态加载类
    spec = importlib.util.spec_from_file_location("module_name", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, selected_class)

    return cls