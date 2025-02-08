import os
import importlib.util
import streamlit as st
import inspect


def load_selected_class():
    """
    从用户选择的脚本文件中加载类。
    """
    selected_model_folder = st.session_state.get('selected_model_folder', None)
    selected_script_file = st.session_state.get('selected_script_file', None)
    selected_class = st.session_state.get('selected_class', None)

    if not all([selected_model_folder, selected_script_file]):
        st.error("请选择模型文件夹和脚本文件")
        return None

    # 构建脚本文件路径
    models_dir = os.path.join("models", st.session_state.get('selected_category', ''),
                              st.session_state.get('selected_subcategory', ''))
    model_folder_path = os.path.join(models_dir, selected_model_folder)
    script_path = os.path.join(model_folder_path, selected_script_file)

    if not os.path.exists(script_path):
        st.error(f"脚本文件不存在: {script_path}")
        return None

    # 动态加载脚本文件
    spec = importlib.util.spec_from_file_location(selected_script_file[:-3], script_path)
    if spec is None:
        st.error(f"无法创建模块规范: {script_path}")
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # 获取模块中所有类
    valid_classes = []
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and not name.startswith('_'):
            valid_classes.append(name)

    if not valid_classes:
        st.error(f"模块 {module.__name__} 中没有可用的类")
        return None

    # 如果未选择类，则显示选择框
    if not selected_class:
        selected_class = st.selectbox("选择类", valid_classes)
        st.session_state['selected_class'] = selected_class

    # 检查选择的类是否存在
    if selected_class not in valid_classes:
        st.error(f"模块 {module.__name__} 中没有类 {selected_class}")
        return None

    # 获取类
    cls = getattr(module, selected_class)
    return cls