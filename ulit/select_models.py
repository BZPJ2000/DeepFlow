# ulit/select_models.py
import streamlit as st
import os
import ast

def select_and_save_options(selected_category, selected_subcategory):
    models_dir = os.path.join("models", selected_category, selected_subcategory)

    if not os.path.exists(models_dir):
        st.error(f"模型目录不存在: {models_dir}")
        return

    model_folders = [f for f in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, f))]

    if not model_folders:
        st.error(f"模型目录中没有可用的模型文件夹: {models_dir}")
        return

    selected_model_folder = st.selectbox("选择一个模型文件夹", model_folders)

    model_folder_path = os.path.join(models_dir, selected_model_folder)
    script_files = [f for f in os.listdir(model_folder_path) if f.endswith(".py") and not f.startswith("_")]

    if not script_files:
        st.error(f"模型文件夹中没有可用的 Python 脚本: {model_folder_path}")
        return

    selected_script_file = st.selectbox("选择一个脚本文件", script_files)

    script_path = os.path.join(model_folder_path, selected_script_file)
    with open(script_path, "r", encoding="utf-8") as f:
        script_content = f.read()

    excluded_functions = ['__init__', 'forward']  # 定义排除列表

    try:
        script_ast = ast.parse(script_content)
        functions = [node.name for node in ast.walk(script_ast) if
                     isinstance(node, ast.FunctionDef) and node.name not in excluded_functions]
    except Exception as e:
        st.error(f"解析脚本文件失败: {e}")
        return

    if not functions:
        st.error(f"脚本文件中没有找到函数定义: {selected_script_file}")
        return

    selected_function = st.selectbox("选择一个函数", functions)

    st.session_state.selected_model_folder = selected_model_folder
    st.session_state.selected_script_file = selected_script_file
    st.session_state.selected_function = selected_function