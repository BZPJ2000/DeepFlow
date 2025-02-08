# ulit/select_optimizer.py
import os
import streamlit as st
import ast

def parse_script_and_select_function(script_path, folder_name):
    """
    解析脚本文件并选择函数
    :param script_path: 脚本文件路径
    :param folder_name: 文件夹名称（用于生成唯一的 key）
    :return: 选择的函数名
    """
    with open(script_path, "r", encoding="utf-8") as f:
        script_content = f.read()

    excluded_functions = ['__init__', 'forward']  # 定义排除列表

    try:
        script_ast = ast.parse(script_content)
        functions = [node.name for node in ast.walk(script_ast) if
                     isinstance(node, ast.FunctionDef) and node.name not in excluded_functions]
    except Exception as e:
        st.error(f"解析脚本文件失败: {e}")
        return None

    if not functions:
        st.error(f"脚本文件中没有找到函数定义: {script_path}")
        return None

    # 为 selectbox 生成唯一的 key
    function_select_key = f"{folder_name}_function_select"
    selected_function = st.selectbox(
        f"选择 {folder_name} 函数",
        functions,
        key=function_select_key  # 添加唯一的 key
    )

    return selected_function


def select_and_save_optimizer_options():
    """
    处理 optimizer 文件夹中的脚本和函数
    """
    optimizer_dir = "optimizer"

    if not os.path.exists(optimizer_dir):
        st.error(f"优化器目录不存在: {optimizer_dir}")
        return

    script_files = [f for f in os.listdir(optimizer_dir) if f.endswith(".py") and not f.startswith("_")]

    if not script_files:
        st.error(f"优化器目录中没有可用的 Python 脚本: {optimizer_dir}")
        return

    # 为 selectbox 生成唯一的 key
    script_select_key = "optimizer_script_select"
    selected_script_file = st.selectbox(
        "选择一个优化器脚本文件",
        script_files,
        key=script_select_key
    )

    script_path = os.path.join(optimizer_dir, selected_script_file)
    selected_function = parse_script_and_select_function(script_path, "optimizer")

    if selected_function:
        st.session_state.selected_optimizer_script_file = selected_script_file
        st.session_state.selected_optimizer_function = selected_function