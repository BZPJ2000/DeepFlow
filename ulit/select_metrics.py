# ulit/select_metrics.py
import os
import streamlit as st
import ast

def parse_script_and_select_functions(script_path, folder_name):
    """
    解析脚本文件并选择函数
    :param script_path: 脚本文件路径
    :param folder_name: 文件夹名称（用于生成唯一的 key）
    :return: 选择的函数列表
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
        return []

    if not functions:
        st.error(f"脚本文件中没有找到函数定义: {script_path}")
        return []

    # 为 multiselect 生成唯一的 key
    function_select_key = f"{folder_name}_function_select"
    selected_functions = st.multiselect(
        f"选择 {folder_name} 函数（可多选）",
        functions,
        key=function_select_key  # 添加唯一的 key
    )

    return selected_functions


def select_and_save_metric_options(selected_category, selected_subcategory):
    """
    处理 metrics 文件夹中的脚本和函数
    :param selected_category: 任务类别
    :param selected_subcategory: 任务子类别
    """
    metrics_dir = os.path.join("metrics", selected_category, selected_subcategory)

    if not os.path.exists(metrics_dir):
        st.error(f"评估指标目录不存在: {metrics_dir}")
        return

    script_files = [f for f in os.listdir(metrics_dir) if f.endswith(".py") and not f.startswith("_")]

    if not script_files:
        st.error(f"评估指标目录中没有可用的 Python 脚本: {metrics_dir}")
        return

    # 为 selectbox 生成唯一的 key
    script_select_key = f"metrics_{selected_category}_{selected_subcategory}_script_select"
    selected_script_file = st.selectbox(
        "选择一个评估指标脚本文件",
        script_files,
        key=script_select_key
    )

    script_path = os.path.join(metrics_dir, selected_script_file)
    selected_functions = parse_script_and_select_functions(script_path, "metrics")

    if selected_functions:
        st.session_state.selected_metric_script_file = selected_script_file
        st.session_state.selected_metric_functions = selected_functions