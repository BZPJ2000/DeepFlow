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


def select_and_save_loss_options(selected_category, selected_subcategory):
    """
    处理 losses 文件夹中的脚本和函数
    :param selected_category: 任务类别
    :param selected_subcategory: 任务子类别
    """
    losses_dir = os.path.join("losses", selected_category, selected_subcategory)

    if not os.path.exists(losses_dir):
        st.error(f"损失函数目录不存在: {losses_dir}")
        return

    script_files = [f for f in os.listdir(losses_dir) if f.endswith(".py") and not f.startswith("_")]

    if not script_files:
        st.error(f"损失函数目录中没有可用的 Python 脚本: {losses_dir}")
        return

    # 为 selectbox 生成唯一的 key
    script_select_key = f"losses_{selected_category}_{selected_subcategory}_script_select"
    selected_script_file = st.selectbox(
        "选择一个损失函数脚本文件",
        script_files,
        key=script_select_key
    )

    # 如果用户选择了脚本文件，则解析该文件并选择函数
    if selected_script_file:
        script_path = os.path.join(losses_dir, selected_script_file)
        selected_function = parse_script_and_select_function(script_path, "losses")

        if selected_function:
            st.session_state.selected_loss_script_file = selected_script_file
            st.session_state.selected_loss_function = selected_function
        else:
            # 如果未选择函数，清空 session_state 中的函数记录
            if "selected_loss_function" in st.session_state:
                del st.session_state.selected_loss_function
    else:
        # 如果未选择脚本文件，清空 session_state 中的脚本和函数记录
        if "selected_loss_script_file" in st.session_state:
            del st.session_state.selected_loss_script_file
        if "selected_loss_function" in st.session_state:
            del st.session_state.selected_loss_function