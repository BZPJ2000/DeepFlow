# pages/1_任务方向.py
import os
from ulit.config import task_categories
import streamlit as st
from ulit.settings_to_file import save_settings_to_file, load_settings_from_file, get_settings_files

st.title("深度学习任务选择")

# 定义大类和小类
task_categories = task_categories

# 加载设置文件功能
st.subheader("加载设置文件")
settings_files = get_settings_files()  # 获取 logs 文件夹中的所有设置文件
if settings_files:
    # 使用下拉框选择文件
    selected_file = st.selectbox("选择一个设置文件", settings_files)

    if st.button("加载选中的设置文件"):
        # 构建完整文件路径
        filepath = os.path.join("logs", selected_file)
        settings = load_settings_from_file(filepath)

        # 将设置加载到 session_state
        for key, value in settings.items():
            st.session_state[key] = value

        st.success("设置文件已加载！")
else:
    st.write("没有找到设置文件。")

# 用户选择大类
selected_category = st.selectbox("选择一个大类", list(task_categories.keys()), index=(
    list(task_categories.keys()).index(st.session_state.get("selected_category", list(task_categories.keys())[0]))
    if "selected_category" in st.session_state else 0
))

# 根据选择的大类，动态显示小类
if selected_category:
    selected_subcategory = st.selectbox("选择一个小类", task_categories[selected_category], index=(
        task_categories[selected_category].index(
            st.session_state.get("selected_subcategory", task_categories[selected_category][0]))
        if "selected_subcategory" in st.session_state else 0
    ))
else:
    selected_subcategory = None

# 添加一个确认按钮
if st.button("确认选择"):
    if selected_category and selected_subcategory:
        st.session_state.selected_category = selected_category
        st.session_state.selected_subcategory = selected_subcategory
        st.success("选择已保存，准备进入模型选择页面。")
    else:
        st.error("请选择完整的大类和小类")

# 导航到下一页
if st.button("下一步"):
    st.switch_page("pages/2_模型选择.py")