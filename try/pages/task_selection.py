# import streamlit as st
# from ulit.load_selected_class import load_selected_class
#
#
#
# # 标题
# st.title("模型选择界面")
#
# # 加载选中的类
# selected_class = load_selected_class()
#
# if selected_class:
#     try:
#         # 实例化类
#         model = selected_class(num_classes=3)
#         st.write(f"成功加载类: {selected_class.__name__}")
#         st.write(model)
#     except Exception as e:
#         st.error(f"实例化类失败: {e}")