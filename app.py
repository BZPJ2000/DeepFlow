import streamlit as st

# 设置页面标题
st.title("深度学习项目管理应用")

# 欢迎语
st.markdown("### 欢迎使用 **深度学习项目管理应用**！")

# 项目简介
st.header("项目简介")
st.write("这是一个用于管理深度学习项目的简单小应用，提供了数据处理、模型训练和结果评估等功能。")

# 分隔线
st.divider()

# 关于作者
st.header("关于作者")
st.markdown("作者: [BZPJ]  \n"
            "GitHub: [链接](https://github.com/BZPJ2000)  \n"
            "邮箱: [136771358@qq.com]")

# 分隔线
st.divider()

# 使用规则
st.header("使用规则")
st.markdown("""
- 选择你深度学习的任务方向
- 选择你的模型文件  
- 选择数据处理页面进行数据准备和处理。  
- 选择你的配置，然后进行训练
- 最后查看模型的性能对比，巴拉巴拉一大堆
""")