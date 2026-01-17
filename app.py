"""DeepFlow 主应用入口

基于 Streamlit 的深度学习实验管理平台。
"""

import streamlit as st
from pathlib import Path

# 配置页面
st.set_page_config(
    page_title="DeepFlow - 深度学习实验管理平台",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .feature-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """主函数"""

    # 标题
    st.markdown('<h1 class="main-header">🚀 DeepFlow</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem;">深度学习实验管理平台 v2.0</p>', unsafe_allow_html=True)

    st.markdown("---")

    # 功能介绍
    st.subheader("✨ 核心功能")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="feature-box">
            <h3>🤖 自动组件发现</h3>
            <p>零配置自动识别模型、损失函数、评估指标和优化器</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-box">
            <h3>📊 实验管理</h3>
            <p>完整的实验配置、执行、结果追踪和可视化</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-box">
            <h3>🎯 多领域支持</h3>
            <p>支持 NLP、计算机视觉、图神经网络、强化学习</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-box">
            <h3>⚡ 快速开始</h3>
            <p>直观的 Web 界面，5 分钟完成首个实验</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 快速开始
    st.subheader("🚀 快速开始")

    st.markdown("""
    1. **选择任务** - 从侧边栏选择 "任务选择" 页面
    2. **选择模型** - 浏览并选择适合的模型
    3. **配置数据** - 设置数据路径和预处理
    4. **训练配置** - 配置训练参数
    5. **开始训练** - 启动实验并查看结果
    """)

    st.markdown("---")

    # 系统状态
    st.subheader("📈 系统状态")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("可用模型", "0", help="已发现的模型数量")

    with col2:
        st.metric("损失函数", "0", help="已发现的损失函数数量")

    with col3:
        st.metric("评估指标", "0", help="已发现的评估指标数量")

    with col4:
        st.metric("优化器", "0", help="已发现的优化器数量")

    st.info("💡 提示: 首次启动时会自动扫描组件库，请稍候...")

if __name__ == "__main__":
    main()
