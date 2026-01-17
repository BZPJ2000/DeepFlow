"""
深度学习实验管理器 - Web应用

一个基于Web的动态深度学习实验管理和可视化应用。
集成PyTorch深度学习框架。

作者: DL Experiment Manager
版本: 1.0.0
"""

import streamlit as st
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 页面配置
st.set_page_config(
    page_title="深度学习实验管理器",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/dl-experiment-manager',
        'Report a Bug': 'https://github.com/your-repo/dl-experiment-manager/issues',
        'About': """
        # 深度学习实验管理器
        
        一个基于Web的深度学习实验管理、比较和可视化应用，
        符合学术研究标准。
        
        版本: 1.0.0
        """
    }
)

# 自定义CSS样式
def apply_custom_css():
    """应用自定义CSS样式"""
    st.markdown("""
    <style>
    /* 主容器样式 */
    .main {
        padding: 2rem;
    }
    
    /* 侧边栏样式 */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    /* 标题样式 */
    .title-text {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 1rem;
    }
    
    /* 卡片样式 */
    .card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        border-left: 5px solid #3b82f6;
    }
    
    /* 按钮样式 */
    .stButton button {
        background-color: #3b82f6;
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 5px;
        font-weight: 600;
        transition: background-color 0.3s;
    }
    
    .stButton button:hover {
        background-color: #2563eb;
    }
    
    /* 指标卡片样式 */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    
    /* 状态指示器 */
    .status-success {
        color: #10b981;
        font-weight: bold;
    }
    
    .status-warning {
        color: #f59e0b;
        font-weight: bold;
    }
    
    .status-error {
        color: #ef4444;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# 初始化会话状态
def init_session_state():
    """初始化会话状态变量"""
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'home'
    
    if 'experiments' not in st.session_state:
        st.session_state.experiments = []
    
    if 'selected_models' not in st.session_state:
        st.session_state.selected_models = []
    
    if 'selected_datasets' not in st.session_state:
        st.session_state.selected_datasets = []
    
    if 'experiment_results' not in st.session_state:
        st.session_state.experiment_results = {}
    
    if 'framework_path' not in st.session_state:
        # 外部深度学习框架的路径
        framework_path = r"E:\Projects\Learning_space\2025_learn\torch-template-for-deep-learning-main"
        st.session_state.framework_path = framework_path
        # 将框架路径添加到Python路径
        if os.path.exists(framework_path):
            sys.path.insert(0, framework_path)

# 侧边栏导航
def render_sidebar():
    """渲染侧边栏导航"""
    with st.sidebar:
        st.markdown("# 🧪 深度学习实验管理器")
        st.markdown("---")
        
        # 页面导航
        st.markdown("### 📋 导航")
        page_options = {
            "🏠 首页": "home",
            "🤖 模型选择": "model_selection",
            "📊 数据集配置": "dataset_config",
            "⚙️ 实验设置": "experiment_setup",
            "📈 性能比较": "performance_comparison",
            "📊 可视化仪表板": "visualization",
            "📚 文档": "documentation"
        }
        
        for label, page_key in page_options.items():
            if st.button(label, key=f"nav_{page_key}", use_container_width=True):
                st.session_state.current_page = page_key
                st.rerun()
        
        st.markdown("---")
        
        # 快速统计
        st.markdown("### 📊 快速统计")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("实验数量", len(st.session_state.experiments))
        with col2:
            st.metric("已选模型", len(st.session_state.selected_models))
        
        st.markdown("---")
        
        # 框架状态
        st.markdown("### 🔗 框架状态")
        framework_path = st.session_state.get('framework_path', '')
        if os.path.exists(framework_path):
            st.success("✅ 框架已连接")
        else:
            st.error("❌ 框架未找到")
            st.info(f"路径: {framework_path}")
        
        st.markdown("---")
        
        # 快速操作
        st.markdown("### ⚡ 快速操作")
        if st.button("🔄 刷新框架", use_container_width=True):
            st.info("正在刷新框架连接...")
            # 添加框架刷新逻辑
            st.rerun()
        
        if st.button("🧹 清除所有实验", use_container_width=True):
            st.session_state.experiments = []
            st.session_state.selected_models = []
            st.session_state.selected_datasets = []
            st.session_state.experiment_results = {}
            st.success("所有实验已清除！")
            st.rerun()

# 首页
def render_home():
    """渲染首页"""
    st.markdown('<div class="title-text">🧪 深度学习实验管理器</div>', unsafe_allow_html=True)
    st.markdown("### 一个基于Web的动态深度学习实验管理、比较和可视化平台")
    
    # 介绍卡片
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        ### 欢迎使用深度学习实验管理器
        
        本应用提供以下功能：
        
        - **动态模型选择**：从广泛的深度学习架构中选择
        - **数据集配置和预处理**：支持多种格式
        - **自动化实验设置**：可自定义参数
        - **性能比较**：跨多个模型和数据集
        - **学术质量可视化**：用于研究出版物
        
        从侧边栏导航选择页面开始使用。
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 快速开始指南
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 1. 🤖 模型选择")
            st.markdown("""
            - 浏览可用的模型架构
            - 按任务类型和复杂度筛选
            - 配置模型参数
            - 预览模型架构
            """)
            if st.button("前往模型选择", key="home_model"):
                st.session_state.current_page = 'model_selection'
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 2. 📊 数据集配置")
            st.markdown("""
            - 从内置数据集中选择
            - 上传自定义数据集
            - 配置数据增强
            - 设置训练/验证/测试分割
            """)
            if st.button("前往数据集配置", key="home_dataset"):
                st.session_state.current_page = 'dataset_config'
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 3. ⚙️ 实验设置")
            st.markdown("""
            - 配置训练参数
            - 选择评估指标
            - 设置实验跟踪
            - 调度多次运行
            """)
            if st.button("前往实验设置", key="home_experiment"):
                st.session_state.current_page = 'experiment_setup'
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    
    # 最近实验
    if st.session_state.experiments:
        st.markdown("### 📋 最近实验")
        for i, exp in enumerate(st.session_state.experiments[-3:]):
            with st.expander(f"实验 {i+1}: {exp.get('name', '未命名')}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**模型:** {exp.get('model', 'N/A')}")
                with col2:
                    st.write(f"**数据集:** {exp.get('dataset', 'N/A')}")
                with col3:
                    st.write(f"**状态:** {exp.get('status', '未知')}")
    
    # 框架信息
    st.markdown("### 🔗 已连接框架")
    framework_path = st.session_state.get('framework_path', '')
    if os.path.exists(framework_path):
        st.success(f"✅ 已连接到框架: {framework_path}")
        
        # 尝试获取框架信息
        try:
            # 这里将被实际的框架检测逻辑替换
            st.info("框架包含大量预训练模型、数据集和评估指标。")
        except Exception as e:
            st.warning(f"无法加载框架详情: {e}")
    else:
        st.error(f"❌ 未找到框架: {framework_path}")
        st.warning("请在代码中更新框架路径以连接到深度学习框架。")

# 主应用逻辑
def main():
    """主应用函数"""
    # 应用自定义CSS
    apply_custom_css()
    
    # 初始化会话状态
    init_session_state()
    
    # 渲染侧边栏
    render_sidebar()
    
    # 渲染当前页面
    current_page = st.session_state.current_page
    
    # 页面路由
    if current_page == 'home':
        render_home()
    
    # 页脚
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            "<div style='text-align: center; color: #6b7280;'>"
            "🧪 深度学习实验管理器 v1.0.0 | "
            "为研究而制作"
            "</div>", 
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()