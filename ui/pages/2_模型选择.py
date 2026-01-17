"""模型选择页面

浏览和选择深度学习模型。
"""

import streamlit as st
from deepflow.core.registry import ComponentRegistry
from deepflow.core.discovery import ComponentDiscovery

def main():
    """主函数"""

    st.title("🤖 模型选择")

    # 检查是否已选择任务
    if 'selected_category' not in st.session_state:
        st.warning("⚠️ 请先选择任务类型")
        if st.button("返回任务选择"):
            st.switch_page("pages/1_任务选择.py")
        return

    # 显示当前任务
    st.info(f"当前任务: {st.session_state['selected_category']} - {st.session_state.get('selected_subcategory', '未选择')}")

    st.markdown("---")

    # 初始化组件发现（如果还未初始化）
    if 'registry' not in st.session_state:
        with st.spinner("正在扫描组件库..."):
            registry = ComponentRegistry()
            discovery = ComponentDiscovery('library')

            # 发现所有组件
            discovered = discovery.discover_all()

            # 注册组件
            for comp_type, components in discovered.items():
                for comp in components:
                    registry.register(comp_type, comp.name, comp)

            st.session_state['registry'] = registry
            st.success("✅ 组件扫描完成")

    # 获取可用模型
    registry = st.session_state['registry']
    category = st.session_state.get('selected_category', '')

    # 映射中文类别到英文
    category_map = {
        "自然语言处理 (NLP)": "nlp",
        "计算机视觉 (CV)": "vision",
        "图神经网络 (GNN)": "graph",
        "强化学习 (RL)": "rl"
    }

    category_en = category_map.get(category, "")
    models = registry.list('models', category=category_en)

    st.subheader(f"可用模型 ({len(models)})")

    if len(models) == 0:
        st.warning("暂无可用模型，请先添加模型到 library/models/ 目录")
        return

    # 筛选选项
    col1, col2 = st.columns(2)
    with col1:
        sort_by = st.selectbox("排序方式", ["名称", "类别"])
    with col2:
        search_query = st.text_input("搜索模型", placeholder="输入模型名称...")

    # 筛选模型
    filtered_models = models
    if search_query:
        filtered_models = [m for m in models if search_query.lower() in m.name.lower()]

    # 显示模型列表
    for model in filtered_models:
        with st.expander(f"📦 {model.name}"):
            st.markdown(f"**描述:** {model.description or '暂无描述'}")
            st.markdown(f"**类别:** {model.category} / {model.subcategory}")

            if model.tags:
                st.markdown("**标签:** " + " ".join([f"`{tag}`" for tag in model.tags]))

            if st.button(f"选择 {model.name}", key=f"select_{model.name}"):
                st.session_state['selected_model'] = model
                st.success(f"✅ 已选择模型: {model.name}")
                st.rerun()

    # 显示已选择的模型
    if 'selected_model' in st.session_state:
        st.markdown("---")
        st.subheader("✅ 已选择模型")
        model = st.session_state['selected_model']
        st.info(f"**{model.name}** - {model.description}")

        if st.button("➡️ 下一步：配置数据", type="primary"):
            st.switch_page("pages/3_数据配置.py")

if __name__ == "__main__":
    main()
