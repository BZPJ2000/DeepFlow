"""任务选择页面

选择深度学习任务的领域和子任务。
"""

import streamlit as st

# 任务分类
TASK_CATEGORIES = {
    "自然语言处理 (NLP)": {
        "icon": "📝",
        "subcategories": ["情感分类", "机器翻译", "命名实体识别", "文本生成"]
    },
    "计算机视觉 (CV)": {
        "icon": "🖼️",
        "subcategories": ["图像分类", "目标检测", "图像分割", "图像生成"]
    },
    "图神经网络 (GNN)": {
        "icon": "🕸️",
        "subcategories": ["图分类", "节点分类", "链接预测", "图生成"]
    },
    "强化学习 (RL)": {
        "icon": "🎮",
        "subcategories": ["Q-Learning", "Deep Q-Network", "Policy Gradient", "Actor-Critic"]
    }
}

def main():
    """主函数"""

    st.title("🎯 任务选择")
    st.markdown("选择您要进行的深度学习任务类型")

    st.markdown("---")

    # 选择主类别
    st.subheader("1️⃣ 选择任务领域")

    cols = st.columns(4)
    selected_category = None

    for idx, (category, info) in enumerate(TASK_CATEGORIES.items()):
        with cols[idx]:
            if st.button(
                f"{info['icon']}\n\n{category}",
                key=f"cat_{idx}",
                use_container_width=True
            ):
                selected_category = category
                st.session_state['selected_category'] = category

    # 显示已选择的类别
    if 'selected_category' in st.session_state:
        selected_category = st.session_state['selected_category']

        st.markdown("---")
        st.subheader("2️⃣ 选择子任务")

        info = TASK_CATEGORIES[selected_category]
        st.info(f"已选择: {info['icon']} {selected_category}")

        # 选择子类别
        subcategories = info['subcategories']
        cols = st.columns(min(4, len(subcategories)))

        for idx, subcategory in enumerate(subcategories):
            with cols[idx % 4]:
                if st.button(
                    subcategory,
                    key=f"subcat_{idx}",
                    use_container_width=True
                ):
                    st.session_state['selected_subcategory'] = subcategory
                    st.success(f"✅ 已选择: {subcategory}")

        # 显示选择结果
        if 'selected_subcategory' in st.session_state:
            st.markdown("---")
            st.subheader("📋 当前选择")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("任务领域", selected_category)
            with col2:
                st.metric("子任务", st.session_state['selected_subcategory'])

            if st.button("➡️ 下一步：选择模型", type="primary"):
                st.switch_page("pages/2_模型选择.py")

if __name__ == "__main__":
    main()
