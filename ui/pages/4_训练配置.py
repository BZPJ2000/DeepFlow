"""训练配置页面

配置训练参数和超参数。
"""

import streamlit as st

def main():
    """主函数"""

    st.title("⚙️ 训练配置")

    # 检查前置条件
    if 'data_config' not in st.session_state:
        st.warning("⚠️ 请先配置数据")
        if st.button("返回数据配置"):
            st.switch_page("pages/3_数据配置.py")
        return

    # 显示当前配置
    st.info(f"模型: {st.session_state['selected_model'].name} | "
            f"Batch Size: {st.session_state['data_config']['batch_size']}")

    st.markdown("---")

    # 训练参数
    st.subheader("1️⃣ 基础训练参数")

    col1, col2 = st.columns(2)

    with col1:
        epochs = st.number_input("训练轮数 (Epochs)", 1, 1000, 10, 1)
        learning_rate = st.number_input("学习率 (Learning Rate)",
                                       0.0001, 1.0, 0.001, 0.0001,
                                       format="%.4f")

    with col2:
        device = st.selectbox("训练设备", ["cuda", "cpu"])
        save_interval = st.number_input("保存间隔 (Epochs)", 1, 100, 5, 1)

    st.markdown("---")

    # 优化器选择
    st.subheader("2️⃣ 优化器")

    optimizer_name = st.selectbox(
        "选择优化器",
        ["Adam", "SGD", "AdamW", "RMSprop"]
    )

    if optimizer_name == "SGD":
        momentum = st.slider("Momentum", 0.0, 1.0, 0.9, 0.05)

    weight_decay = st.number_input("Weight Decay", 0.0, 0.1, 0.0001, 0.0001, format="%.4f")

    st.markdown("---")

    # 损失函数
    st.subheader("3️⃣ 损失函数")

    loss_name = st.selectbox(
        "选择损失函数",
        ["CrossEntropyLoss", "MSELoss", "BCELoss", "L1Loss"]
    )

    st.markdown("---")

    # 高级选项
    with st.expander("🔧 高级选项"):
        use_scheduler = st.checkbox("使用学习率调度器")
        if use_scheduler:
            scheduler_type = st.selectbox("调度器类型", ["StepLR", "CosineAnnealingLR"])

        use_early_stopping = st.checkbox("使用早停")
        if use_early_stopping:
            patience = st.number_input("耐心值 (Patience)", 1, 50, 10, 1)

    # 保存并开始训练
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("💾 保存配置", use_container_width=True):
            st.session_state['training_config'] = {
                'epochs': epochs,
                'learning_rate': learning_rate,
                'device': device,
                'save_interval': save_interval,
                'optimizer': optimizer_name,
                'loss': loss_name,
                'weight_decay': weight_decay
            }
            st.success("✅ 配置已保存")

    with col2:
        if st.button("🚀 开始训练", type="primary", use_container_width=True):
            st.session_state['training_config'] = {
                'epochs': epochs,
                'learning_rate': learning_rate,
                'device': device,
                'save_interval': save_interval,
                'optimizer': optimizer_name,
                'loss': loss_name,
                'weight_decay': weight_decay
            }
            st.success("✅ 训练配置已保存，准备开始训练...")
            st.info("💡 训练功能正在开发中，敬请期待！")

if __name__ == "__main__":
    main()
