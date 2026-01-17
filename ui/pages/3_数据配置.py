"""数据配置页面

配置数据集路径和预处理参数。
"""

import streamlit as st
from pathlib import Path

def main():
    """主函数"""

    st.title("📊 数据配置")

    # 检查前置条件
    if 'selected_model' not in st.session_state:
        st.warning("⚠️ 请先选择模型")
        if st.button("返回模型选择"):
            st.switch_page("pages/2_模型选择.py")
        return

    # 显示当前选择
    st.info(f"当前模型: {st.session_state['selected_model'].name}")

    st.markdown("---")

    # 数据路径配置
    st.subheader("1️⃣ 数据路径")

    data_path = st.text_input(
        "数据集路径",
        value="data/samples",
        help="输入数据集所在目录的路径"
    )

    if Path(data_path).exists():
        st.success(f"✅ 路径有效: {data_path}")
    else:
        st.error(f"❌ 路径不存在: {data_path}")

    st.markdown("---")

    # 数据分割配置
    st.subheader("2️⃣ 数据分割")

    col1, col2, col3 = st.columns(3)

    with col1:
        train_ratio = st.slider("训练集比例", 0.0, 1.0, 0.8, 0.05)
    with col2:
        val_ratio = st.slider("验证集比例", 0.0, 1.0, 0.1, 0.05)
    with col3:
        test_ratio = st.slider("测试集比例", 0.0, 1.0, 0.1, 0.05)

    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        st.error(f"⚠️ 比例总和应为 1.0，当前为 {total_ratio:.2f}")
    else:
        st.success(f"✅ 比例配置正确")

    st.markdown("---")

    # Batch Size 配置
    st.subheader("3️⃣ 批次大小")

    batch_size = st.number_input(
        "Batch Size",
        min_value=1,
        max_value=512,
        value=32,
        step=1,
        help="每个批次的样本数量"
    )

    st.markdown("---")

    # 数据增强（可选）
    st.subheader("4️⃣ 数据增强（可选）")

    use_augmentation = st.checkbox("启用数据增强")

    if use_augmentation:
        augmentations = st.multiselect(
            "选择增强方法",
            ["随机翻转", "随机旋转", "随机裁剪", "颜色抖动", "归一化"],
            default=["随机翻转", "归一化"]
        )
        st.info(f"已选择 {len(augmentations)} 种增强方法")

    # 保存配置
    if st.button("💾 保存配置并继续", type="primary"):
        st.session_state['data_config'] = {
            'data_path': data_path,
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'batch_size': batch_size,
            'use_augmentation': use_augmentation,
            'augmentations': augmentations if use_augmentation else []
        }
        st.success("✅ 配置已保存")
        st.switch_page("pages/4_训练配置.py")

if __name__ == "__main__":
    main()
