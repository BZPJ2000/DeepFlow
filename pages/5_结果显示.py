# pages/5_结果显示.py
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_echarts import st_echarts
import random
# Set page title
st.set_page_config(page_title='模型结果展示')

import streamlit as st

# 创建一个按钮来控制是否显示session_state
if st.button("显示当前session_state内容"):
    # 使用expander折叠内容
    with st.expander("查看session_state详情", expanded=True):
        st.subheader("当前session_state内容")

        # 遍历所有session_state项
        for key in st.session_state:
            st.markdown(f"**{key}**:")

            try:
                # 尝试用JSON格式显示
                st.json({
                    "类型": str(type(st.session_state[key])),
                    "值": st.session_state[key]
                })
            except TypeError:
                # 如果无法序列化则显示为普通文本
                st.code(f"""
                Type: {type(st.session_state[key])}
                Value: {repr(st.session_state[key])}
                """, language='text')

        st.markdown("---")
        st.caption("⚠️ 注意：部分复杂对象可能无法完整显示")


# ===================== 数据初始化 =====================
def init_demo_data():
    """生成演示用的虚拟数据（全部转换为Python原生类型）"""
    required_keys = ["metrics", "losses", "confusion_matrix"]

    if not all(key in st.session_state for key in required_keys):
        # 模型指标数据（Python字典）
        st.session_state.metrics = {
            "ResNet-50": {
                "Accuracy": round(random.uniform(0.85, 0.95), 3),
                "Precision": round(random.uniform(0.82, 0.93), 3),
                "Recall": round(random.uniform(0.83, 0.94), 3),
                "F1 Score": round(random.uniform(0.84, 0.92), 3)
            },
            "ViT-Base": {
                "Accuracy": round(random.uniform(0.88, 0.94), 3),
                "Precision": round(random.uniform(0.85, 0.91), 3),
                "Recall": round(random.uniform(0.86, 0.93), 3),
                "F1 Score": round(random.uniform(0.87, 0.93), 3)
            },
            "EfficientNet-B4": {
                "Accuracy": round(random.uniform(0.87, 0.93), 3),
                "Precision": round(random.uniform(0.84, 0.92), 3),
                "Recall": round(random.uniform(0.85, 0.91), 3),
                "F1 Score": round(random.uniform(0.86, 0.94), 3)
            }
        }

        # 损失曲线数据（转换为Python列表）
        epochs = 30
        base_train = (np.exp(-np.linspace(0, 2, epochs)) * 10).tolist()
        noise_train = (np.random.randn(epochs) * 0.2).tolist()
        st.session_state.losses = {
            "Train": [round(base + noise, 3) for base, noise in zip(base_train, noise_train)],
            "Val": [round(base + random.uniform(-0.5, 0.5), 3) for base in base_train]
        }

        # 混淆矩阵（Python嵌套列表）
        classes = 5
        matrix = [[random.randint(20, 50) for _ in range(classes)] for _ in range(classes)]
        for i in range(classes):
            matrix[i][i] = random.randint(80, 100)  # 强化对角线
        st.session_state.confusion_matrix = matrix


# ===================== 数据初始化 =====================
def init_simple_data():
    """生成简单的演示数据（纯Python类型）"""
    if "metrics" not in st.session_state:
        # 模型指标数据
        st.session_state.metrics = {
            "Model A": {"Accuracy": 0.89, "Precision": 0.85, "Recall": 0.88},
            "Model B": {"Accuracy": 0.87, "Precision": 0.83, "Recall": 0.85}
        }

        # 损失曲线数据（纯Python列表）
        st.session_state.losses = {
            "Train": [10 - 0.3 * i + random.random() for i in range(30)],
            "Val": [10 - 0.25 * i + random.random() for i in range(30)]
        }

        # 简单混淆矩阵（2x2）
        st.session_state.confusion_matrix = [
            [random.randint(80, 100), random.randint(5, 20)],
            [random.randint(5, 20), random.randint(80, 100)]
        ]


# ===================== 简单可视化组件 =====================
def show_metrics_table():
    """显示指标表格"""
    st.write("模型性能对比：")
    st.table(pd.DataFrame(st.session_state.metrics).T)


def show_loss_chart():
    """显示损失曲线（使用Streamlit原生组件）"""
    st.write("训练过程监控：")
    loss_df = pd.DataFrame({
        "Epoch": list(range(1, 31)),
        "Train Loss": st.session_state.losses["Train"],
        "Val Loss": st.session_state.losses["Val"]
    })
    st.line_chart(loss_df.set_index("Epoch"))


def show_confusion_matrix():
    """显示混淆矩阵（简单表格）"""
    st.write("分类结果分析：")
    matrix_df = pd.DataFrame(
        st.session_state.confusion_matrix,
        index=["True Class 1", "True Class 2"],
        columns=["Pred Class 1", "Pred Class 2"]
    )
    st.table(matrix_df.style.highlight_max(axis=0))


# ===================== 页面布局 =====================
init_simple_data()

st.header("模型训练结果展示")
st.divider()

show_metrics_table()
st.divider()

show_loss_chart()
st.divider()

# show_confusion_matrix()

















# Section for Image Placeholders
st.subheader('模型可视化结果')

# Image Placeholder 1
st.write('图像1')
image1_placeholder = st.empty()
st.write('（此处显示图像1的说明）')

# Image Placeholder 2
st.write('图像2')
image2_placeholder = st.empty()
st.write('（此处显示图像2的说明）')

# Image Placeholder 3
st.write('图像3')
image3_placeholder = st.empty()
st.write('（此处显示图像3的说明）')

# Image Placeholder 4
st.write('图像4')
image4_placeholder = st.empty()
st.write('（此处显示图像4的说明）')

# Image Placeholder 5
st.write('图像5')
image5_placeholder = st.empty()
st.write('（此处显示图像5的说明）')

