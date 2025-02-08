# ulit/display_model_info.py
import streamlit as st
from io import StringIO
from torchsummary import summary
import torch


def display_model_info(model):
    # 打印模型结构
    st.write("模型结构:")
    st.write(model)

    # 计算参数数量
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())

    st.write(f"参数数量: {count_parameters(model)}")

    # 估计内存消耗
    st.write("内存消耗估计:")
    buf = StringIO()
    try:
        summary(model, input_size=(3, 224, 224), file=buf)
        st.write(buf.getvalue())
    except Exception as e:
        st.error(f"估计内存消耗失败: {e}")

    # 查看GPU内存使用情况
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        st.write(f"已分配的GPU内存: {allocated} bytes")
    else:
        st.write("没有检测到GPU设备")