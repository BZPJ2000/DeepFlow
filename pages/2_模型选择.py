import streamlit as st
import os
import ast
import importlib.util

from ulit.load_selected_class import load_selected_class
from ulit.model_utils import estimate_training_time, plot_resource_usage, get_system_resources, \
    calculate_model_parameters
from ulit.select_models import select_and_save_options

from ulit.settings_to_file import save_settings_to_file, load_settings_from_file, get_settings_files

# 页面标题
st.title("模型结构选择")

# 从 session_state 中获取用户选择的大类和小类
selected_category = st.session_state.get("selected_category", None)
selected_subcategory = st.session_state.get("selected_subcategory", None)

if not selected_category or not selected_subcategory:
    st.error("请先返回任务选择页面选择任务类型。")
    st.stop()

st.write(f"你选择的任务: {selected_category} -> {selected_subcategory}")

# 判断是否需要加载两个模型
is_dual_model_task = selected_subcategory in ["遮挡物修复", "图像生成"]

# 加载选中的函数
if is_dual_model_task:
    st.subheader("选择生成器模型")
    generator_function = load_selected_class(prefix="generator")  # 传入唯一前缀

    st.subheader("选择判别器模型")
    discriminator_function = load_selected_class(prefix="discriminator")  # 传入唯一前缀

    if generator_function and discriminator_function:
        try:
            generator_model = generator_function()
            discriminator_model = discriminator_function()
        except Exception as e:
            st.error(f"实例化模型失败: {e}")
            generator_model, discriminator_model = None, None

        if generator_model and discriminator_model:
            # 生成器模型结构打印区域
            st.subheader("生成器模型结构")
            with st.expander("查看生成器模型详细结构"):
                st.write(generator_model)

            # 判别器模型结构打印区域
            st.subheader("判别器模型结构")
            with st.expander("查看判别器模型详细结构"):
                st.write(discriminator_model)

            # 参数计算
            st.subheader("参数计算")
            total_params_generator, trainable_params_generator = calculate_model_parameters(generator_model)
            total_params_discriminator, trainable_params_discriminator = calculate_model_parameters(discriminator_model)

            st.write("生成器模型:")
            st.write(f"总参数量: {total_params_generator}")
            st.write(f"可训练参数量: {trainable_params_generator}")

            st.write("判别器模型:")
            st.write(f"总参数量: {total_params_discriminator}")
            st.write(f"可训练参数量: {trainable_params_discriminator}")

            # 将总参数量保存到 session_state（假设我们只保存生成器的参数）
            st.session_state.total_params = total_params_generator
            st.session_state.trainable_params = trainable_params_generator
else:
    selected_function = load_selected_class(prefix="single_model")  # 传入唯一前缀

    if selected_function:
        try:
            model = selected_function(num_classes=3)
        except Exception as e:
            st.error(f"实例化模型失败: {e}")
            model = None

        if model:
            # 模型结构打印区域
            st.subheader("模型结构")
            with st.expander("查看模型详细结构"):
                st.write(model)

            # 参数计算
            st.subheader("参数计算")
            total_params, trainable_params = calculate_model_parameters(model)
            st.session_state.total_params = total_params
            st.session_state.trainable_params = trainable_params
            st.write(f"总参数量: {total_params}")
            st.write(f"可训练参数量: {trainable_params}")

# 以下是通用逻辑，无论是单模型还是双模型都会执行

# 计算机资源统计
st.subheader("计算机资源统计")
cpu_usage, memory_usage, gpu_info = get_system_resources()
st.session_state.cpu_usage = cpu_usage
st.session_state.memory_usage = memory_usage
st.session_state.gpu_info = gpu_info
st.write(f"CPU 使用率: {cpu_usage}%")
st.write(f"内存使用率: {memory_usage}%")
if gpu_info:
    for gpu in gpu_info:
        st.write(f"GPU {gpu['id']} ({gpu['name']}):")
        st.write(f"- 使用率: {gpu['load']:.2f}%")
        st.write(f"- 显存使用: {gpu['memory_used']} MB / {gpu['memory_total']} MB")
else:
    st.write("未检测到 GPU 设备。")

# 资源统计图表
st.subheader("资源统计图表")
fig = plot_resource_usage(cpu_usage, memory_usage, gpu_info)
st.pyplot(fig)

# 训练时间 vs 批次大小
st.subheader("训练时间 vs 批次大小")

# 让用户输入批次大小
st.write("**设置批次大小:**")
batch_sizes = st.text_input(
    "请输入批次大小（用逗号分隔，例如：1,2,4,8,16,32,64,128,256）",
    value=",".join(map(str, st.session_state.get("batch_sizes", [1, 2, 4, 8, 16, 32, 64, 128, 256])))
)

# 将用户输入的字符串转换为整数列表
try:
    batch_sizes = [int(size.strip()) for size in batch_sizes.split(",")]
    st.session_state.batch_sizes = batch_sizes
except ValueError:
    st.error("请输入有效的批次大小（例如：1,2,4,8,16,32,64,128,256）")
    st.stop()

# 检查批次大小是否有效
if not batch_sizes:
    st.error("批次大小不能为空！")
    st.stop()

# 估算训练时间和内存占用
if st.button("估算训练时间和内存占用"):
    with st.spinner("计算中..."):
        fig, memory_usages, training_times = estimate_training_time(st.session_state.total_params, batch_sizes)
        st.pyplot(fig)

        # 将结果保存到 session_state
        st.session_state.memory_usages = memory_usages
        st.session_state.training_times = training_times

        # 使用 st.columns 创建两列
        col1, col2 = st.columns(2)

        # 在第一列显示内存占用结果
        with col1:
            st.write("**内存占用结果:**")
            for batch_size, memory_usage in zip(batch_sizes, memory_usages):
                st.write(f"批次大小 {batch_size}: {memory_usage:.2f} MB")

        # 在第二列显示训练时间结果
        with col2:
            st.write("**训练时间结果:**")
            for batch_size, time_per_batch in zip(batch_sizes, training_times):
                st.write(f"批次大小 {batch_size}: {time_per_batch:.4f} 秒/批次")

# 保存模型相关设置
if st.button("保存模型设置部分内容"):
    memory_usages = st.session_state.get("memory_usages", [])
    training_times = st.session_state.get("training_times", [])
    st.session_state.model_settings = {
        "total_params": st.session_state.total_params,
        "trainable_params": st.session_state.trainable_params,
        "cpu_usage": st.session_state.cpu_usage,
        "memory_usage": st.session_state.memory_usage,
        "gpu_info": st.session_state.gpu_info,
        "batch_sizes": st.session_state.batch_sizes,
        "memory_usages": memory_usages,
        "training_times": training_times
    }
    st.success("模型相关信息设置已保存")

# 导航到下一页
if st.button("下一步"):
    st.switch_page("pages/3_数据处理.py")