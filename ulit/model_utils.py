import time

import numpy as np
import torch
import psutil
import GPUtil
import matplotlib.pyplot as plt
import streamlit as st
import random
def calculate_model_parameters(model):
    """
    计算模型的参数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def get_system_resources():
    """
    获取当前计算机的 CPU 和 GPU 资源信息
    """
    # CPU 使用率
    cpu_usage = psutil.cpu_percent(interval=1)
    # 内存使用情况
    memory_info = psutil.virtual_memory()
    memory_usage = memory_info.percent
    # GPU 使用情况
    gpus = GPUtil.getGPUs()
    gpu_info = []
    for gpu in gpus:
        gpu_info.append({
            "id": gpu.id,
            "name": gpu.name,
            "load": gpu.load * 100,  # GPU 使用率
            "memory_used": gpu.memoryUsed,  # 已用显存
            "memory_total": gpu.memoryTotal  # 总显存
        })
    return cpu_usage, memory_usage, gpu_info

def plot_resource_usage(cpu_usage, memory_usage, gpu_info):
    """
    绘制资源使用情况的图表
    """
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # CPU 和内存使用情况
    ax[0].bar(["CPU Usage (%)", "Memory Usage (%)"], [cpu_usage, memory_usage], color=['blue', 'green'])
    ax[0].set_ylim(0, 100)
    ax[0].set_title("CPU and Memory Usage")

    # GPU 使用情况
    if gpu_info:
        gpu_names = [f"GPU {gpu['id']} ({gpu['name']})" for gpu in gpu_info]
        gpu_loads = [gpu["load"] for gpu in gpu_info]
        ax[1].bar(gpu_names, gpu_loads, color='orange')
        ax[1].set_ylim(0, 100)
        ax[1].set_title("GPU Usage")

    plt.tight_layout()
    return fig

def estimate_training_time(total_params, batch_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 256]):
    """
    根据模型参数和批次大小估算训练时间和内存占用
    """
    # 假设每个参数需要 4 字节（float32）
    param_size_bytes = total_params * 4

    # 估算内存占用（假设每个批次需要额外的内存）
    memory_usages = []
    for batch_size in batch_sizes:
        # 假设每个批次的内存占用与批次大小成正比
        memory_usage = param_size_bytes * batch_size / (1024 ** 2)  # 转换为 MB
        memory_usages.append(memory_usage)

    # 估算训练时间（假设训练时间与批次大小成正比）
    training_times = []
    for batch_size in batch_sizes:
        # 假设每个批次的训练时间与批次大小成正比
        time_per_batch = batch_size * 0.001  # 假设每个样本需要 1 毫秒
        training_times.append(time_per_batch)

    # 绘制柱状图
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))  # 调整图表大小

    # 生成随机颜色
    def generate_random_colors(n):
        colors = []
        for _ in range(n):
            # 随机生成 RGB 颜色
            color = (random.random(), random.random(), random.random())
            colors.append(color)
        return colors

    # 将批次大小转换为字符串，作为分类变量
    batch_labels = [str(size) for size in batch_sizes]

    # 内存占用图
    colors = generate_random_colors(len(batch_sizes))  # 使用随机颜色
    bars1 = ax[0].bar(batch_labels, memory_usages, color=colors, edgecolor='black')
    ax[0].set_xlabel("Batch Size")
    ax[0].set_ylabel("Memory Usage (MB)")
    ax[0].set_title("Memory Usage vs Batch Size")
    ax[0].grid(True, linestyle='--', alpha=0.6)
    # 在柱子上方添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax[0].text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f} MB',
                   ha='center', va='bottom', fontsize=9)

    # 训练时间图
    colors = generate_random_colors(len(batch_sizes))  # 使用另一种随机颜色
    bars2 = ax[1].bar(batch_labels, training_times, color=colors, edgecolor='black')
    ax[1].set_xlabel("Batch Size")
    ax[1].set_ylabel("Training Time per Batch (s)")
    ax[1].set_title("Training Time vs Batch Size")
    ax[1].grid(True, linestyle='--', alpha=0.6)
    # 在柱子上方添加数值标签
    for bar in bars2:
        height = bar.get_height()
        ax[1].text(bar.get_x() + bar.get_width() / 2, height, f'{height:.4f} s',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()  # 紧凑布局
    return fig, memory_usages, training_times