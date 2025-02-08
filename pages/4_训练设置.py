import os
import time
import json
from datetime import datetime
import streamlit as st
from ulit.load_loss_function import load_loss_function
from ulit.load_metric_function import load_metric_function
from ulit.load_optimizer import load_optimizer

# 页面标题
st.title("训练设置")

# 从 session_state 中获取用户选择的任务类型和小类
selected_category = st.session_state.get("selected_category")
selected_subcategory = st.session_state.get("selected_subcategory")

# 检查任务类型是否已选择
if not selected_category or not selected_subcategory:
    st.error("请先返回任务选择页面选择任务类型。")
    st.stop()

# 显示任务类型和小类
st.write(f"你选择的任务: {selected_category} -> {selected_subcategory}")

# 从 session_state 中获取数据处理设置
settings = st.session_state.get("settings", {})
dataset_path = settings.get("dataset_path")
data_split_option = settings.get("data_split_option")
train_percent = settings.get("train_percent")
val_percent = settings.get("val_percent")
test_percent = settings.get("test_percent")
shuffle_data = settings.get("shuffle_data")
selected_augmentations = settings.get("selected_augmentations")

# 显示数据处理设置
st.divider()
st.header("数据处理设置")
st.write(f"数据集路径: {dataset_path}")
st.write(f"数据集分割选项: {data_split_option}")
st.write(f"训练集比例: {train_percent}%")
st.write(f"验证集比例: {val_percent}%")
st.write(f"测试集比例: {test_percent}%")
st.write(f"打乱数据集: {shuffle_data}")
st.write(f"选择的数据增强方式: {selected_augmentations}")

# 评估指标、损失函数和优化器选择
st.divider()
col1, col2 = st.columns(2)

with col1:
    st.header("评估指标")
    metric_classes = load_metric_function()

with col2:
    st.header("损失函数")
    loss_class = load_loss_function()

st.header("优化器")
optimizer_class = load_optimizer()

# 训练参数设置
st.divider()
st.header("训练参数")
learning_rate = st.number_input("学习率", min_value=0.0001, max_value=1.0, value=0.001, step=0.0001)
batch_size = st.number_input("批次大小", min_value=1, max_value=1024, value=32, step=1)

# 将设置保存到 session_state
if optimizer_class:
    st.session_state["optimizer_class"] = optimizer_class
if loss_class:
    st.session_state["loss_class"] = loss_class
if metric_classes:
    st.session_state["metric_classes"] = metric_classes
st.session_state["learning_rate"] = learning_rate
st.session_state["batch_size"] = batch_size

# 保存所有设置到文件
def save_all_settings_to_file():
    """将当前所有设置保存到 JSON 文件中"""
    # 确保 logs 文件夹存在
    os.makedirs("logs", exist_ok=True)

    # 生成唯一的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join("logs", f"settings_{timestamp}.json")

    # 将类对象转换为字符串
    optimizer_class_name = optimizer_class.__name__ if optimizer_class else None
    loss_class_name = loss_class.__name__ if loss_class else None
    metric_classes_names = [metric.__name__ for metric in metric_classes] if metric_classes else None

    # 从 session_state 中提取所有相关值
    all_settings = {
        "selected_category": selected_category,
        "selected_subcategory": selected_subcategory,
        "dataset_path": dataset_path,
        "data_split_option": data_split_option,
        "train_percent": train_percent,
        "val_percent": val_percent,
        "test_percent": test_percent,
        "shuffle_data": shuffle_data,
        "selected_augmentations": selected_augmentations,
        "optimizer_class": optimizer_class_name,  # 保存类名
        "loss_class": loss_class_name,  # 保存类名
        "metric_classes": metric_classes_names,  # 保存类名列表
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        # 其他可能的参数
        "total_params": st.session_state.get("total_params"),
        "trainable_params": st.session_state.get("trainable_params"),
        "cpu_usage": st.session_state.get("cpu_usage"),
        "memory_usage": st.session_state.get("memory_usage"),
        "gpu_info": st.session_state.get("gpu_info"),
        "batch_sizes": st.session_state.get("batch_sizes"),
        "memory_usages": st.session_state.get("memory_usages"),
        "training_times": st.session_state.get("training_times")
    }

    # 保存数据到文件
    with open(filename, "w") as f:
        json.dump(all_settings, f, indent=4)

    return filename

# 保存所有设置
if st.button("保存所有设置"):
    filename = save_all_settings_to_file()
    st.success(f"所有设置已保存到文件: {filename}")

# 训练日志
st.divider()
st.header("训练日志")
log_area = st.empty()  # 使用 st.empty() 创建一个占位符

import torch
from torch.utils.data import DataLoader
import numpy as np
from matplotlib import pyplot as plt

# 添加在文件开头部分的常量
LOG_SAVE_PATH = "./training_logs"
MODEL_SAVE_PATH = "./saved_models"
os.makedirs(LOG_SAVE_PATH, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# 修改原有的"开始训练"按钮部分
if st.button("开始训练"):
    # 初始化训练状态
    st.session_state.setdefault('training_logs', [])
    st.session_state.setdefault('training_running', True)
    st.session_state.setdefault('current_epoch', 0)

    try:
        # 获取所有必要组件
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. 数据准备
        DatasetClass = st.session_state.settings['dataset_class']
        train_dataset = DatasetClass(st.session_state.train_path, augmentations=selected_augmentations)
        val_dataset = DatasetClass(st.session_state.val_path) if st.session_state.val_path else None
        test_dataset = DatasetClass(st.session_state.test_path)

        batch_size = st.session_state.batch_size
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # 2. 模型准备
        if st.session_state.get('is_dual_model_task', False):
            generator = st.session_state.generator_model.to(device)
            discriminator = st.session_state.discriminator_model.to(device)
            # 双模型优化器设置（示例）
            g_optimizer = st.session_state.optimizer_class(generator.parameters(), lr=learning_rate)
            d_optimizer = st.session_state.optimizer_class(discriminator.parameters(), lr=learning_rate)
        else:
            model = st.session_state.model.to(device)
            optimizer = st.session_state.optimizer_class(model.parameters(), lr=learning_rate)

        criterion = st.session_state.loss_class()

        # 3. 训练循环
        total_epochs = 10  # 可以改为用户配置
        progress_bar = st.progress(0)
        loss_chart = st.line_chart()
        metric_chart = st.line_chart()

        for epoch in range(st.session_state.current_epoch, total_epochs):
            if not st.session_state.training_running:
                break

            epoch_losses = []
            epoch_metrics = []

            # 训练阶段
            model.train()
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                # 计算指标
                with torch.no_grad():
                    preds = torch.argmax(outputs, dim=1)
                    accuracy = (preds == targets).float().mean()

                # 更新日志
                log_entry = f"Epoch {epoch + 1}/{total_epochs} | Batch {batch_idx + 1}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {accuracy:.4f}"
                st.session_state.training_logs.append(log_entry)
                if len(st.session_state.training_logs) > 50:
                    st.session_state.training_logs.pop(0)

                # 更新图表
                epoch_losses.append(loss.item())
                epoch_metrics.append(accuracy.item())

            # 验证阶段
            if val_loader:
                model.eval()
                val_loss = 0
                val_acc = 0
                with torch.no_grad():
                    for data, targets in val_loader:
                        data, targets = data.to(device), targets.to(device)
                        outputs = model(data)
                        val_loss += criterion(outputs, targets).item()
                        preds = torch.argmax(outputs, dim=1)
                        val_acc += (preds == targets).float().mean().item()

                val_loss /= len(val_loader)
                val_acc /= len(val_loader)
                log_entry = f"Validation | Loss: {val_loss:.4f} | Acc: {val_acc:.4f}"
                st.session_state.training_logs.append(log_entry)

            # 更新进度和图表
            progress = (epoch + 1) / total_epochs
            progress_bar.progress(progress)

            # 更新损失和指标图表
            avg_loss = np.mean(epoch_losses)
            avg_metric = np.mean(epoch_metrics)
            loss_chart.add_rows([{"loss": avg_loss}])
            metric_chart.add_rows([{"accuracy": avg_metric}])

            # 保存检查点
            if (epoch + 1) % 5 == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = os.path.join(MODEL_SAVE_PATH, f"model_epoch{epoch + 1}_{timestamp}.pth")
                torch.save(model.state_dict(), model_path)
                st.session_state.training_logs.append(f"Model saved at {model_path}")

            st.session_state.current_epoch = epoch + 1
            st.experimental_rerun()

        # 训练完成
        st.session_state.training_running = False
        st.success("训练完成！")

        # 保存最终模型
        final_model_path = os.path.join(MODEL_SAVE_PATH, f"final_model_{timestamp}.pth")
        torch.save(model.state_dict(), final_model_path)
        st.session_state.training_logs.append(f"Final model saved at {final_model_path}")

    except Exception as e:
        st.error(f"训练出错: {str(e)}")
        st.session_state.training_running = False

# 添加停止训练按钮
if st.button("停止训练") and st.session_state.get('training_running', False):
    st.session_state.training_running = False
    st.warning("训练正在停止...")

# 实时日志显示
if st.session_state.get('training_logs'):
    st.subheader("训练日志（最近50条）")
    log_text = "\n".join(st.session_state.training_logs[-50:])
    st.text_area("日志内容", value=log_text, height=300, key="training_logs_area")