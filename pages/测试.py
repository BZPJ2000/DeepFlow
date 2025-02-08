import os
import time
import json
from datetime import datetime
import streamlit as st
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

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

# 从 session_state 中获取模型、优化器、损失函数等
model = st.session_state.get("model")  # 假设模型已经加载到 session_state
optimizer_class = st.session_state.get("optimizer_class")
loss_class = st.session_state.get("loss_class")
metric_classes = st.session_state.get("metric_classes")
learning_rate = st.session_state.get("learning_rate")
batch_size = st.session_state.get("batch_size")
train_path = st.session_state.get("train_path")
val_path = st.session_state.get("val_path")
test_path = st.session_state.get("test_path")
dataset_class = st.session_state.get("dataset_class")

# 检查是否所有必要的组件都已加载
if not all([model, optimizer_class, loss_class, metric_classes, learning_rate, batch_size, train_path, dataset_class]):
    st.error("缺少必要的组件，请确保模型、优化器、损失函数、数据集等已正确加载。")
    st.stop()

# 初始化优化器和损失函数
optimizer = optimizer_class(model.parameters(), lr=learning_rate)
criterion = loss_class()

# 加载数据集
train_dataset = dataset_class(train_path)
val_dataset = dataset_class(val_path) if val_path else None
test_dataset = dataset_class(test_path) if test_path else None

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) if test_dataset else None

# 训练日志
st.divider()
st.header("训练日志")
log_area = st.empty()  # 使用 st.empty() 创建一个占位符
log_messages = []


# 训练函数
def train_model(model, train_loader, val_loader, optimizer, criterion, metric_classes, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_metrics = {metric.__name__: 0.0 for metric in metric_classes}

        # 训练阶段
        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
            inputs, targets = inputs.to(device), targets.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 计算损失和指标
            epoch_loss += loss.item()
            for metric in metric_classes:
                epoch_metrics[metric.__name__] += metric(outputs, targets).item()

            # 更新日志
            log_message = f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}"
            log_messages.append(log_message)
            if len(log_messages) > 50:
                log_messages.pop(0)
            log_area.text_area("日志输出", value="\n".join(log_messages[-50:]), height=200)

        # 计算平均损失和指标
        epoch_loss /= len(train_loader)
        for metric_name in epoch_metrics:
            epoch_metrics[metric_name] /= len(train_loader)

        # 验证阶段
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_metrics = {metric.__name__: 0.0 for metric in metric_classes}

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item()
                    for metric in metric_classes:
                        val_metrics[metric.__name__] += metric(outputs, targets).item()

            val_loss /= len(val_loader)
            for metric_name in val_metrics:
                val_metrics[metric_name] /= len(val_loader)

            # 更新日志
            log_message = f"Epoch [{epoch + 1}/{num_epochs}], Val Loss: {val_loss:.4f}, Val Metrics: {val_metrics}"
            log_messages.append(log_message)
            if len(log_messages) > 50:
                log_messages.pop(0)
            log_area.text_area("日志输出", value="\n".join(log_messages[-50:]), height=200)

        # 打印每个 epoch 的结果
        log_message = f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Metrics: {epoch_metrics}"
        log_messages.append(log_message)
        if len(log_messages) > 50:
            log_messages.pop(0)
        log_area.text_area("日志输出", value="\n".join(log_messages[-50:]), height=200)


# 开始训练按钮
if st.button("开始训练"):
    st.write("训练开始...")
    train_model(model, train_loader, val_loader, optimizer, criterion, metric_classes, num_epochs=10)
    st.write("训练完成！")