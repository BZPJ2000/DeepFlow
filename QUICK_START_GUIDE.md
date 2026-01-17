# DeepFlow 快速开始指南

## 快速开始与部署

---

## 1. 环境准备

### 1.1 系统要求

**最低配置:**
- Python 3.8+
- 8GB RAM
- 10GB 磁盘空间

**推荐配置:**
- Python 3.9+
- 16GB RAM
- NVIDIA GPU (8GB+ VRAM)
- 50GB 磁盘空间

### 1.2 安装步骤

```bash
# 克隆项目
git clone https://github.com/yourusername/DeepFlow.git
cd DeepFlow

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 验证安装
python scripts/check_dependencies.py
```

---

## 2. 5 分钟快速体验

### 2.1 启动应用

```bash
# 启动 Streamlit 应用
streamlit run app.py

# 或使用新版实验管理器
streamlit run ui/app.py
```

### 2.2 创建第一个实验

**步骤 1: 选择任务**
- 打开浏览器访问 http://localhost:8501
- 选择 "图像处理" → "图像分类"

**步骤 2: 选择模型**
- 浏览可用模型列表
- 选择 "ResNet50"
- 查看模型参数和资源需求

**步骤 3: 配置数据**
- 设置数据路径: `data/samples/cifar10`
- 训练集比例: 80%
- 验证集比例: 10%
- 测试集比例: 10%

**步骤 4: 训练设置**
- 损失函数: CrossEntropyLoss
- 优化器: Adam (lr=0.001)
- Batch Size: 32
- Epochs: 10

**步骤 5: 开始训练**
- 点击 "开始训练"
- 实时查看训练进度
- 查看损失曲线和指标

---

## 3. 使用 API 方式

### 3.1 基础使用

```python
from deepflow.api.experiment import ExperimentAPI

# 创建 API 实例
api = ExperimentAPI()

# 获取可用模型
models = api.get_available_models(
    category='vision',
    subcategory='classification'
)

print(f"找到 {len(models)} 个模型")
for model in models[:5]:
    print(f"- {model.name}: {model.description}")
```

### 3.2 创建实验

```python
# 配置实验
config = {
    'name': 'my_first_experiment',
    'task': {
        'category': 'vision',
        'subcategory': 'classification'
    },
    'model': {
        'name': 'ResNet50',
        'params': {'num_classes': 10}
    },
    'data': {
        'path': 'data/samples/cifar10',
        'split': {'train': 0.8, 'val': 0.1, 'test': 0.1},
        'batch_size': 32
    },
    'training': {
        'loss': 'CrossEntropyLoss',
        'optimizer': 'Adam',
        'optimizer_params': {'lr': 0.001},
        'epochs': 10,
        'device': 'cuda'
    }
}

# 创建并运行实验
experiment = api.create_experiment(config)
results = experiment.run()

print(f"训练完成!")
print(f"最终损失: {results['final_loss']:.4f}")
print(f"最佳准确率: {results['best_accuracy']:.2%}")
```

### 3.3 加载已保存的实验

```python
# 加载实验
experiment = api.load_experiment('my_first_experiment')

# 查看结果
history = experiment.get_history()
print(f"训练轮数: {len(history['train_loss'])}")

# 可视化
experiment.plot_history()

# 导出结果
experiment.export_results('results/my_experiment.json')
```

---

## 4. 命令行工具

### 4.1 组件管理

```bash
# 列出所有可用模型
python -m deepflow.cli list models

# 搜索组件
python -m deepflow.cli search "resnet"

# 查看组件详情
python -m deepflow.cli info ResNet50

# 验证组件
python -m deepflow.cli validate library/models/vision/resnet/
```

### 4.2 实验管理

```bash
# 从配置文件运行实验
python -m deepflow.cli run configs/experiments/example.yaml

# 列出所有实验
python -m deepflow.cli experiments list

# 查看实验详情
python -m deepflow.cli experiments show my_first_experiment

# 比较实验
python -m deepflow.cli experiments compare exp1 exp2 exp3
```

---

## 5. 配置文件示例

### 5.1 实验配置

```yaml
# configs/experiments/image_classification.yaml

name: "cifar10_resnet50"
description: "CIFAR-10 图像分类实验"

task:
  category: "vision"
  subcategory: "classification"

model:
  name: "ResNet50"
  params:
    num_classes: 10
    pretrained: false

data:
  path: "data/cifar10"
  split:
    train: 0.8
    val: 0.1
    test: 0.1
  batch_size: 32
  num_workers: 4
  augmentation:
    - RandomHorizontalFlip
    - RandomCrop:
        size: 32
        padding: 4
    - Normalize:
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]

training:
  loss: "CrossEntropyLoss"
  optimizer: "Adam"
  optimizer_params:
    lr: 0.001
    weight_decay: 0.0001
  scheduler: "StepLR"
  scheduler_params:
    step_size: 30
    gamma: 0.1
  epochs: 100
  device: "cuda"
  mixed_precision: true
  gradient_accumulation: 1

callbacks:
  - EarlyStopping:
      patience: 10
      min_delta: 0.001
  - ModelCheckpoint:
      save_best: true
      monitor: "val_accuracy"
  - TensorBoard:
      log_dir: "outputs/tensorboard"

output:
  save_dir: "outputs/experiments/cifar10_resnet50"
  save_model: true
  save_history: true
```

---

## 6. Docker 部署

### 6.1 Dockerfile

```dockerfile
# Dockerfile

FROM python:3.9-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 暴露端口
EXPOSE 8501

# 启动命令
CMD ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 6.2 Docker Compose

```yaml
# docker-compose.yml

version: '3.8'

services:
  deepflow:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./outputs:/app/outputs
      - ./library:/app/library
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 6.3 使用 Docker

```bash
# 构建镜像
docker build -t deepflow:latest .

# 运行容器
docker run -p 8501:8501 -v $(pwd)/data:/app/data deepflow:latest

# 使用 Docker Compose
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

---

## 7. 常见问题

### 7.1 CUDA 相关

**问题: CUDA out of memory**
```python
# 解决方案 1: 减小 batch size
config['data']['batch_size'] = 16

# 解决方案 2: 使用梯度累积
config['training']['gradient_accumulation'] = 4

# 解决方案 3: 使用混合精度
config['training']['mixed_precision'] = True
```

### 7.2 组件未发现

**问题: 模型未被自动发现**
```bash
# 检查文件结构
python -m deepflow.cli validate library/models/

# 手动触发发现
python -m deepflow.cli discover --force

# 清除缓存
rm .deepflow_cache.json
```

### 7.3 性能问题

**问题: 数据加载慢**
```python
# 增加工作进程
config['data']['num_workers'] = 8

# 启用固定内存
config['data']['pin_memory'] = True

# 使用预取
config['data']['prefetch_factor'] = 4
```

---

## 8. 下一步

### 学习资源
- 📖 [用户指南](docs/user_guide.md)
- 🔧 [开发者指南](docs/developer_guide.md)
- 📚 [API 参考](docs/api_reference.md)
- 🏗️ [架构文档](ARCHITECTURE_DESIGN.md)

### 示例项目
- [图像分类](examples/image_classification/)
- [目标检测](examples/object_detection/)
- [文本分类](examples/text_classification/)
- [图神经网络](examples/graph_neural_networks/)

### 社区
- GitHub Issues: 报告问题
- Discussions: 讨论交流
- Wiki: 知识库

---

**祝你使用愉快！** 🚀
