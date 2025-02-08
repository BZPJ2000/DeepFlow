import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
class ImageClassification_Trainer:
    def __init__(self, model, dataset_class, train_path, val_path=None, test_path=None,
                 optimizer_class=torch.optim.Adam,
                 loss_class=torch.nn.CrossEntropyLoss, metric_classes=[], learning_rate=0.001, batch_size=32,
                 num_epochs=10, device=None):
        """
        初始化训练器
        :param model: 模型实例
        :param dataset_class: 数据集类
        :param train_path: 训练集路径
        :param val_path: 验证集路径（可选）
        :param test_path: 测试集路径（可选）
        :param optimizer_class: 优化器类，默认为 Adam
        :param loss_class: 损失函数类，默认为 CrossEntropyLoss
        :param metric_classes: 评价指标类列表
        :param learning_rate: 学习率，默认为 0.001
        :param batch_size: 批次大小，默认为 32
        :param num_epochs: 训练轮数，默认为 10
        :param device: 训练设备（如 'cuda' 或 'cpu'），默认为 None（自动选择）
        """
        self.model = model
        self.dataset_class = dataset_class
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.optimizer_class = optimizer_class
        self.loss_class = loss_class
        self.metric_classes = metric_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化优化器和损失函数
        self.optimizer = self.optimizer_class(self.model.parameters(), lr=self.learning_rate)
        self.criterion = self.loss_class()

        # 加载数据集
        self.train_dataset = self.dataset_class(self.train_path)
        self.val_dataset = self.dataset_class(self.val_path) if self.val_path else None
        self.test_dataset = self.dataset_class(self.test_path) if self.test_path else None

        # 创建 DataLoader
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size,
                                     shuffle=False) if self.val_dataset else None
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size,
                                      shuffle=False) if self.test_dataset else None

        # 将模型移动到设备
        self.model = self.model.to(self.device)

    def train(self):
        """
        训练模型
        """
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_metrics = {metric.__name__: 0.0 for metric in self.metric_classes}

            # 训练阶段
            for batch_idx, (inputs, targets) in enumerate(tqdm(self.train_loader)):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # 前向传播
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 计算损失和指标
                epoch_loss += loss.item()
                for metric in self.metric_classes:
                    epoch_metrics[metric.__name__] += metric(outputs, targets).item()

            # 计算平均损失和指标
            epoch_loss /= len(self.train_loader)
            for metric_name in epoch_metrics:
                epoch_metrics[metric_name] /= len(self.train_loader)

            # 验证阶段
            if self.val_loader:
                self.model.eval()
                val_loss = 0.0
                val_metrics = {metric.__name__: 0.0 for metric in self.metric_classes}

                with torch.no_grad():
                    for inputs, targets in self.val_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        outputs = self.model(inputs)
                        val_loss += self.criterion(outputs, targets).item()
                        for metric in self.metric_classes:
                            val_metrics[metric.__name__] += metric(outputs, targets).item()

                val_loss /= len(self.val_loader)
                for metric_name in val_metrics:
                    val_metrics[metric_name] /= len(self.val_loader)

                # 打印验证结果
                print(f"Epoch [{epoch + 1}/{self.num_epochs}], Val Loss: {val_loss:.4f}, Val Metrics: {val_metrics}")

            # 打印每个 epoch 的结果
            print(
                f"Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {epoch_loss:.4f}, Train Metrics: {epoch_metrics}")

    def evaluate(self):
        """
        在测试集上评估模型
        """
        if not self.test_loader:
            print("没有提供测试集路径，无法进行评估。")
            return

        self.model.eval()
        test_loss = 0.0
        test_metrics = {metric.__name__: 0.0 for metric in self.metric_classes}

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                test_loss += self.criterion(outputs, targets).item()
                for metric in self.metric_classes:
                    test_metrics[metric.__name__] += metric(outputs, targets).item()

        test_loss /= len(self.test_loader)
        for metric_name in test_metrics:
            test_metrics[metric_name] /= len(self.test_loader)

        # 打印测试结果
        print(f"Test Loss: {test_loss:.4f}, Test Metrics: {test_metrics}")