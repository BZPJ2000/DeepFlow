import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import Module
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, criterion, optimizer, metrics, device='cuda', batch_size=8, num_epochs=10):
        """
        初始化训练器。

        Args:
            model (Module): 模型实例。
            train_dataset (Dataset): 训练数据集。
            val_dataset (Dataset): 验证数据集。
            criterion (Module): 损失函数。
            optimizer (Optimizer): 优化器。
            metrics (dict): 评价指标字典，键为指标名称，值为计算函数。
            device (str): 设备类型，默认为 'cuda'。
            batch_size (int): 批量大小，默认为 8。
            num_epochs (int): 训练轮数，默认为 10。
        """
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics = metrics
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        # 创建数据加载器
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    def train_one_epoch(self):
        """
        训练一个 epoch。
        """
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)

        for batch in progress_bar:
            inputs = batch['image'].to(self.device)
            targets = batch['mask'].to(self.device)

            # 前向传播
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 记录损失
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        return total_loss / len(self.train_loader)

    def validate(self):
        """
        在验证集上评估模型。
        """
        self.model.eval()
        total_loss = 0.0
        metric_results = {name: 0.0 for name in self.metrics.keys()}

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                inputs = batch['image'].to(self.device)
                targets = batch['mask'].to(self.device)

                # 前向传播
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                # 计算评价指标
                for name, metric_fn in self.metrics.items():
                    metric_results[name] += metric_fn(outputs, targets).item()

        # 计算平均损失和指标
        avg_loss = total_loss / len(self.val_loader)
        avg_metrics = {name: value / len(self.val_loader) for name, value in metric_results.items()}

        return avg_loss, avg_metrics

    def train(self):
        """
        训练模型。
        """
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            train_loss = self.train_one_epoch()
            val_loss, val_metrics = self.validate()

            # 打印训练和验证结果
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            for name, value in val_metrics.items():
                print(f"Validation {name}: {value:.4f}")

# 示例用法
if __name__ == "__main__":
    import torch.nn as nn
    from torch.optim import Adam
    from torchvision.models.segmentation import fcn_resnet50

    # 自定义数据集类（假设已定义）
    from dataset import SemanticSegmentationDataset

    # 定义评价指标（示例：IoU）
    def iou_score(outputs, targets):
        intersection = (outputs.argmax(dim=1) & targets).float().sum((1, 2))
        union = (outputs.argmax(dim=1) | targets).float().sum((1, 2))
        iou = (intersection + 1e-6) / (union + 1e-6)  # 避免除零
        return iou.mean()

    # 数据集路径
    train_image_dir = "E:/Github_Project/troch_GUI/train/图像处理/语义分割/train_images"
    train_mask_dir = "E:/Github_Project/troch_GUI/train/图像处理/语义分割/train_masks"
    val_image_dir = "E:/Github_Project/troch_GUI/train/图像处理/语义分割/val_images"
    val_mask_dir = "E:/Github_Project/troch_GUI/train/图像处理/语义分割/val_masks"

    # 创建数据集实例
    train_dataset = SemanticSegmentationDataset(train_image_dir, train_mask_dir)
    val_dataset = SemanticSegmentationDataset(val_image_dir, val_mask_dir)

    # 模型实例化
    model = fcn_resnet50(num_classes=21)  # 假设有 21 个类别

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 优化器
    optimizer = Adam(model.parameters(), lr=1e-4)

    # 评价指标
    metrics = {'IoU': iou_score}

    # 创建训练器并开始训练
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        criterion=criterion,
        optimizer=optimizer,
        metrics=metrics,
        device='cuda',
        batch_size=8,
        num_epochs=10
    )
    trainer.train()