"""训练器模块

提供模型训练的核心功能。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable
from pathlib import Path


class Trainer:
    """训练器

    负责模型的训练和验证。
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda',
        metrics: Optional[Dict[str, Callable]] = None
    ):
        """初始化训练器

        Args:
            model: 模型实例
            loss_fn: 损失函数
            optimizer: 优化器
            device: 训练设备
            metrics: 评估指标字典
        """
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.metrics = metrics or {}

        self.history = {
            'train_loss': [],
            'val_loss': [],
        }

    def train_epoch(self, train_loader: DataLoader) -> float:
        """训练一个 epoch

        Args:
            train_loader: 训练数据加载器

        Returns:
            float: 平均训练损失
        """
        self.model.train()
        total_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader: DataLoader) -> float:
        """验证模型

        Args:
            val_loader: 验证数据加载器

        Returns:
            float: 平均验证损失
        """
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.loss_fn(output, target)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        save_dir: Optional[Path] = None
    ):
        """训练模型

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            save_dir: 模型保存目录
        """
        for epoch in range(epochs):
            # 训练
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)

            # 验证
            if val_loader:
                val_loss = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")

            # 保存检查点
            if save_dir and (epoch + 1) % 5 == 0:
                self.save_checkpoint(save_dir, epoch)

    def save_checkpoint(self, save_dir: Path, epoch: int):
        """保存检查点

        Args:
            save_dir: 保存目录
            epoch: 当前轮数
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }

        torch.save(checkpoint, save_dir / f'checkpoint_epoch_{epoch+1}.pth')
