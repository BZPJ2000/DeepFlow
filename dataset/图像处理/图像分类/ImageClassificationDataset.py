import os

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class ImageClassificationDataset(Dataset):
    def __init__(self, data_dir, transform=None, image_size=(224, 224)):
        """
        初始化图像分类数据集类。

        Args:
            data_dir (str): 包含图像文件的目录路径（目录结构应为 class1/, class2/, ...）。
            transform (callable, optional): 可选的图像变换操作。
            image_size (tuple): 图像的目标尺寸，默认为(224, 224)。
        """
        self.data_dir = data_dir
        self.transform = transform
        self.image_size = image_size

        # 获取类别列表
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # 获取图像路径和对应的类别标签
        self.image_paths = []
        self.labels = []
        for cls in self.classes:
            cls_dir = os.path.join(data_dir, cls)
            for img_name in os.listdir(cls_dir):
                self.image_paths.append(os.path.join(cls_dir, img_name))
                self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        """
        返回数据集的样本数量。
        """
        return len(self.image_paths)

    def __getitem__(self, index):
        """
        根据索引获取一个样本。

        Args:
            index (int): 样本的索引。

        Returns:
            dict: 包含图像和标签的字典。
        """
        # 加载图像
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")

        # 调整图像尺寸
        image = image.resize(self.image_size, Image.BILINEAR)

        # 获取标签
        label = self.labels[index]

        # 将图像转换为numpy数组
        image = np.array(image)

        # 应用变换（如果有）
        if self.transform is not None:
            image = self.transform(image=image)['image']

        # 将图像转换为PyTorch张量
        image = transforms.ToTensor()(image)

        return {
            'image': image,  # 图像张量 (C, H, W)
            'label': label  # 标签
        }

# 示例用法
if __name__ == "__main__":
    # 示例数据路径
    data_dir = "E:/Github_Project/troch_GUI/dataset/图像处理/图像分类/train"

    # 创建数据集实例
    dataset = ImageClassificationDataset(data_dir)

    sample = dataset[0]
    print("Image shape:", sample['image'].shape)
    print("Label:", sample['label'])