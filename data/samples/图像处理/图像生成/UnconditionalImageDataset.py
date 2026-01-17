import os

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class UnconditionalImageDataset(Dataset):
    def __init__(self, image_dir, transform=None, image_size=(64, 64)):
        """
        初始化无条件图像生成数据集类。

        Args:
            image_dir (str): 包含图像文件的目录路径。
            transform (callable, optional): 可选的图像变换操作。
            image_size (tuple): 图像的目标尺寸，默认为(64, 64)。
        """
        self.image_dir = image_dir
        self.transform = transform
        self.image_size = image_size

        # 获取图像文件名列表
        self.image_filenames = sorted(os.listdir(image_dir))

    def __len__(self):
        """
        返回数据集的样本数量。
        """
        return len(self.image_filenames)

    def __getitem__(self, index):
        """
        根据索引获取一个样本。

        Args:
            index (int): 样本的索引。

        Returns:
            torch.Tensor: 图像张量。
        """
        # 加载图像
        image_path = os.path.join(self.image_dir, self.image_filenames[index])
        image = Image.open(image_path).convert("RGB")

        # 调整图像尺寸
        image = image.resize(self.image_size, Image.BILINEAR)

        # 将图像转换为numpy数组
        image = np.array(image)

        # 应用变换（如果有）
        if self.transform is not None:
            image = self.transform(image=image)['image']

        # 将图像转换为PyTorch张量
        image = transforms.ToTensor()(image)

        return image

# 示例用法
if __name__ == "__main__":
    # 示例数据路径
    image_dir = "path/to/images"

    # 创建数据集实例
    dataset = UnconditionalImageDataset(image_dir)

    sample = dataset[0]
    print("Image shape:", sample.shape)