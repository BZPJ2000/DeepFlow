import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms

class InpaintingDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, image_size=(256, 256)):
        """
        初始化遮挡物修复数据集类。

        Args:
            image_dir (str): 包含原始图像文件的目录路径。
            mask_dir (str): 包含遮挡掩码文件的目录路径。
            transform (callable, optional): 可选的图像变换操作。
            image_size (tuple): 图像和掩码的目标尺寸，默认为(256, 256)。
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_size = image_size

        # 获取图像和掩码的文件名列表
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))

        # 确保图像和掩码文件一一对应
        assert len(self.image_filenames) == len(self.mask_filenames), "图像和掩码文件数量不匹配"

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
            dict: 包含原始图像、遮挡掩码和遮挡后图像的字典。
        """
        # 加载原始图像和遮挡掩码
        image_path = os.path.join(self.image_dir, self.image_filenames[index])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[index])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 转换为灰度图像

        # 调整图像和掩码的尺寸
        image = image.resize(self.image_size, Image.BILINEAR)
        mask = mask.resize(self.image_size, Image.NEAREST)  # 使用最近邻插值保持掩码的二进制信息

        # 将图像和掩码转换为numpy数组
        image = np.array(image)
        mask = np.array(mask)

        # 将掩码归一化为0和1（0表示遮挡区域，1表示非遮挡区域）
        mask = (mask > 0).astype(np.float32)

        # 生成遮挡后的图像
        masked_image = image * (1 - mask[..., np.newaxis])  # 将掩码扩展到3个通道

        # 应用变换（如果有）
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask, masked_image=masked_image)
            image = augmented['image']
            mask = augmented['mask']
            masked_image = augmented['masked_image']

        # 将图像、掩码和遮挡后的图像转换为PyTorch张量
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)
        masked_image = transforms.ToTensor()(masked_image)

        return {
            'image': image,  # 原始图像
            'mask': mask,  # 遮挡掩码
            'masked_image': masked_image  # 遮挡后的图像
        }

# 示例用法
if __name__ == "__main__":
    # 示例数据路径
    image_dir = "path/to/images"
    mask_dir = "path/to/masks"

    # 创建数据集实例
    dataset = InpaintingDataset(image_dir, mask_dir)

    sample = dataset[0]
    print("Original Image shape:", sample['image'].shape)
    print("Mask shape:", sample['mask'].shape)
    print("Masked Image shape:", sample['masked_image'].shape)