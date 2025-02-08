import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
import xml.etree.ElementTree as ET

class ObjectDetectionDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None, image_size=(416, 416)):
        """
        初始化目标检测数据集类。

        Args:
            image_dir (str): 包含图像文件的目录路径。
            annotation_dir (str): 包含标注文件（XML格式）的目录路径。
            transform (callable, optional): 可选的图像变换操作。
            image_size (tuple): 图像的目标尺寸，默认为(416, 416)。
        """
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.image_size = image_size

        # 获取图像和标注的文件名列表
        self.image_filenames = sorted(os.listdir(image_dir))
        self.annotation_filenames = sorted(os.listdir(annotation_dir))

        # 确保图像和标注文件一一对应
        assert len(self.image_filenames) == len(self.annotation_filenames), "图像和标注文件数量不匹配"

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
            dict: 包含图像、边界框和类别标签的字典。
        """
        # 加载图像
        image_path = os.path.join(self.image_dir, self.image_filenames[index])
        image = Image.open(image_path).convert("RGB")

        # 加载标注文件（XML格式）
        annotation_path = os.path.join(self.annotation_dir, self.annotation_filenames[index])
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        # 解析标注文件中的边界框和类别标签
        boxes = []
        labels = []
        for obj in root.findall("object"):
            # 类别标签
            label = obj.find("name").text
            labels.append(label)

            # 边界框坐标（xmin, ymin, xmax, ymax）
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])

        # 将边界框和标签转换为numpy数组
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels)

        # 调整图像尺寸并相应地缩放边界框
        original_width, original_height = image.size
        image = image.resize(self.image_size, Image.BILINEAR)
        new_width, new_height = self.image_size

        # 缩放边界框坐标
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * (new_width / original_width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * (new_height / original_height)

        # 将图像转换为numpy数组
        image = np.array(image)

        # 应用变换（如果有）
        if self.transform is not None:
            augmented = self.transform(image=image, bboxes=boxes, labels=labels)
            image = augmented['image']
            boxes = augmented['bboxes']
            labels = augmented['labels']

        # 将图像、边界框和标签转换为PyTorch张量
        image = transforms.ToTensor()(image)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        return {
            'image': image,  # 图像张量 (C, H, W)
            'boxes': boxes,  # 边界框张量 (N, 4)
            'labels': labels  # 类别标签张量 (N,)
        }

# 示例用法
if __name__ == "__main__":
    # 示例数据路径
    image_dir = "path/to/images"
    annotation_dir = "path/to/annotations"

    # 创建数据集实例
    dataset = ObjectDetectionDataset(image_dir, annotation_dir)

    sample = dataset[0]
    print("Image shape:", sample['image'].shape)
    print("Boxes:", sample['boxes'])
    print("Labels:", sample['labels'])