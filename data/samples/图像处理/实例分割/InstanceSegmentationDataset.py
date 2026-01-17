import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
import json
from pycocotools.coco import COCO

class InstanceSegmentationDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None, image_size=(512, 512)):
        """
        初始化实例分割数据集类。

        Args:
            image_dir (str): 包含图像文件的目录路径。
            annotation_file (str): COCO格式的标注文件路径（JSON文件）。
            transform (callable, optional): 可选的图像变换操作。
            image_size (tuple): 图像的目标尺寸，默认为(512, 512)。
        """
        self.image_dir = image_dir
        self.annotation_file = annotation_file
        self.transform = transform
        self.image_size = image_size

        # 加载COCO标注文件
        self.coco = COCO(annotation_file)
        self.image_ids = self.coco.getImgIds()

    def __len__(self):
        """
        返回数据集的样本数量。
        """
        return len(self.image_ids)

    def __getitem__(self, index):
        """
        根据索引获取一个样本。

        Args:
            index (int): 样本的索引。

        Returns:
            dict: 包含图像、边界框、类别标签和分割掩码的字典。
        """
        # 获取图像ID和图像信息
        image_id = self.image_ids[index]
        image_info = self.coco.loadImgs(image_id)[0]
        image_filename = image_info['file_name']
        image_path = os.path.join(self.image_dir, image_filename)

        # 加载图像
        image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.size

        # 调整图像尺寸
        image = image.resize(self.image_size, Image.BILINEAR)
        new_width, new_height = self.image_size

        # 获取标注信息
        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(annotation_ids)

        # 初始化边界框、类别标签和分割掩码
        boxes = []
        labels = []
        masks = []

        for ann in annotations:
            # 类别标签
            label = ann['category_id']
            labels.append(label)

            # 边界框 (xmin, ymin, width, height) -> (xmin, ymin, xmax, ymax)
            x, y, w, h = ann['bbox']
            xmin = x
            ymin = y
            xmax = x + w
            ymax = y + h
            boxes.append([xmin, ymin, xmax, ymax])

            # 分割掩码
            mask = self.coco.annToMask(ann)  # 将COCO多边形标注转换为二值掩码
            mask = Image.fromarray(mask).resize(self.image_size, Image.NEAREST)  # 调整掩码尺寸
            masks.append(np.array(mask))

        # 将边界框、标签和掩码转换为numpy数组
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        masks = np.stack(masks, axis=0) if masks else np.zeros((0, *self.image_size), dtype=np.uint8)

        # 缩放边界框坐标
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * (new_width / original_width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * (new_height / original_height)

        # 将图像转换为numpy数组
        image = np.array(image)

        # 应用变换（如果有）
        if self.transform is not None:
            augmented = self.transform(image=image, bboxes=boxes, masks=masks, labels=labels)
            image = augmented['image']
            boxes = augmented['bboxes']
            masks = augmented['masks']
            labels = augmented['labels']

        # 将图像、边界框、标签和掩码转换为PyTorch张量
        image = transforms.ToTensor()(image)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        masks = torch.tensor(masks, dtype=torch.uint8)

        return {
            'image': image,  # 图像张量 (C, H, W)
            'boxes': boxes,  # 边界框张量 (N, 4)
            'labels': labels,  # 类别标签张量 (N,)
            'masks': masks  # 分割掩码张量 (N, H, W)
        }

# 示例用法
if __name__ == "__main__":
    # 示例数据路径
    image_dir = "E:/Github_Project/troch_GUI/dataset/图像处理/实例分割/images"
    annotation_file = "E:/Github_Project/troch_GUI/dataset/图像处理/实例分割/annotations.json"

    # 创建数据集实例
    dataset = InstanceSegmentationDataset(image_dir, annotation_file)

    sample = dataset[0]
    print("Image shape:", sample['image'].shape)
    print("Boxes:", sample['boxes'])
    print("Labels:", sample['labels'])
    print("Masks shape:", sample['masks'].shape)


