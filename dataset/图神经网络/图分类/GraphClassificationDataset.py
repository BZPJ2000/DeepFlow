import os
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
import numpy as np

class GraphClassificationDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        """
        初始化图分类数据集类。

        Args:
            root (str): 数据集的根目录。
            transform (callable, optional): 图数据的实时变换操作。
            pre_transform (callable, optional): 图数据的预处理操作。
        """
        super(GraphClassificationDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """
        返回原始数据文件的文件名列表。
        """
        return ['data_1.pt', 'data_2.pt', 'data_3.pt']  # 示例文件名

    @property
    def processed_file_names(self):
        """
        返回处理后的数据文件的文件名列表。
        """
        return ['data_1.pt', 'data_2.pt', 'data_3.pt']  # 示例文件名

    def download(self):
        """
        下载数据集（如果需要）。
        """
        # 如果数据集需要下载，可以在这里实现
        pass

    def process(self):
        """
        处理原始数据并保存为 PyTorch Geometric 的 Data 对象。
        """
        idx = 0
        for raw_path in self.raw_paths:
            # 加载原始数据
            data = torch.load(raw_path)  # 假设原始数据是 PyTorch 张量或字典

            # 将原始数据转换为 PyTorch Geometric 的 Data 对象
            x = data['node_features']  # 节点特征 (num_nodes, num_node_features)
            edge_index = data['edge_index']  # 边索引 (2, num_edges)
            y = data['graph_label']  # 图标签 (1,)

            graph_data = Data(x=x, edge_index=edge_index, y=y)

            # 保存处理后的数据
            torch.save(graph_data, os.path.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1

    def len(self):
        """
        返回数据集的样本数量。
        """
        return len(self.processed_file_names)

    def get(self, idx):
        """
        根据索引获取一个样本。

        Args:
            idx (int): 样本的索引。

        Returns:
            Data: PyTorch Geometric 的 Data 对象。
        """
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data

# 示例用法
if __name__ == "__main__":
    # 数据集根目录
    root = "E:/Github_Project/troch_GUI/dataset/图像处理/图分类"

    # 创建数据集实例
    dataset = GraphClassificationDataset(root)

    # 获取第一个样本
    sample = dataset[0]
    print("Node features:", sample.x)
    print("Edge index:", sample.edge_index)
    print("Graph label:", sample.y)

    # 使用 DataLoader 加载数据
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for batch in dataloader:
        print("Batch node features:", batch.x)
        print("Batch edge index:", batch.edge_index)
        print("Batch graph labels:", batch.y)