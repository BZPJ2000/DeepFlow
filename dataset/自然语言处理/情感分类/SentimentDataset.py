import os
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer_name='bert-base-uncased', max_length=128):
        """
        初始化情感分类数据集类。

        Args:
            texts (list): 包含文本数据的列表。
            labels (list): 包含情感标签的列表。
            tokenizer_name (str): 使用的预训练tokenizer名称，默认为'bert-base-uncased'。
            max_length (int): 文本的最大长度，默认为128。
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        """
        返回数据集的样本数量。
        """
        return len(self.texts)

    def __getitem__(self, index):
        """
        根据索引获取一个样本。

        Args:
            index (int): 样本的索引。

        Returns:
            dict: 包含输入ID、注意力掩码和标签的字典。
        """
        text = self.texts[index]
        label = self.labels[index]

        # 使用tokenizer将文本转换为模型可接受的输入格式
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        # 将标签转换为tensor
        label = torch.tensor(label, dtype=torch.long)

        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # 去掉batch维度
            'attention_mask': encoding['attention_mask'].squeeze(0),  # 去掉batch维度
            'label': label
        }

# 示例用法
if __name__ == "__main__":
    # 示例数据
    texts = ["I love this movie!", "This film is terrible.", "What a great experience!"]
    labels = [1, 0, 1]  # 1表示正面情感，0表示负面情感

    # 创建数据集实例
    dataset = SentimentDataset(texts, labels)

    sample = dataset[0]
    print("Input IDs:", sample['input_ids'])
    print("Attention Mask:", sample['attention_mask'])
    print("Label:", sample['label'])