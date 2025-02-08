import os
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class NERDataset(Dataset):
    def __init__(self, file_path, tokenizer_name='bert-base-uncased', max_length=128):
        """
        初始化实体命名数据集类。

        Args:
            file_path (str): 包含文本和标签的文件路径。
            tokenizer_name (str): 使用的预训练tokenizer名称，默认为'bert-base-uncased'。
            max_length (int): 文本的最大长度，默认为128。
        """
        self.file_path = file_path
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        # 加载数据
        self.sentences, self.labels = self._load_data(file_path)

        # 构建标签到ID的映射
        self.label2id = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-LOC': 3, 'I-LOC': 4, 'B-ORG': 5, 'I-ORG': 6}
        self.id2label = {v: k for k, v in self.label2id.items()}

    def _load_data(self, file_path):
        """
        加载数据文件。

        Args:
            file_path (str): 数据文件路径。

        Returns:
            list: 句子列表。
            list: 标签列表。
        """
        sentences = []
        labels = []
        with open(file_path, 'r', encoding='utf-8') as f:
            sentence = []
            label = []
            for line in f:
                line = line.strip()
                if line:
                    token, tag = line.split()
                    sentence.append(token)
                    label.append(tag)
                else:
                    if sentence:
                        sentences.append(sentence)
                        labels.append(label)
                        sentence = []
                        label = []
            if sentence:
                sentences.append(sentence)
                labels.append(label)
        return sentences, labels

    def __len__(self):
        """
        返回数据集的样本数量。
        """
        return len(self.sentences)

    def __getitem__(self, index):
        """
        根据索引获取一个样本。

        Args:
            index (int): 样本的索引。

        Returns:
            dict: 包含输入ID、注意力掩码和标签的字典。
        """
        sentence = self.sentences[index]
        label = self.labels[index]

        # 将标签转换为ID
        label_ids = [self.label2id[l] for l in label]

        # 使用tokenizer将文本转换为模型可接受的输入格式
        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        # 将标签填充到最大长度
        label_ids = label_ids[:self.max_length]
        label_ids += [self.label2id['O']] * (self.max_length - len(label_ids))

        # 将标签转换为tensor
        label_ids = torch.tensor(label_ids, dtype=torch.long)

        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # 去掉batch维度
            'attention_mask': encoding['attention_mask'].squeeze(0),  # 去掉batch维度
            'labels': label_ids
        }

# 示例用法
if __name__ == "__main__":
    # 示例数据路径
    file_path = "E:/Github_Project/troch_GUI/dataset/图像处理/实体命名/data.txt"

    # 创建数据集实例
    dataset = NERDataset(file_path)

    sample = dataset[0]
    print("Input IDs:", sample['input_ids'])
    print("Attention Mask:", sample['attention_mask'])
    print("Labels:", sample['labels'])