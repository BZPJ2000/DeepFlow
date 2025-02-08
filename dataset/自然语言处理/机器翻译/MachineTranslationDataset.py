import os
import torch
from torch.utils.data import Dataset
from transformers import MarianTokenizer

class MachineTranslationDataset(Dataset):
    def __init__(self, file_path, src_lang='en', tgt_lang='fr', max_length=128):
        """
        初始化机器翻译数据集类。

        Args:
            file_path (str): 包含源语言和目标语言句子的文件路径。
            src_lang (str): 源语言代码，默认为 'en'（英语）。
            tgt_lang (str): 目标语言代码，默认为 'fr'（法语）。
            max_length (int): 文本的最大长度，默认为128。
        """
        self.file_path = file_path
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_length = max_length

        # 加载 tokenizer
        self.tokenizer = MarianTokenizer.from_pretrained(f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}')

        # 加载数据
        self.src_texts, self.tgt_texts = self._load_data(file_path)

    def _load_data(self, file_path):
        """
        加载数据文件。

        Args:
            file_path (str): 数据文件路径。

        Returns:
            list: 源语言句子列表。
            list: 目标语言句子列表。
        """
        src_texts = []
        tgt_texts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                src, tgt = line.strip().split('\t')  # 假设源语言和目标语言用制表符分隔
                src_texts.append(src)
                tgt_texts.append(tgt)
        return src_texts, tgt_texts

    def __len__(self):
        """
        返回数据集的样本数量。
        """
        return len(self.src_texts)

    def __getitem__(self, index):
        """
        根据索引获取一个样本。

        Args:
            index (int): 样本的索引。

        Returns:
            dict: 包含源语言输入ID、目标语言输入ID和注意力掩码的字典。
        """
        src_text = self.src_texts[index]
        tgt_text = self.tgt_texts[index]

        # 使用 tokenizer 将源语言和目标语言文本转换为模型可接受的输入格式
        src_encoding = self.tokenizer(
            src_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        tgt_encoding = self.tokenizer(
            tgt_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        return {
            'input_ids': src_encoding['input_ids'].squeeze(0),  # 源语言输入ID
            'attention_mask': src_encoding['attention_mask'].squeeze(0),  # 源语言注意力掩码
            'labels': tgt_encoding['input_ids'].squeeze(0),  # 目标语言输入ID（作为标签）
        }

# 示例用法
if __name__ == "__main__":
    # 示例数据路径
    file_path = "E:/Github_Project/troch_GUI/dataset/图像处理/机器翻译/data.txt"

    # 创建数据集实例
    dataset = MachineTranslationDataset(file_path, src_lang='en', tgt_lang='fr')

    sample = dataset[0]
    print("Source Input IDs:", sample['input_ids'])
    print("Source Attention Mask:", sample['attention_mask'])
    print("Target Labels:", sample['labels'])