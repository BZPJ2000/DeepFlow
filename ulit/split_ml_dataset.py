import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def split_ml_dataset(file_path, train_ratio=70, val_ratio=20, test_ratio=10, shuffle=True, random_state=42):
    """
    划分机器学习数据集（CSV 或 Excel 文件）。

    参数:
        file_path (str): 数据文件路径（CSV 或 Excel）。
        train_ratio (float): 训练集比例。
        val_ratio (float): 验证集比例。
        test_ratio (float): 测试集比例。
        shuffle (bool): 是否打乱数据。
        random_state (int): 随机种子。

    返回:
        dict: 包含划分后的数据集路径的字典。
    """
    # 确保比例总和为1
    print(train_ratio,val_ratio,test_ratio )
    assert train_ratio + val_ratio + test_ratio == 1.0, "比例总和必须为1"

    # 读取数据
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        data = pd.read_excel(file_path)
    else:
        raise ValueError("文件格式不支持，请提供 CSV 或 Excel 文件。")

    # 是否打乱数据
    if shuffle:
        data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # 划分数据集
    if val_ratio > 0:
        # 划分为 train, val, maskovrd
        train_data, test_data = train_test_split(data, test_size=test_ratio, random_state=random_state)
        train_data, val_data = train_test_split(train_data, test_size=val_ratio / (train_ratio + val_ratio),
                                                random_state=random_state)
    else:
        # 划分为 train, maskovrd
        train_data, test_data = train_test_split(data, test_size=test_ratio, random_state=random_state)
        val_data = None  # 不需要验证集

    # 创建输出目录
    output_dir = os.path.dirname(file_path)
    output_paths = {
        'train': os.path.join(output_dir, 'train.csv'),
        'maskovrd': os.path.join(output_dir, 'maskovrd.csv')
    }
    if val_ratio > 0:
        output_paths['val'] = os.path.join(output_dir, 'val.csv')

    # 保存划分后的数据集
    train_data.to_csv(output_paths['train'], index=False)
    test_data.to_csv(output_paths['maskovrd'], index=False)
    if val_ratio > 0:
        val_data.to_csv(output_paths['val'], index=False)

    return output_paths


# 示例用法
# file_path = '/path/to/your/dataset.csv'  # 替换为你的数据集路径

# # 划分为 train, val, maskovrd（打乱数据集）
# output_paths = split_ml_dataset(file_path, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, shuffle=True)
# print("训练集路径:", output_paths['train'])
# print("验证集路径:", output_paths.get('val', '无验证集'))
# print("测试集路径:", output_paths['maskovrd'])
#
# # 划分为 train, maskovrd（不打乱数据集）
# output_paths = split_ml_dataset(file_path, train_ratio=0.8, val_ratio=0.0, test_ratio=0.2, shuffle=False)
# print("训练集路径:", output_paths['train'])
# print("验证集路径:", output_paths.get('val', '无验证集'))
# print("测试集路径:", output_paths['maskovrd'])