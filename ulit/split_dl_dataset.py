import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split


def split_dl_dataset(data_path, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, shuffle=True):
    # 确保比例总和为1，允许一定的误差范围
    total_ratio = train_ratio + val_ratio + test_ratio
    assert abs(total_ratio - 1.0) < 1e-6, f"比例总和必须为1，当前总和为 {total_ratio}"

    # 获取data和label的路径
    data_dir = os.path.join(data_path, 'data')
    label_dir = os.path.join(data_path, 'label')

    # 获取所有文件名
    files = os.listdir(data_dir)
    files = [f for f in files if os.path.isfile(os.path.join(data_dir, f))]  # 确保是文件

    # 是否打乱数据集
    if shuffle:
        np.random.shuffle(files)

    # 划分数据集
    if val_ratio > 0:
        # 划分为 train, val, maskovrd
        train_files, test_files = train_test_split(files, test_size=test_ratio, random_state=42)
        train_files, val_files = train_test_split(train_files, test_size=val_ratio/(train_ratio+val_ratio), random_state=42)
    else:
        # 划分为 train, maskovrd
        train_files, test_files = train_test_split(files, test_size=test_ratio, random_state=42)
        val_files = []  # 不需要验证集

    # 创建输出目录
    output_dirs = {
        'train': os.path.join(data_path, 'train'),
        'maskovrd': os.path.join(data_path, 'maskovrd')
    }
    if val_ratio > 0:
        output_dirs['val'] = os.path.join(data_path, 'val')

    for dir_name in output_dirs.values():
        os.makedirs(os.path.join(dir_name, 'data'), exist_ok=True)
        os.makedirs(os.path.join(dir_name, 'label'), exist_ok=True)

    # 复制文件到相应的目录
    def copy_files(files, output_dir):
        for f in files:
            data_file = os.path.join(data_dir, f)
            label_file = os.path.join(label_dir, f)

            # 检查文件是否存在
            if not os.path.exists(data_file):
                print(f"警告：数据文件 {data_file} 不存在，跳过")
                continue
            if not os.path.exists(label_file):
                print(f"警告：标签文件 {label_file} 不存在，跳过")
                continue

            # 复制数据文件
            shutil.copy(data_file, os.path.join(output_dir, 'data', f))
            # 复制标签文件
            shutil.copy(label_file, os.path.join(output_dir, 'label', f))

    copy_files(train_files, output_dirs['train'])
    copy_files(test_files, output_dirs['maskovrd'])
    if val_ratio > 0:
        copy_files(val_files, output_dirs['val'])

    return output_dirs
# # 示例用法
# data_path = '/path/to/your/dataset'  # 替换为你的数据集路径
#
# # 划分为 train, val, maskovrd（打乱数据集）
# output_dirs = split_dataset(data_path, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, shuffle=True)
# print("训练集路径:", output_dirs['train'])
# print("验证集路径:", output_dirs.get('val', '无验证集'))
# print("测试集路径:", output_dirs['maskovrd'])
#
# # 划分为 train, maskovrd（不打乱数据集）
# output_dirs = split_dataset(data_path, train_ratio=0.8, val_ratio=0.0, test_ratio=0.2, shuffle=False)
# print("训练集路径:", output_dirs['train'])
# print("验证集路径:", output_dirs.get('val', '无验证集'))
# print("测试集路径:", output_dirs['maskovrd'])