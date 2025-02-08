import os
import json
from datetime import datetime

def save_settings_to_file(settings, folder="logs"):
    """
    将设置保存到 logs 文件夹中的 JSON 文件。
    文件名格式：settings_YYYYMMDD_N.json，其中 N 是递增的尾号。
    """
    # 确保 logs 文件夹存在
    if not os.path.exists(folder):
        os.makedirs(folder)

    # 生成文件名前缀（以年月日命名）
    timestamp = datetime.now().strftime("%Y%m%d")  # 只保留年月日
    prefix = f"settings_{timestamp}_"

    # 获取文件夹中所有以 prefix 开头的文件
    existing_files = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith(".json")]

    # 提取尾号并找到最大值
    max_suffix = 0
    for file in existing_files:
        try:
            # 提取尾号部分（去掉前缀和后缀）
            suffix = int(file[len(prefix):-5])  # -5 是为了去掉 ".json"
            if suffix > max_suffix:
                max_suffix = suffix
        except ValueError:
            continue  # 如果尾号不是数字，跳过该文件

    # 新文件的尾号是最大值加 1
    new_suffix = max_suffix + 1

    # 生成新文件名
    filename = os.path.join(folder, f"{prefix}{new_suffix:04d}.json")  # 使用4位数的编号

    # 保存设置到文件
    with open(filename, "w") as f:
        json.dump(settings, f, indent=4)

    return filename


def load_settings_from_file(filepath):
    """
    从 JSON 文件中加载设置。
    """
    with open(filepath, "r") as f:
        settings = json.load(f)
    return settings


def get_settings_files(folder="logs"):
    """
    获取 logs 文件夹中的所有设置文件，并返回文件名列表。
    """
    if not os.path.exists(folder):
        return []

    settings_files = []
    for filename in os.listdir(folder):
        if filename.endswith(".json"):
            settings_files.append(filename)
    return settings_files






















