def save_to_txt(param_name, param_var):
    """
    将参数名和参数值保存到文件selections.txt中，如果参数名已存在，则更新其值。
    """
    filename = 'config.txt'

    # 尝试读取文件内容到字典中
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        lines = []

    param_dict = {}
    for line in lines:
        parts = line.split(':', 1)
        if len(parts) < 2:
            continue  # 跳过没有冒号的行
        name = parts[0].strip()
        value = parts[1].strip()
        param_dict[name] = value

    # 更新参数
    param_dict[param_name] = param_var

    # 将字典写回文件
    with open(filename, 'w', encoding='utf-8') as f:
        for name, value in param_dict.items():
            f.write(f"{name}: {value}\n")

# 示例调用
param_name = 'qwe'
param_value = 'qw23'
save_to_txt(param_name, param_value)