
# 任务类别参数设置
task_categories = {
    "自然语言处理": ["情感分类", "机器翻译", "实体命名", "文本生成"],
    "图像处理": ["目标检测", "图像分类", "实例分割", "语义分割", "遮挡物修复", "图像生成"],
    "图神经网络": ["图分类", "节点分类", "链接预测", "图生成"],
    "强化学习": ["Q-Learning", "Deep Q-Network", "Policy_Gradient", "Actor-Critic"]
}

# 数据增强参数设置
augmentations = {
        '旋转': {
            'type': 'slider',
            'label': '旋转角度范围 (度)',
            'min_value': 0.0,
            'max_value': 180.0,
            'value': (0.0, 180.0),
            'step': 1.0
        },
        '翻转': {
            'type': 'checkbox',
            'label': '是否进行翻转'
        },
        '缩放': {
            'type': 'slider',
            'label': '缩放范围',
            'min_value': 0.8,
            'max_value': 1.2,
            'value': (0.8, 1.2),
            'step': 0.1
        },
        '平移': {
            'type': 'slider',
            'label': '水平/垂直偏移',
            'min_value': -0.5,
            'max_value': 0.5,
            'value': (-0.2, 0.2),
            'step': 0.1
        },
        '裁剪': {
            'type': 'slider',
            'label': '裁剪比例',
            'min_value': 0.1,
            'max_value': 1.0,
            'value': (0.8, 0.8),
            'step': 0.1
        },
        '亮度调整': {
            'type': 'slider',
            'label': '亮度因子',
            'min_value': 0.0,
            'max_value': 2.0,
            'value': 1.0,
            'step': 0.1
        },
        '对比度调整': {
            'type': 'slider',
            'label': '对比度因子',
            'min_value': 0.0,
            'max_value': 2.0,
            'value': 1.0,
            'step': 0.1
        },
        '噪声添加': {
            'type': 'slider',
            'label': '噪声水平',
            'min_value': 0.0,
            'max_value': 1.0,
            'value': 0.1,
            'step': 0.01
        },
        '色彩抖动': {
            'type': 'slider',
            'label': '抖动程度',
            'min_value': 0.0,
            'max_value': 1.0,
            'value': 0.2,
            'step': 0.05
        },
        '模糊': {
            'type': 'slider',
            'label': '模糊程度',
            'min_value': 0.0,
            'max_value': 5.0,
            'value': 1.0,
            'step': 0.5
        },
        '锐化': {
            'type': 'slider',
            'label': '锐化程度',
            'min_value': 0.0,
            'max_value': 2.0,
            'value': 1.0,
            'step': 0.1
        },
        '直方图均衡化': {
            'type': 'checkbox',
            'label': '是否进行直方图均衡化'
        }
    }