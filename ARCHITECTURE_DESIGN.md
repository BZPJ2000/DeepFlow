# DeepFlow 新架构设计方案

## 第二部分：项目架构设计

---

## 2.1 设计原则

### 核心原则
1. **模块化设计** - 高内聚、低耦合
2. **可扩展性** - 易于添加新模型、新任务
3. **自动发现** - 动态加载组件，无需手动注册
4. **配置驱动** - 通过配置文件控制行为
5. **类型安全** - 使用类型注解提升代码质量

### 设计目标
- ✅ 自动感知脚本中的函数和类
- ✅ 快速选择模型与训练流程
- ✅ 支持多领域深度学习任务
- ✅ 提供友好的 Web 界面
- ✅ 实验结果可追溯、可复现

---

## 2.2 新目录结构

```
DeepFlow/
├── README.md                       # 项目主文档
├── requirements.txt                # 依赖清单
├── setup.py                        # 安装配置
├── .gitignore                      # Git 忽略规则
├── pyproject.toml                  # 项目配置 (PEP 518)
│
├── app.py                          # 主应用入口 (Streamlit)
│
├── deepflow/                       # 核心框架包
│   ├── __init__.py
│   ├── core/                       # 核心功能模块
│   │   ├── __init__.py
│   │   ├── discovery.py            # 自动发现引擎 ⭐
│   │   ├── registry.py             # 组件注册中心 ⭐
│   │   ├── loader.py               # 动态加载器 ⭐
│   │   ├── config.py               # 配置管理
│   │   └── validator.py            # 组件验证器
│   │
│   ├── components/                 # 组件基类
│   │   ├── __init__.py
│   │   ├── base_model.py           # 模型基类
│   │   ├── base_loss.py            # 损失函数基类
│   │   ├── base_metric.py          # 评估指标基类
│   │   ├── base_optimizer.py       # 优化器基类
│   │   └── base_dataset.py         # 数据集基类
│   │
│   ├── utils/                      # 工具函数
│   │   ├── __init__.py
│   │   ├── file_utils.py           # 文件操作
│   │   ├── model_utils.py          # 模型工具
│   │   ├── data_utils.py           # 数据处理
│   │   ├── visualization.py        # 可视化
│   │   └── logger.py               # 日志管理
│   │
│   ├── training/                   # 训练模块
│   │   ├── __init__.py
│   │   ├── trainer.py              # 训练器
│   │   ├── evaluator.py            # 评估器
│   │   └── callbacks.py            # 回调函数
│   │
│   └── api/                        # API 接口
│       ├── __init__.py
│       └── experiment.py           # 实验管理 API
│
├── ui/                             # 用户界面
│   ├── __init__.py
│   ├── app.py                      # Streamlit 主应用
│   ├── pages/                      # 页面模块
│   │   ├── __init__.py
│   │   ├── 1_task_selection.py     # 任务选择
│   │   ├── 2_model_selection.py    # 模型选择
│   │   ├── 3_data_config.py        # 数据配置
│   │   ├── 4_training_config.py    # 训练配置
│   │   ├── 5_results_view.py       # 结果查看
│   │   └── 6_batch_experiments.py  # 批量实验
│   │
│   ├── components/                 # UI 组件
│   │   ├── __init__.py
│   │   ├── model_card.py           # 模型卡片
│   │   ├── config_panel.py         # 配置面板
│   │   └── result_chart.py         # 结果图表
│   │
│   └── static/                     # 静态资源
│       ├── css/
│       └── images/
│
├── library/                        # 组件库 (原 models/, losses/ 等)
│   ├── models/                     # 模型库
│   │   ├── __init__.py
│   │   ├── nlp/                    # 自然语言处理
│   │   │   ├── __init__.py
│   │   │   ├── sentiment/          # 情感分类
│   │   │   ├── translation/        # 机器翻译
│   │   │   ├── ner/                # 命名实体识别
│   │   │   └── generation/         # 文本生成
│   │   │
│   │   ├── vision/                 # 计算机视觉
│   │   │   ├── __init__.py
│   │   │   ├── detection/          # 目标检测
│   │   │   ├── classification/     # 图像分类
│   │   │   ├── segmentation/       # 图像分割
│   │   │   └── generation/         # 图像生成
│   │   │
│   │   ├── graph/                  # 图神经网络
│   │   │   ├── __init__.py
│   │   │   ├── classification/     # 图分类
│   │   │   ├── node_classification/# 节点分类
│   │   │   └── link_prediction/    # 链接预测
│   │   │
│   │   └── rl/                     # 强化学习
│   │       ├── __init__.py
│   │       ├── q_learning/
│   │       ├── dqn/
│   │       └── policy_gradient/
│   │
│   ├── losses/                     # 损失函数库
│   │   ├── __init__.py
│   │   ├── nlp/
│   │   ├── vision/
│   │   ├── graph/
│   │   └── rl/
│   │
│   ├── metrics/                    # 评估指标库
│   │   ├── __init__.py
│   │   ├── nlp/
│   │   ├── vision/
│   │   ├── graph/
│   │   └── rl/
│   │
│   └── optimizers/                 # 优化器库
│       ├── __init__.py
│       ├── nlp/
│       ├── vision/
│       ├── graph/
│       └── rl/
│
├── configs/                        # 配置文件
│   ├── default.yaml                # 默认配置
│   ├── tasks.yaml                  # 任务定义
│   ├── augmentation.yaml           # 数据增强配置
│   └── experiments/                # 实验配置
│       └── example.yaml
│
├── data/                           # 数据目录
│   ├── raw/                        # 原始数据
│   ├── processed/                  # 处理后数据
│   └── samples/                    # 样例数据
│
├── outputs/                        # 输出目录
│   ├── logs/                       # 日志
│   ├── checkpoints/                # 模型检查点
│   ├── results/                    # 实验结果
│   └── visualizations/             # 可视化结果
│
├── tests/                          # 测试代码
│   ├── __init__.py
│   ├── test_discovery.py
│   ├── test_loader.py
│   └── test_trainer.py
│
├── docs/                           # 文档
│   ├── user_guide.md               # 用户指南
│   ├── developer_guide.md          # 开发者指南
│   ├── api_reference.md            # API 参考
│   └── architecture.md             # 架构文档
│
└── scripts/                        # 脚本工具
    ├── setup_env.py                # 环境设置
    ├── validate_components.py      # 组件验证
    └── export_config.py            # 配置导出
```

---

## 2.3 核心模块设计

### 2.3.1 自动发现引擎 (discovery.py)

**功能:** 自动扫描并识别组件库中的所有可用组件

**核心类:**
```python
class ComponentDiscovery:
    """组件自动发现引擎"""

    def discover_all(self, base_path: str) -> Dict[str, List[ComponentInfo]]
    def discover_models(self, path: str) -> List[ModelInfo]
    def discover_losses(self, path: str) -> List[LossInfo]
    def discover_metrics(self, path: str) -> List[MetricInfo]
    def discover_optimizers(self, path: str) -> List[OptimizerInfo]
```

**发现规则:**
1. 扫描指定目录下的所有 Python 文件
2. 解析 AST 提取类定义和函数定义
3. 检查是否继承自基类或符合接口规范
4. 提取元数据 (名称、描述、参数、依赖)
5. 注册到组件注册中心

---

### 2.3.2 组件注册中心 (registry.py)

**功能:** 统一管理所有已发现的组件

**核心类:**
```python
class ComponentRegistry:
    """组件注册中心 (单例模式)"""

    def register(self, component_type: str, name: str, info: ComponentInfo)
    def get(self, component_type: str, name: str) -> ComponentInfo
    def list(self, component_type: str, filters: Dict = None) -> List[ComponentInfo]
    def search(self, query: str) -> List[ComponentInfo]
```

**支持的组件类型:**
- `models` - 模型
- `losses` - 损失函数
- `metrics` - 评估指标
- `optimizers` - 优化器
- `datasets` - 数据集

---

### 2.3.3 动态加载器 (loader.py)

**功能:** 根据配置动态加载和实例化组件

**核心类:**
```python
class ComponentLoader:
    """组件动态加载器"""

    def load_model(self, name: str, **kwargs) -> nn.Module
    def load_loss(self, name: str, **kwargs) -> nn.Module
    def load_metric(self, name: str, **kwargs) -> Callable
    def load_optimizer(self, name: str, model_params, **kwargs) -> Optimizer
```

**加载流程:**
1. 从注册中心获取组件信息
2. 动态导入模块
3. 验证参数
4. 实例化组件
5. 返回实例

---

## 2.4 文件命名规范

### Python 文件
- 模块文件: `snake_case.py` (如 `model_utils.py`)
- 类名: `PascalCase` (如 `ComponentLoader`)
- 函数名: `snake_case` (如 `load_model`)
- 常量: `UPPER_CASE` (如 `MAX_BATCH_SIZE`)

### 配置文件
- YAML 配置: `lowercase.yaml` (如 `default.yaml`)
- JSON 配置: `lowercase.json` (如 `experiment.json`)

### 目录命名
- 包目录: `lowercase` (如 `deepflow`, `library`)
- 多词目录: `snake_case` (如 `node_classification`)

---

## 2.5 模块职责划分

| 模块 | 职责 | 依赖 |
|------|------|------|
| `deepflow.core` | 核心功能 (发现、注册、加载) | 无 |
| `deepflow.components` | 组件基类定义 | `core` |
| `deepflow.utils` | 工具函数 | `core` |
| `deepflow.training` | 训练评估逻辑 | `core`, `components` |
| `deepflow.api` | API 接口 | 所有模块 |
| `ui` | 用户界面 | `deepflow.api` |
| `library` | 组件实现 | `deepflow.components` |

---

下一部分将详细说明实现步骤和技术细节。
