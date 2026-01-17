"""
文档页面

提供完整的使用指南、API文档和常见问题解答。
"""

import streamlit as st
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """文档页面的主函数"""
    
    st.set_page_config(
        page_title="文档 - 深度学习实验管理器",
        page_icon="📚",
        layout="wide"
    )
    
    # 标题和描述
    st.title("📚 文档")
    st.markdown("### 完整的使用指南、API文档和常见问题解答")
    
    # 主内容标签页
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📖 快速开始", 
        "🎯 功能说明", 
        "🔧 配置指南", 
        "❓ 常见问题",
        "📞 联系我们"
    ])
    
    with tab1:
        st.header("快速开始")
        
        # 安装指南
        st.subheader("安装指南")
        
        with st.expander("1. 环境要求", expanded=True):
            st.markdown("""
            ### Python环境
            - Python 3.8 或更高版本
            - pip 包管理器
            
            ### 系统要求
            - Windows/Linux/macOS
            - 至少 4GB RAM（推荐 8GB+）
            - CUDA-capable GPU（可选但推荐）
            
            ### 必需依赖
            - streamlit >= 1.41.0
            - torch >= 1.13.0
            - torchvision >= 0.14.0
            - numpy >= 1.26.0
            - pandas >= 2.0.0
            - plotly >= 5.18.0
            - matplotlib >= 3.7.0
            """)
        
        with st.expander("2. 安装步骤", expanded=True):
            st.markdown("""
            ### 克隆仓库
            ```bash
            git clone https://github.com/your-repo/dl-experiment-manager.git
            cd dl-experiment-manager
            ```
            
            ### 安装依赖
            ```bash
            pip install -r requirements.txt
            ```
            
            ### 配置框架路径
            编辑 `app.py` 中的 `framework_path` 变量：
            ```python
            framework_path = r"E:\Projects\Learning_space\2025_learn\torch-template-for-deep-learning-main"
            ```
            
            ### 运行应用
            ```bash
            streamlit run app.py
            ```
            
            应用将在 http://localhost:8501 启动
            """)
        
        # 使用流程
        st.subheader("使用流程")
        
        st.markdown("""
        ### 基本工作流程
        
        1. **模型选择** 🤖
           - 浏览可用的深度学习模型
           - 选择适合您任务的模型
           - 配置模型参数
           - 添加到选择列表
        
        2. **数据集配置** 📊
           - 选择内置数据集或上传自定义数据
           - 配置数据加载参数
           - 设置数据增强选项
           - 定义训练/验证/测试分割
        
        3. **实验设置** ⚙️
           - 配置训练超参数
           - 选择评估指标
           - 设置学习率调度
           - 创建实验组合
           - 配置检查点和日志
        
        4. **性能比较** 📈
           - 查看实验结果概览
           - 创建可视化图表
           - 执行统计检验
           - 比较模型性能
        
        5. **可视化仪表板** 📊
           - 生成学术质量图表
           - 创建出版就绪的表格
           - 导出多种格式
           - 自定义可视化样式
        
        ### 高级工作流程
        
        1. **批量实验**
           - 创建多个模型-数据集组合
           - 自动化实验执行
           - 并行运行实验
        
        2. **结果分析**
           - 深入分析实验结果
           - 识别最佳模型配置
           - 生成研究报告
        
        3. **配置管理**
           - 保存常用配置为模板
           - 加载保存的配置
           - 导出配置为脚本
           - 在实验间共享配置
        """)
        
        # 示例教程
        st.subheader("示例教程")
        
        with st.expander("图像分类示例", expanded=True):
            st.markdown("""
            ### 场景
            在CIFAR10数据集上比较ResNet和VGG模型
            
            ### 步骤
            1. 前往"模型选择"页面
            2. 选择"图像分类" → "经典网络" → "ResNet"
            3. 配置参数：类别数=10，预训练=True
            4. 添加到选择
            
            5. 前往"数据集配置"页面
            6. 选择"CIFAR10"内置数据集
            7. 配置：批次大小=32，训练/验证/测试=80/10/10
            8. 添加到选择
            
            9. 前往"实验设置"页面
            10. 配置训练参数：轮数=100，学习率=0.001
            11. 选择评估指标：准确率、F1分数
            12. 创建实验组合
            
            13. 前往"性能比较"页面
            14. 查看训练结果和性能指标
            15. 生成可视化图表
            
            16. 前往"可视化仪表板"页面
            17. 生成出版就绪的图表和表格
            18. 导出为PDF用于论文
            """)
        
        with st.expander("目标检测示例", expanded=False):
            st.markdown("""
            ### 场景
            在COCO数据集上比较YOLO和Faster R-CNN模型
            
            ### 关键步骤
            1. 选择检测模型（YOLOv3, Faster R-CNN）
            2. 配置检测参数（锚框、置信度阈值）
            3. 选择COCO数据集
            4. 配置评估指标（mAP, mAP@50, AR）
            5. 运行实验并比较结果
            6. 生成检测框可视化
            """)
    
    with tab2:
        st.header("功能说明")
        
        # 模型选择功能
        st.subheader("模型选择 🤖")
        
        st.markdown("""
        ### 功能特性
        
        - **动态模型发现**
          - 自动扫描框架中的模型目录
          - 按类别组织模型（经典网络、注意力网络、轻量级网络等）
          - 显示模型文件和类信息
        
        - **模型配置**
          - 通用参数配置（类别数、输入通道、预训练权重）
          - 模型特定参数（全连接层、隐藏单元、锚框等）
          - 学习率和优化器配置
          - Dropout和正则化设置
        
        - **模型预览**
          - 模型架构可视化
          - 参数量、FLOPs、内存使用统计
          - 层详情和参数分布
          - 计算图可视化
        
        - **模型管理**
          - 添加/移除模型
          - 模型配置保存/加载
          - 多模型比较准备
          - 配置导出（JSON、Python脚本）
        """)
        
        # 数据集配置功能
        st.subheader("数据集配置 📊")
        
        st.markdown("""
        ### 功能特性
        
        - **数据集选择**
          - 内置数据集支持（CIFAR10、ImageNet、COCO等）
          - 自定义数据集上传
          - 数据集预览和统计
          - 数据集格式验证
        
        - **数据配置**
          - 批次大小和工作者数量
          - 数据分割（训练/验证/测试）
          - 数据增强选项
          - 归一化和标准化设置
          - Pin memory和Drop last batch
        
        - **数据增强**
          - 水平/垂直翻转
          - 旋转和缩放
          - 亮度和对比度调整
          - 随机裁剪和噪声添加
        """)
        
        # 实验设置功能
        st.subheader("实验设置 ⚙️")
        
        st.markdown("""
        ### 功能特性
        
        - **训练参数**
          - 轮数、批次大小、学习率
          - 优化器选择（Adam、SGD、AdamW等）
          - 权重衰减和动量设置
          - 梯度裁剪和混合精度
        
        - **学习率调度**
          - StepLR、CosineAnnealingLR
          - ReduceLROnPlateau、OneCycleLR
          - 自定义调度参数
        
        - **评估指标**
          - 主要指标（准确率、精确率、召回率、F1分数）
          - 任务特定指标（mAP、IoU、FID等）
          - 混淆矩阵和ROC曲线
          - 评估频率设置
        
        - **实验管理**
          - 实验组合生成（全组合、手动选择、顺序）
          - 实验队列管理
          - 实验状态跟踪
          - 检查点管理
        """)
        
        # 性能比较功能
        st.subheader("性能比较 📈")
        
        st.markdown("""
        ### 功能特性
        
        - **结果概览**
          - 实验统计摘要
          - 模型排名和最佳性能
          - 快速指标卡片
        
        - **可视化类型**
          - 柱状图（性能对比）
          - 折线图（训练曲线）
          - 雷达图（多指标比较）
          - 热力图（性能矩阵）
          - 箱线图（指标分布）
        
        - **统计检验**
          - 配对t检验
          - ANOVA方差分析
          - Wilcoxon符号秩检验
          - 效应量计算
          - 显著性水平设置
        
        - **结果管理**
          - 结果过滤和搜索
          - 详细表格查看
          - 结果导出（CSV、JSON）
          - 实验对比
        """)
        
        # 可视化仪表板功能
        st.subheader("可视化仪表板 📊")
        
        st.markdown("""
        ### 功能特性
        
        - **交互式图表**
          - Plotly交互式可视化
          - 多图表类型支持
          - 自定义颜色和样式
          - 缩放和平移功能
        
        - **学术质量图表**
          - 出版就绪的图表（300+ DPI）
          - LaTeX表格生成
          - 多面板图表
          - 模型架构图
          - 结果概览图
        
        - **导出格式**
          - PNG（栅格图像）
          - PDF（矢量图形）
          - SVG（可缩放矢量）
          - EPS（封装PostScript）
        
        - **自定义设置**
          - 颜色方案选择
          - 字体配置（字体、大小）
          - 布局设置（边距、紧凑布局）
          - 图例和标签样式
        """)
    
    with tab3:
        st.header("配置指南")
        
        # 框架配置
        st.subheader("框架配置")
        
        st.markdown("""
        ### 路径配置
        
        在 `app.py` 中设置框架路径：
        ```python
        framework_path = r"E:\Projects\Learning_space\2025_learn\torch-template-for-deep-learning-main"
        ```
        
        ### 框架要求
        
        框架目录应包含：
        - `models/` - 模型实现
        - `dataloder.py` - 数据加载器
        - `losses/` - 损失函数
        - `metrics/` - 评估指标
        - `optimizer/` - 优化器
        
        ### 支持的模型类别
        
        - **经典网络**: ResNet, VGG, AlexNet, Inception
        - **注意力网络**: Transformer, Vision Transformer, Swin Transformer
        - **轻量级网络**: MobileNet, ShuffleNet, EfficientNet
        - **GAN模型**: DCGAN, StyleGAN, CycleGAN
        - **目标检测**: YOLO, Faster R-CNN, SSD
        - **语义分割**: UNet, DeepLab, FCN
        """)
        
        # 应用配置
        st.subheader("应用配置")
        
        st.markdown("""
        ### Streamlit配置
        
        在 `app.py` 中可以配置：
        - 页面标题和图标
        - 布局模式（宽/窄）
        - 初始侧边栏状态
        - 菜单项和链接
        
        ### 会话状态
        
        应用使用以下会话状态：
        - `current_page` - 当前页面
        - `experiments` - 实验列表
        - `selected_models` - 已选模型
        - `selected_datasets` - 已选数据集
        - `experiment_results` - 实验结果
        - `framework_path` - 框架路径
        - `model_config` - 模型配置
        - `dataset_config` - 数据集配置
        
        ### 配置文件
        
        配置保存在 `configs/` 目录：
        - JSON格式的实验配置
        - Python脚本导出
        - 配置模板
        """)
        
        # 性能优化
        st.subheader("性能优化")
        
        st.markdown("""
        ### 训练速度
        
        - 使用混合精度训练（AMP）
        - 增加批次大小（如果GPU内存允许）
        - 使用多个数据加载工作者
        - Pin memory以加快数据传输
        
        ### 内存使用
        
        - 减小批次大小（如果内存不足）
        - 使用梯度检查点处理大模型
        - 实验间清除缓存
        - 使用较小的图像尺寸（如果可能）
        
        ### 可重现性
        
        - 为所有实验设置随机种子
        - 使用确定性算法
        - 禁用cuDNN benchmark以保持一致性
        - 保存和加载配置
        """)
        
        # 故障排除
        st.subheader("故障排除")
        
        st.markdown("""
        ### 框架连接问题
        
        **问题**: 框架未找到
        - 检查 `framework_path` 是否正确
        - 确保框架目录存在
        - 验证Python路径包含框架目录
        
        **问题**: 模型加载失败
        - 检查模型文件是否存在于框架中
        - 验证模型类名与文件名匹配
        - 检查模型文件中的依赖
        
        ### 数据集加载问题
        
        **问题**: 数据集加载失败
        - 验证数据集路径是否正确
        - 检查数据集格式是否受支持
        - 确保数据集具有所需结构
        
        ### 可视化问题
        
        **问题**: 图表不渲染
        - 检查matplotlib后端是否配置
        - 验证所需包已安装
        - 尝试清除缓存并重新加载
        
        ### 性能问题
        
        **问题**: 训练速度慢
        - 启用混合精度训练
        - 增加批次大小
        - 使用更多数据加载工作者
        - 检查GPU利用率
        
        **问题**: 内存不足
        - 减小批次大小
        - 使用梯度检查点
        - 减小图像尺寸
        - 使用更小的模型
        """)
    
    with tab4:
        st.header("常见问题")
        
        # 一般问题
        st.subheader("一般问题")
        
        with st.expander("安装和设置", expanded=True):
            st.markdown("""
            **Q: 如何安装应用？**
            A: 运行 `pip install -r requirements.txt` 然后运行 `streamlit run app.py`
            
            **Q: 如何更改框架路径？**
            A: 编辑 `app.py` 中的 `framework_path` 变量，指向您的框架目录
            
            **Q: 应用支持哪些框架？**
            A: 目前支持PyTorch框架，需要包含标准目录结构（models/, dataloder.py等）
            
            **Q: 可以离线使用吗？**
            A: 可以，但需要预先安装所有依赖和框架
            
            **Q: 如何更新应用？**
            A: 拉取最新代码并重新运行 `streamlit run app.py`
            """)
        
        with st.expander("模型和数据集", expanded=False):
            st.markdown("""
            **Q: 如何添加自定义模型？**
            A: 将模型文件放入框架的 `models/` 目录中，应用会自动发现
            
            **Q: 如何使用自定义数据集？**
            A: 在数据集配置页面选择"自定义数据集"选项，上传您的数据
            
            **Q: 支持哪些数据集格式？**
            A: 支持图像文件夹、CSV、JSON和自定义加载器
            
            **Q: 如何配置模型参数？**
            A: 在模型选择页面的"配置模型参数"标签中设置
            
            **Q: 模型参数在哪里保存？**
            A: 保存在会话状态中，可以在实验设置页面使用
            """)
        
        with st.expander("实验和训练", expanded=False):
            st.markdown("""
            **Q: 如何运行实验？**
            A: 在实验设置页面配置参数后点击"开始实验"
            
            **Q: 如何监控训练进度？**
            A: 训练进度会实时显示在实验设置页面
            
            **Q: 如何保存检查点？**
            A: 在实验设置的"高级设置"中配置检查点目录和频率
            
            **Q: 如何恢复训练？**
            A: 从保存的检查点加载训练状态
            
            **Q: 如何比较多个模型？**
            A: 在性能比较页面选择多个模型并运行比较
            
            **Q: 如何导出结果？**
            A: 在性能比较或可视化仪表板页面使用导出功能
            """)
        
        with st.expander("可视化和导出", expanded=False):
            st.markdown("""
            **Q: 如何生成出版就绪的图表？**
            A: 在可视化仪表板选择"学术质量图表"并设置高DPI
            
            **Q: 如何导出LaTeX表格？**
            A: 在可视化仪表板的"表格"标签中使用LaTeX导出功能
            
            **Q: 支持哪些导出格式？**
            A: PNG、PDF、SVG、EPS，用于不同用途
            
            **Q: 如何自定义图表样式？**
            A: 在可视化仪表板的"设置"标签中配置颜色、字体和布局
            
            **Q: 如何生成多面板图表？**
            A: 选择"多面板图表"类型并配置子图数量
            """)
        
        # 技术问题
        st.subheader("技术问题")
        
        with st.expander("性能和内存", expanded=False):
            st.markdown("""
            **Q: 训练很慢怎么办？**
            A: 
            - 启用混合精度（AMP）
            - 增加批次大小
            - 使用更多数据加载工作者
            - 检查GPU利用率
            
            **Q: 内存不足错误？**
            A:
            - 减小批次大小
            - 使用梯度检查点
            - 减小图像尺寸
            - 使用更小的模型
            - 清理缓存
            
            **Q: GPU利用率低？**
            A:
            - 增加批次大小
            - 减少数据加载开销
            - 使用混合精度
            - 检查数据增强是否过于复杂
            """)
        
        with st.expander("框架集成", expanded=False):
            st.markdown("""
            **Q: 如何集成新框架？**
            A: 确保框架有标准目录结构，更新framework_path
            
            **Q: 模型类加载失败？**
            A: 检查模型文件语法，验证依赖，查看错误日志
            
            **Q: 如何调试模型加载？**
            A: 在模型选择页面查看模型加载状态和错误信息
            
            **Q: 支持分布式训练吗？**
            A: 当前版本支持单GPU，多GPU支持在开发中
            """)
    
    with tab5:
        st.header("联系我们")
        
        # 联系信息
        st.subheader("获取帮助")
        
        st.markdown("""
        ### 官方渠道
        
        - **GitHub Issues**: 
          https://github.com/your-repo/dl-experiment-manager/issues
          报告bug、功能请求和问题
        
        - **GitHub Discussions**: 
          https://github.com/your-repo/dl-experiment-manager/discussions
          参与社区讨论
        
        - **文档**: 
          https://github.com/your-repo/dl-experiment-manager/wiki
          查看完整文档和教程
        
        ### 社区资源
        
        - **示例和教程**
          - 查看GitHub仓库中的examples目录
          - 观看社区贡献的配置
        
        - **贡献**
          - 欢迎提交pull request
          - 帮助改进文档
          - 分享您的使用经验
        
        ### 反馈
        
        我们重视您的反馈！
        - 功能建议
        - Bug报告
        - 用户体验改进
        - 文档改进
        
        请通过GitHub Issues提交反馈。
        """)
        
        # 版本信息
        st.subheader("版本信息")
        
        st.markdown("""
        ### 当前版本
        
        - **版本**: 1.0.0
        - **发布日期**: 2025年1月
        - **Python版本**: 3.8+
        - **Streamlit版本**: 1.41.0+
        
        ### 更新日志
        
        #### v1.0.0 (当前版本)
        - 初始发布
        - 完整的中文界面
        - 动态模型加载集成
        - 五个功能页面
        - 配置管理系统
        - 性能比较和可视化
        - 完整的文档系统
        
        #### 未来计划
        
        - [ ] 实验执行引擎
        - [ ] 实时训练监控
        - [ ] 分布式训练支持
        - [ ] 云存储集成
        - [ ] 更多可视化类型
        - [ ] 自动超参数优化
        - [ ] 实验版本控制
        """)
        
        # 致谢
        st.subheader("致谢")
        
        st.markdown("""
        ### 核心技术
        
        - **Streamlit**: Web框架
        - **PyTorch**: 深度学习框架
        - **Plotly**: 交互式可视化
        - **Matplotlib**: 静态图表生成
        - **Pandas**: 数据处理
        
        ### 开源项目
        
        感谢以下开源项目的贡献：
        - PyTorch团队
        - Streamlit团队
        - Plotly团队
        - 所有贡献者
        
        ### 特别感谢
        
        - 感谢所有测试用户
        - 感谢社区反馈
        - 感谢文档贡献者
        
        本项目为深度学习研究社区而制作。
        """)
    
    # 侧边栏
    with st.sidebar:
        st.header("文档导航")
        
        st.markdown("""
        ### 快速链接
        
        - [🏠 返回首页](#)
        - [🤖 模型选择](#)
        - [📊 数据集配置](#)
        - [⚙️ 实验设置](#)
        - [📈 性能比较](#)
        - [📊 可视化仪表板](#)
        
        ### 资源
        
        - [GitHub仓库](https://github.com/your-repo/dl-experiment-manager)
        - [问题跟踪](https://github.com/your-repo/dl-experiment-manager/issues)
        - [讨论区](https://github.com/your-repo/dl-experiment-manager/discussions)
        """)
        
        st.markdown("---")
        
        # 搜索功能
        st.subheader("搜索文档")
        
        search_query = st.text_input("搜索文档", placeholder="输入关键词...")
        
        if search_query:
            st.info(f"搜索功能正在开发中。关键词：{search_query}")
        
        # 下载文档
        st.subheader("下载文档")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("下载完整文档", type="primary"):
                st.info("完整文档PDF下载功能正在开发中。")
        
        with col2:
            if st.button("下载快速开始指南"):
                st.info("快速开始指南PDF下载功能正在开发中。")

if __name__ == "__main__":
    main()
