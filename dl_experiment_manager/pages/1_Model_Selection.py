"""
模型选择页面

此页面允许用户从集成框架中浏览、选择和配置深度学习模型。
"""

import streamlit as st
import sys
import os
from pathlib import Path
from typing import Dict, Any, List

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入动态加载器
from core.dynamic_loader import ModelLoader

def main():
    """模型选择页面的主函数"""
    
    st.set_page_config(
        page_title="模型选择 - 深度学习实验管理器",
        page_icon="🤖",
        layout="wide"
    )
    
    # 标题和描述
    st.title("🤖 模型选择")
    st.markdown("""
    从集成框架中浏览和选择深度学习模型。
    配置模型参数并预览模型架构。
    """)
    
    # 主内容标签页
    tab1, tab2, tab3, tab4 = st.tabs([
        "📁 浏览模型", 
        "⚙️ 配置模型", 
        "📊 模型预览", 
        "📋 已选模型"
    ])
    
    # 初始化动态加载器
    framework_path = st.session_state.get('framework_path', '')
    loader = ModelLoader(framework_path)
    
    with tab1:
        st.header("浏览可用模型")
        
        # 框架状态
        if os.path.exists(framework_path):
            st.success(f"✅ 框架已连接: {framework_path}")
            
            # 动态发现模型类别
            st.subheader("模型类别")
            
            try:
                model_categories = loader.discover_models()
                
                if model_categories:
                    selected_category = st.selectbox(
                        "选择模型类别",
                        list(model_categories.keys()),
                        index=0
                    )
                    
                    if selected_category and model_categories[selected_category]:
                        models = model_categories[selected_category]
                        
                        st.subheader(f"可用模型 ({selected_category})")
                        
                        # 模型选择
                        selected_model = st.selectbox(
                            "选择模型",
                            models,
                            index=0
                        )
                        
                        if selected_model:
                            # 模型信息
                            with st.expander("模型信息", expanded=True):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write(f"**名称:** {selected_model}")
                                    st.write(f"**类别:** {selected_category}")
                                    st.write(f"**文件:** {selected_model}.py")
                                
                                with col2:
                                    # 尝试加载模型类以获取信息
                                    try:
                                        model_class = loader.load_model_class(
                                            f"models.{selected_category}.{selected_model}"
                                        )
                                        
                                        if model_class:
                                            st.write(f"**状态:** ✅ 可加载")
                                            
                                            # 尝试获取类签名
                                            params = loader.get_class_signature(model_class)
                                            if params:
                                                st.write("**参数:**")
                                                for param_name, default_val in params.items():
                                                    if default_val is not None:
                                                        st.write(f"  - {param_name}: {default_val}")
                                                    else:
                                                        st.write(f"  - {param_name}: (必需)")
                                        else:
                                            st.write("**状态:** ⚠️ 无法加载模型类")
                                    except Exception as e:
                                        st.warning(f"加载模型信息时出错: {e}")
                            
                            # 快速操作
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("添加到选择", type="primary", key=f"add_{selected_model}"):
                                    if 'selected_models' not in st.session_state:
                                        st.session_state.selected_models = []
                                    
                                    model_info = {
                                        'name': selected_model,
                                        'category': selected_category,
                                        'file': f"{selected_model}.py"
                                    }
                                    
                                    if model_info not in st.session_state.selected_models:
                                        st.session_state.selected_models.append(model_info)
                                        st.success(f"已添加 {selected_model} 到选择")
                                    else:
                                        st.warning(f"{selected_model} 已经在选择中")
                            
                            with col2:
                                if st.button("查看详情", key=f"view_{selected_model}"):
                                    st.session_state.model_details = selected_model
                                    st.rerun()
                
                else:
                    st.info("在框架中未找到任何模型类别。")
            
            except Exception as e:
                st.error(f"加载模型时出错: {e}")
                st.info("请检查框架路径是否正确。")
        
        else:
            st.error(f"❌ 框架未找到: {framework_path}")
            st.warning("请在设置中更新框架路径。")
    
    with tab2:
        st.header("配置模型参数")
        
        # 通用配置
        st.subheader("通用配置")
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_classes = st.number_input(
                "类别数量",
                min_value=2,
                max_value=1000,
                value=10,
                help="输出类别的数量"
            )
            
            input_channels = st.number_input(
                "输入通道数",
                min_value=1,
                max_value=10,
                value=3,
                help="输入图像的通道数（RGB=3, 灰度=1）"
            )
            
            pretrained = st.checkbox("使用预训练权重", value=True)
        
        with col2:
            learning_rate = st.number_input(
                "学习率",
                min_value=0.00001,
                max_value=1.0,
                value=0.001,
                format="%.5f",
                help="初始学习率"
            )
            
            weight_decay = st.number_input(
                "权重衰减",
                min_value=0.0,
                max_value=0.1,
                value=0.0001,
                format="%.5f",
                help="L2正则化参数"
            )
            
            dropout_rate = st.slider(
                "Dropout率",
                min_value=0.0,
                max_value=0.5,
                value=0.2,
                step=0.05,
                help="Dropout层的丢弃率"
            )
        
        # 模型特定配置
        st.subheader("模型特定参数")
        
        model_type = st.selectbox(
            "模型类型",
            ["分类", "检测", "分割", "生成", "回归"],
            index=0
        )
        
        if model_type == "分类":
            fc_layers = st.number_input("全连接层数量", min_value=1, max_value=10, value=2)
            hidden_units = st.number_input("隐藏单元数", min_value=64, max_value=4096, value=512)
        
        elif model_type == "检测":
            anchors = st.number_input("锚框数量", min_value=3, max_value=20, value=9)
            confidence_threshold = st.slider("置信度阈值", min_value=0.1, max_value=0.9, value=0.5)
        
        elif model_type == "分割":
            encoder_depth = st.slider("编码器深度", min_value=3, max_value=7, value=5)
            decoder_channels = st.multiselect(
                "解码器通道",
                options=[64, 128, 256, 512, 1024],
                default=[64, 128, 256, 512]
            )
        
        elif model_type == "生成":
            latent_dim = st.number_input("潜在空间维度", min_value=16, max_value=512, value=128)
            noise_type = st.selectbox("噪声类型", ["高斯", "均匀", "拉普拉斯"], index=0)
        
        # 保存配置
        if st.button("保存配置", type="primary"):
            config = {
                'num_classes': num_classes,
                'input_channels': input_channels,
                'pretrained': pretrained,
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'dropout_rate': dropout_rate,
                'model_type': model_type
            }
            st.session_state.model_config = config
            st.success("配置已保存！")
    
    with tab3:
        st.header("模型预览和分析")
        
        # 模型架构可视化
        st.subheader("模型架构")
        
        # 占位符用于模型可视化
        st.image(
            "https://via.placeholder.com/800x400/3b82f6/ffffff?text=模型架构可视化",
            caption="模型架构预览"
        )
        
        # 模型统计
        st.subheader("模型统计")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("参数量", "25.6M")
        with col2:
            st.metric("FLOPs", "4.2G")
        with col3:
            st.metric("内存", "98.3MB")
        with col4:
            st.metric("推理时间", "23.4ms")
        
        # 详细分析
        st.subheader("详细分析")
        
        analysis_tabs = st.tabs(["层详情", "参数分布", "计算图"])
        
        with analysis_tabs[0]:
            st.write("层详情将在此显示")
            
            import pandas as pd
            layer_data = {
                "层": ["Conv1", "Conv2", "FC1", "FC2", "输出"],
                "类型": ["卷积", "卷积", "全连接", "全连接", "全连接"],
                "输入尺寸": ["3x224x224", "64x112x112", "512x7x7", "4096", "1024"],
                "输出尺寸": ["64x112x112", "128x56x56", "10", "1024", "10"],
                "参数量": ["1.7K", "73.7K", "102.8M", "4.2M", "10.2K"]
            }
            df = pd.DataFrame(layer_data)
            st.dataframe(df, use_container_width=True)
        
        with analysis_tabs[1]:
            st.write("参数分布将在此显示")
            
            import plotly.express as px
            import numpy as np
            
            # 示例参数分布
            param_data = {
                "参数类型": ["卷积核", "偏置", "全连接权重", "批归一化"],
                "数量": [100, 50, 1000, 500, 10]
            }
            df = pd.DataFrame(param_data)
            
            fig = px.bar(df, x="参数类型", y="数量", title="参数分布")
            st.plotly_chart(fig, use_container_width=True)
        
        with analysis_tabs[2]:
            st.write("计算图将在此显示")
            st.info("计算图可视化需要额外的依赖库。")
    
    with tab4:
        st.header("已选模型")
        
        if 'selected_models' in st.session_state and st.session_state.selected_models:
            st.write(f"**已选总数:** {len(st.session_state.selected_models)} 个模型")
            
            for i, model in enumerate(st.session_state.selected_models):
                with st.expander(f"模型 {i+1}: {model['name']}", expanded=True):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**名称:** {model['name']}")
                        st.write(f"**类别:** {model['category']}")
                        st.write(f"**文件:** {model['file']}")
                    
                    with col2:
                        # 尝试获取模型参数
                        if 'model_config' in st.session_state:
                            config = st.session_state.model_config
                            st.write(f"**类别数:** {config.get('num_classes', 10)}")
                            st.write(f"**学习率:** {config.get('learning_rate', 0.001):.5f}")
                    
                    if st.button(f"移除", key=f"remove_{i}"):
                        st.session_state.selected_models.pop(i)
                        st.rerun()
            
            # 比较选项
            st.subheader("比较已选模型")
            
            if len(st.session_state.selected_models) > 1:
                comparison_metrics = st.multiselect(
                    "选择比较指标",
                    ["准确率", "精确率", "召回率", "F1分数", "推理时间", "内存使用", "参数量"],
                    default=["准确率", "推理时间", "参数量"]
                )
                
                if st.button("运行比较", type="primary"):
                    st.info("比较功能将在性能比较页面中实现。")
            else:
                st.info("选择至少2个模型以启用比较。")
        
        else:
            st.info("尚未选择任何模型。从"浏览模型"标签页中添加模型到您的选择。")
    
    # 侧边栏
    with st.sidebar:
        st.header("模型选择帮助")
        
        st.markdown("""
        ### 如何使用此页面
        
        1. **浏览模型**: 按类别探索可用模型
        2. **配置**: 根据需要调整模型参数
        3. **预览**: 查看模型架构和统计
        4. **选择**: 将模型添加到选择以进行比较
        
        ### 提示
        
        - 从少量模型开始以快速比较
        - 考虑模型复杂度与性能的平衡
        - 检查框架兼容性
        - 保存配置以供将来使用
        """)
        
        st.markdown("---")
        
        # 快速操作
        st.header("快速操作")
        
        if st.button("清除所有选择"):
            if 'selected_models' in st.session_state:
                st.session_state.selected_models = []
                st.rerun()
        
        if st.button("导出配置"):
            if 'model_config' in st.session_state:
                import json
                config = st.session_state.model_config
                st.download_button(
                    label="下载配置",
                    data=json.dumps(config, indent=4, ensure_ascii=False),
                    file_name="model_config.json",
                    mime="application/json"
                )
            else:
                st.warning("请先保存模型配置。")

if __name__ == "__main__":
    main()
