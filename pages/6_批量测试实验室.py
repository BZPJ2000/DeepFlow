import inspect

import streamlit as st
from ulit.load_selected_class import load_selected_class

# 标题
st.title("模型对比实验设置")

# 初始化session_state存储模型配置
if 'model_configs' not in st.session_state:
    st.session_state.model_configs = []

# 侧边栏 - 公共参数设置
with st.sidebar:
    st.header("公共实验参数")
    common_epochs = st.number_input("训练轮数 (epochs)", min_value=1, value=10)
    common_batch_size = st.number_input("批大小 (batch_size)", min_value=1, value=32)
    common_optimizer = st.selectbox("优化器", ["Adam", "SGD", "RMSprop"])

    # 添加实验描述
    experiment_name = st.text_input("实验名称", "experiment_01")
    experiment_desc = st.text_area("实验描述")

# 主界面布局分为模型选择和参数配置两部分
tab1, tab2 = st.tabs(["📚 模型选择", "⚙️ 参数配置"])

with tab1:
    # 模型选择区域
    st.header("模型选择")
    available_models = load_selected_class()  # 需要修改这个函数返回多个可用模型

    # 多选模型组件
    selected_models = st.multiselect(
        "选择要对比的模型",
        options=available_models,
        format_func=lambda x: x.__name__,
        help="可多选需要对比的模型"
    )

with tab2:
    # 参数配置区域
    st.header("模型参数配置")

    # 为每个选中的模型创建配置区域
    for i, model_class in enumerate(selected_models):
        with st.expander(f"{model_class.__name__} 参数配置", expanded=True):
            cols = st.columns([1, 3])

            with cols[0]:
                # 模型实例化基础参数
                st.subheader("基础配置")
                model_name = st.text_input(
                    "模型别名",
                    value=model_class.__name__,
                    key=f"model_name_{i}"
                )

                # 自动检测类签名中的参数
                init_params = inspect.signature(model_class.__init__).parameters

            with cols[1]:
                # 动态生成参数输入
                st.subheader("高级参数")
                params = {}
                for param_name in list(init_params.keys())[1:]:  # 跳过self参数
                    param_type = init_params[param_name].annotation
                    default_value = init_params[param_name].default

                    # 根据参数类型显示不同的输入组件
                    if param_type == int:
                        val = st.number_input(
                            param_name,
                            value=default_value if default_value != inspect.Parameter.empty else 0,
                            key=f"{model_class.__name__}_{param_name}_{i}"
                        )
                    elif param_type == float:
                        val = st.number_input(
                            param_name,
                            value=default_value if default_value != inspect.Parameter.empty else 0.0,
                            format="%f",
                            key=f"{model_class.__name__}_{param_name}_{i}"
                        )
                    else:  # 其他类型使用文本输入
                        val = st.text_input(
                            param_name,
                            value=str(default_value) if default_value != inspect.Parameter.empty else "",
                            key=f"{model_class.__name__}_{param_name}_{i}"
                        )
                    params[param_name] = val

            # 保存配置到session_state
            st.session_state.model_configs.append({
                "class": model_class,
                "name": model_name,
                "params": params
            })

# 对比实验控制按钮
if st.button("🚀 启动对比实验"):
    if len(st.session_state.model_configs) == 0:
        st.error("请至少选择一个模型进行配置")
    else:
        # 构建实验配置
        experiment_config = {
            "name": experiment_name,
            "description": experiment_desc,
            "common_params": {
                "epochs": common_epochs,
                "batch_size": common_batch_size,
                "optimizer": common_optimizer
            },
            "models": st.session_state.model_configs
        }

        # 保存到session_state供后续页面使用
        st.session_state.experiment_config = experiment_config
        st.success("实验配置已保存！")
        st.write("即将跳转到训练监控页面...")
        st.switch_page("pages/5_训练监控.py")