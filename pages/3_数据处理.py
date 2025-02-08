# pages/3_数据处理.py
import streamlit as st
from ulit.config import augmentations
from ulit.load_dataset_class import load_dataset_class
from ulit.split_dl_dataset import split_dl_dataset

st.title("数据处理和数据增强")

# 从 session_state 中获取用户选择的任务类型和小类
selected_category = st.session_state.get("selected_category", None)
selected_subcategory = st.session_state.get("selected_subcategory", None)

if not selected_category or not selected_subcategory:
    st.error("请先返回任务选择页面选择任务类型。")
    st.stop()

st.write(f"你选择的任务: {selected_category} -> {selected_subcategory}")
# 分隔线
st.divider()
if selected_category == "图像处理":
    st.header("数据集导入和分割")

    # 数据集路径输入
    dataset_path = st.text_input("请输入数据集的绝对路径")

    # 选择数据集是否已经分好
    data_split_option = st.radio("数据集是否已经分好?", ("否，需要分割", "是，已经分好"))

    # 根据用户选择动态显示不同的界面
    if data_split_option == "否，需要分割":
        # 输入训练集、验证集和测试集的比例
        st.subheader("数据集分割比例")
        col1, col2, col3 = st.columns(3)
        with col1:
            train_percent = st.number_input("训练集比例（%）", min_value=0, max_value=100, value=70)
        with col2:
            val_percent = st.number_input("验证集比例（%）", min_value=0, max_value=100, value=20)
        with col3:
            test_percent = st.number_input("测试集比例（%）", min_value=0, max_value=100, value=10)

        # 确保比例总和不超过100%
        if train_percent + val_percent + test_percent > 100:
            st.warning("比例总和不能超过100%")
        else:
            # 比例有效，可以进行后续操作
            pass

        # 是否打乱数据集
        shuffle_data = st.checkbox("在分割前打乱数据集")

        # 保存比例和打乱设置到 session_state
        st.session_state.train_percent = train_percent
        st.session_state.val_percent = val_percent
        st.session_state.test_percent = test_percent
        st.session_state.shuffle_data = shuffle_data

        # 分割数据集
        if st.button("分割数据集"):
            if not dataset_path:
                st.error("请输入数据集的绝对路径")
            else:
                # 将百分比转换为比例
                train_ratio = train_percent / 100
                val_ratio = val_percent / 100
                test_ratio = test_percent / 100
                # 调用分割函数
                output_dirs = split_dl_dataset(
                    data_path=dataset_path,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    test_ratio=test_ratio,
                    shuffle=shuffle_data
                )

                # 将分割后的路径保存到 session_state
                st.session_state.train_path = output_dirs['train']
                st.session_state.val_path = output_dirs.get('val', None)
                st.session_state.test_path = output_dirs['maskovrd']

                st.success("数据集分割完成！")

    elif data_split_option == "是，已经分好":
        # 已经分好的数据集路径
        st.subheader("已分好的数据集路径")
        train_path = st.text_input("训练集路径")
        val_path = st.text_input("验证集路径")
        test_path = st.text_input("测试集路径")

        # 保存路径到 session_state
        st.session_state.train_path = train_path
        st.session_state.val_path = val_path
        st.session_state.test_path = test_path

        if st.button("保存数据集路径"):
            if not train_path or not test_path:
                st.error("请输入训练集和测试集路径")
            else:
                st.success("数据集路径已保存！")
    # 分隔线
    st.divider()
    st.header("数据增强选项")
    # 定义数据增强选项及其参数
    augmentations = augmentations
    st.subheader("选择数据增强方式")
    selected_augmentations = {}
    # 将数据增强选项分成两列显示
    col1, col2 = st.columns(2)
    for i, (aug_name, props) in enumerate(augmentations.items()):
        with col1 if i % 2 == 0 else col2:
            # 显示增强选项的复选框
            if st.checkbox(aug_name):
                selected_augmentations[aug_name] = {}
                # 根据增强选项的类型动态显示对应的输入组件
                if props['type'] == 'slider':
                    value = st.slider(
                        props['label'],
                        min_value=props['min_value'],
                        max_value=props['max_value'],
                        value=props['value'],
                        step=props['step']
                    )
                    selected_augmentations[aug_name]['value'] = value
                elif props['type'] == 'checkbox':
                    value = st.checkbox(props['label'])
                    selected_augmentations[aug_name]['value'] = value
    # 分隔线
    st.divider()
    st.header("数据集类选择")
    # 加载数据集类
    dataset_class = load_dataset_class()

    # 保存到session_state
    if dataset_class:
        st.session_state.dataset_class = dataset_class
        st.success("数据集类已加载！")


    # 保存设置的按钮
    if st.button("确认并保存设置"):
        settings = {
            'dataset_path': dataset_path,
            'data_split_option': data_split_option,
            'train_percent': train_percent if data_split_option == "否，需要分割" else None,
            'val_percent': val_percent if data_split_option == "否，需要分割" else None,
            'test_percent': test_percent if data_split_option == "否，需要分割" else None,
            'shuffle_data': shuffle_data if data_split_option == "否，需要分割" else None,
            'train_path': train_path if data_split_option == "是，已经分好" else None,
            'val_path': val_path if data_split_option == "是，已经分好" else None,
            'test_path': test_path if data_split_option == "是，已经分好" else None,
            'selected_augmentations': selected_augmentations,
            'dataset_class': dataset_class  # 添加数据集类
        }
        st.session_state.settings = settings
        st.success("设置已保存，准备进入训练页面。")

    # 分隔线
    st.divider()
    # 导航到下一页
    if st.button("下一步"):
        st.switch_page("pages/4_训练设置.py")




