import os
import re
import shutil
import subprocess
import streamlit as st

# 设置缓存文件夹路径
CACHE_FOLDER = "cached_projects"

def copy_folder_to_cache(source_folder):
    """
    将用户选择的文件夹复制到缓存文件夹中。
    """
    if not os.path.exists(CACHE_FOLDER):
        os.makedirs(CACHE_FOLDER)

    # 清空缓存文件夹
    for item in os.listdir(CACHE_FOLDER):
        item_path = os.path.join(CACHE_FOLDER, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
        else:
            os.remove(item_path)

    # 复制文件夹
    dest_folder = os.path.join(CACHE_FOLDER, os.path.basename(source_folder))
    shutil.copytree(source_folder, dest_folder)
    return dest_folder

def find_readme_file(folder_path):
    """
    在文件夹中查找 README.md 文件。
    """
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower() == "readme.md":
                return os.path.join(root, file)
    return None

def extract_python_commands(readme_path):
    """
    从 README.md 文件中提取 Python 指令。
    """
    with open(readme_path, "r", encoding="utf-8") as file:
        content = file.read()

    # 使用正则表达式提取 Markdown 代码块中的 Python 指令
    python_commands = re.findall(r"```python\n(.*?)\n```", content, re.DOTALL)
    return python_commands

def run_command(command):
    """
    在终端中运行指定的命令。
    """
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"

def open_terminal(folder_path):
    """
    打开终端并切换到指定文件夹。
    """
    if os.name == "nt":  # Windows
        os.system(f"start cmd /K cd {folder_path}")
    elif os.name == "posix":  # macOS/Linux
        os.system(f"gnome-terminal --working-directory={folder_path} &")
    else:
        st.error("不支持的操作系统。")

def main():
    st.title("GitHub 项目快捷运行工具")
    st.write("选择一个包含 README.md 的 GitHub 项目文件夹，提取并运行 Python 指令。")

    # 选择文件夹
    if st.button("选择文件夹"):
        source_folder = st.text_input("请输入项目文件夹路径：")
        if source_folder and os.path.isdir(source_folder):
            st.success(f"已选择文件夹: {source_folder}")

            # 复制文件夹到缓存
            cached_folder = copy_folder_to_cache(source_folder)
            st.write(f"文件夹已复制到缓存: {cached_folder}")

            # 查找 README.md 文件
            readme_path = find_readme_file(cached_folder)
            if readme_path:
                st.success(f"找到 README.md 文件: {readme_path}")

                # 提取 Python 指令
                python_commands = extract_python_commands(readme_path)
                if python_commands:
                    st.write("提取的 Python 指令：")
                    for i, cmd in enumerate(python_commands, 1):
                        st.code(cmd, language="python")
                else:
                    st.warning("未找到 Python 指令。")

                # 打开终端
                if st.button("打开终端"):
                    open_terminal(cached_folder)
                    st.success("终端已打开。")

                # 运行指令
                st.write("### 运行指令")
                user_command = st.text_input("输入要运行的指令：")
                if st.button("执行指令"):
                    if user_command:
                        st.write(f"运行指令: `{user_command}`")
                        output = run_command(user_command)
                        st.write("运行结果：")
                        st.code(output)
                    else:
                        st.warning("请输入指令。")
            else:
                st.error("未找到 README.md 文件。")
        else:
            st.warning("请输入有效的文件夹路径。")

if __name__ == "__main__":
    main()