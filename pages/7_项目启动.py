import os
import re
import shutil
import stat
import subprocess
import streamlit as st

# 缓存项目的根目录
CACHED_PROJECTS_DIR = "cached_projects"

def extract_all_code_blocks(file_path):
    """
    提取文件中所有 ``` ``` 格式的代码块
    :param file_path: 文件路径
    :return: 包含所有代码块的列表
    """
    try:
        # 打开文件并读取内容
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # 使用正则表达式提取所有 ``` ``` 格式的代码块
        code_blocks = re.findall(r'```([\s\S]*?)```', content)

        # 返回提取的代码块列表
        return code_blocks
    except FileNotFoundError:
        st.error(f"错误：文件 '{file_path}' 未找到！")
        return []
    except Exception as e:
        st.error(f"发生错误：{e}")
        return []

def filter_bash_blocks(code_blocks):
    """
    过滤出以 ```bash 开头的代码块
    :param code_blocks: 所有代码块的列表
    :return: 包含 ```bash 代码块的列表
    """
    bash_blocks = []
    for block in code_blocks:
        # 去除代码块的首尾空白字符
        block = block.strip()
        # 判断是否以 bash 开头
        if block.startswith('bash\n'):
            # 提取 bash 代码块的内容（去掉开头的 'bash\n'）
            bash_blocks.append(block[5:].strip())
    return bash_blocks

def extract_bash_commands_from_readme(readme_path):
    """
    从 README.md 文件中提取所有 Bash 指令。
    :param readme_path: README.md 文件的路径。
    :return: 包含所有 Bash 指令的列表。
    """
    # 提取所有代码块
    code_blocks = extract_all_code_blocks(readme_path)
    # 过滤出 Bash 代码块
    bash_commands = filter_bash_blocks(code_blocks)
    return bash_commands

def run_command(command, cwd=None):
    """
    在终端中运行指定的命令。
    :param command: 要运行的命令。
    :param cwd: 运行命令的工作目录。
    :return: 命令的输出结果。
    """
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True, cwd=cwd)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"

def ignore_git(dir, contents):
    """
    忽略 .git 目录。
    :param dir: 当前目录。
    :param contents: 目录内容。
    :return: 需要忽略的文件和目录列表。
    """
    return [item for item in contents if item == ".git"]

def copy_project_to_cache(folder_path):
    """
    将用户指定的项目文件夹复制到 cached_projects 目录中。
    :param folder_path: 用户指定的项目文件夹路径。
    :return: 复制后的项目路径。
    """
    if not os.path.exists(CACHED_PROJECTS_DIR):
        os.makedirs(CACHED_PROJECTS_DIR)

    # 获取项目文件夹名称
    project_name = os.path.basename(folder_path)
    cached_project_path = os.path.join(CACHED_PROJECTS_DIR, project_name)

    # 如果目标路径已存在，先删除
    if os.path.exists(cached_project_path):
        shutil.rmtree(cached_project_path, onerror=handle_remove_readonly)

    # 复制项目文件夹，忽略 .git 目录
    shutil.copytree(folder_path, cached_project_path, ignore=ignore_git)
    return cached_project_path

def handle_remove_readonly(func, path, exc_info):
    """
    处理删除只读文件时的错误。
    :param func: 删除函数。
    :param path: 文件路径。
    :param exc_info: 异常信息。
    """
    # 修改文件权限为可写
    os.chmod(path, stat.S_IWRITE)
    # 再次尝试删除
    func(path)

def main():
    st.title("GitHub 项目快捷运行工具")
    st.write("选择一个包含 README.md 的 GitHub 项目文件夹，提取并运行 Bash 指令。")

    # 输入文件夹路径
    folder_path = st.text_input("请输入项目文件夹路径：")
    if folder_path and os.path.isdir(folder_path):
        st.success(f"已选择文件夹: {folder_path}")

        # 将项目复制到 cached_projects 目录
        cached_project_path = copy_project_to_cache(folder_path)
        st.success(f"项目已复制到: {cached_project_path}")

        # 查找 README.md 文件
        readme_path = os.path.join(cached_project_path, "README.md")
        if os.path.exists(readme_path):
            st.success(f"找到 README.md 文件: {readme_path}")

            # 提取所有 Bash 指令
            bash_commands = extract_bash_commands_from_readme(readme_path)
            if bash_commands:
                st.write("提取的 Bash 指令：")
                for i, cmd in enumerate(bash_commands, 1):
                    st.write(f"**指令 {i}:**")
                    st.code(cmd, language="bash")

                # 运行指令
                st.write("### 运行指令")
                user_command = st.text_input("输入要运行的指令：", key="command_input")
                if st.button("执行指令", key="run_button"):
                    if user_command:
                        st.write(f"运行指令: `{user_command}`")
                        # 在 cached_project_path 目录下运行指令
                        output = run_command(user_command, cwd=cached_project_path)
                        st.write("运行结果：")
                        st.code(output)
                    else:
                        st.warning("请输入指令。")
            else:
                st.warning("未找到 Bash 指令。")
        else:
            st.error("未找到 README.md 文件。")
    else:
        st.warning("请输入有效的文件夹路径。")

if __name__ == "__main__":
    main()