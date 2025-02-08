"""
env_safety_check.py
功能：检测安装包的环境安全性并推荐兼容版本
支持：交互式输入/Conda/Pip
"""

import subprocess
import sys
import re
import json
import requests
from typing import List, Tuple
from packaging import version

# 配置参数
MAX_VERSION_CHECKS = 5  # 最多检测的版本数

# ---------------------- 基础工具函数 ----------------------
def run_command(command: str) -> Tuple[str, str, int]:
    """执行终端命令并返回输出"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result.stdout, result.stderr, result.returncode
    except Exception as e:
        return "", str(e), -1

# ---------------------- 版本检测函数 ----------------------
def get_conda_versions(package: str) -> List[str]:
    """获取 Conda 可用版本列表"""
    cmd = f"conda search --json {package}"
    stdout, _, _ = run_command(cmd)
    try:
        data = json.loads(stdout)
        return sorted(
            [v["version"] for v in data.get(package, [])],
            key=version.parse,
            reverse=True
        )[:MAX_VERSION_CHECKS]
    except:
        return []

def get_pypi_versions(package: str) -> List[str]:
    """获取 PyPI 可用版本列表"""
    try:
        response = requests.get(f"https://pypi.org/pypi/{package}/json", timeout=5)
        return sorted(
            list(response.json()["releases"].keys()),
            key=version.parse,
            reverse=True
        )[:MAX_VERSION_CHECKS]
    except:
        return []

def check_conda_compatibility(package: str, version: str) -> bool:
    """检测 Conda 版本兼容性"""
    cmd = f"conda install --dry-run {package}={version}"
    stdout, stderr, _ = run_command(cmd)
    return not any(re.search(r"conflict|unsatisfiable", stdout + stderr, re.I))

def check_pip_compatibility(package: str, version: str) -> bool:
    """检测 Pip 版本兼容性"""
    cmd = f"pip install --dry-run {package}=={version}"
    _, stderr, _ = run_command(cmd)
    return "Successfully" in stderr

# ---------------------- 核心检测逻辑 ----------------------
def analyze_conda(packages: List[str]) -> Tuple[bool, List[str]]:
    """Conda 环境检测"""
    conflict_found = False
    recommendations = []

    # 检测全局冲突
    dry_run = f"conda install --dry-run {' '.join(packages)}"
    stdout, stderr, _ = run_command(dry_run)
    if re.search(r"conflict|unsatisfiable", stdout + stderr, re.I):
        conflict_found = True

        # 逐个包推荐版本
        for pkg in packages:
            safe_versions = []
            for v in get_conda_versions(pkg):
                if check_conda_compatibility(pkg, v):
                    safe_versions.append(v)
                if len(safe_versions) >= 3:
                    break
            if safe_versions:
                recommendations.append(f"{pkg}: {', '.join(safe_versions)}")

    return conflict_found, recommendations

def analyze_pip(packages: List[str]) -> Tuple[bool, List[str]]:
    """Pip 环境检测"""
    conflict_found = False
    recommendations = []

    # 确保 pipdeptree 已安装
    run_command("pip install --quiet pipdeptree")

    for pkg in packages:
        # 检测当前包冲突
        cmd = f"pip install --dry-run {pkg}"
        _, stderr, _ = run_command(cmd)
        if "Successfully" not in stderr:
            conflict_found = True

            # 获取安全版本
            safe_versions = []
            for v in get_pypi_versions(pkg):
                if check_pip_compatibility(pkg, v):
                    safe_versions.append(v)
                if len(safe_versions) >= 3:
                    break
            if safe_versions:
                recommendations.append(f"{pkg}: {', '.join(safe_versions)}")

    return conflict_found, recommendations

# ---------------------- 交互界面 ----------------------
def show_welcome():
    """显示欢迎界面"""
    print("\n" + "="*40)
    print("  Environment Safety Checker")
    print("="*40)

def get_user_input() -> Tuple[str, List[str]]:
    """获取用户输入"""
    print("\n选择包管理器:")
    print("1. Conda (推荐)")
    print("2. Pip")
    manager = input("请输入选项 (1/2): ").strip()

    if manager not in ("1", "2"):
        print("\033[91m错误: 无效选择\033[0m")
        sys.exit(1)

    print("\n输入要检测的包 (多个包用空格分隔)")
    print("示例: numpy pandas tensorflow")
    packages = input("包列表: ").strip().split()

    if not packages:
        print("\033[91m错误: 未输入包名\033[0m")
        sys.exit(1)

    return "conda" if manager == "1" else "pip", packages

# ---------------------- 主程序 ----------------------
def main():
    show_welcome()

    # 判断是否使用命令行参数
    if len(sys.argv) > 1:
        if len(sys.argv) < 3:
            print("命令行用法:")
            print("  python env_safety_check.py conda numpy pandas")
            print("  python env_safety_check.py pip requests flask")
            sys.exit(1)
        manager = sys.argv[1]
        packages = sys.argv[2:]
    else:
        manager, packages = get_user_input()

    print(f"\n🔍 正在检测 {manager.upper()} 环境安全性...")

    # 执行检测
    if manager == "conda":
        has_conflict, recommendations = analyze_conda(packages)
    else:
        has_conflict, recommendations = analyze_pip(packages)

    # 显示结果
    if has_conflict:
        print("\n\033[91m⚠️ 发现潜在依赖冲突！\033[0m")
        if recommendations:
            print("\n✅ 推荐的安全版本:")
            for rec in recommendations:
                print(f"   \033[93m{rec}\033[0m")
        else:
            print("\n\033[93m⛔ 未找到自动推荐的兼容版本\033[0m")
        print("\n建议操作:")
        print("1. 使用推荐版本安装 (例如: conda install numpy=1.21.2)")
        print("2. 创建新测试环境")
        print("3. 检查现有包版本 (conda list / pip list)")
    else:
        print("\n\033[92m✅ 环境安全检查通过！可安全安装\033[0m")

if __name__ == "__main__":
    # 确保必要的依赖
    try:
        from packaging import version
    except ImportError:
        print("正在安装必要依赖...")
        run_command("pip install packaging")

    main()