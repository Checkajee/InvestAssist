#!/usr/bin/env python3
"""
智能交易代理系统 - 快速安装脚本
"""
import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 10):
        print("❌ 错误：需要Python 3.10或更高版本")
        print(f"当前版本：{sys.version}")
        return False
    print(f"✅ Python版本检查通过：{sys.version}")
    return True

def install_requirements():
    """安装依赖包"""
    print("\n📦 正在安装依赖包...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ 依赖包安装完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 依赖包安装失败：{e}")
        return False

def check_config():
    """检查配置文件"""
    config_file = Path("config/config.yaml")
    if not config_file.exists():
        print("❌ 配置文件不存在：config/config.yaml")
        return False
    
    # 读取配置文件内容
    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查API密钥是否已配置
    if "你的DeepSeek_API密钥" in content:
        print("⚠️  警告：请先在config/config.yaml中配置你的DeepSeek API密钥")
        print("   编辑config/config.yaml文件，将'你的DeepSeek_API密钥'替换为实际的API密钥")
        return False
    
    print("✅ 配置文件检查通过")
    return True

def create_directories():
    """创建必要的目录"""
    directories = [
        "data_cache",
        "data_cache/sina_news_crawl",
        "data_cache/sina_news_crawl/llm_processed",
        "data_cache/price_market_akshare", 
        "data_cache/hot_money_akshare"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ 目录结构创建完成")

def test_imports():
    """测试关键模块导入"""
    try:
        import pandas
        import aiohttp
        import loguru
        import matplotlib
        import yaml
        import tiktoken
        import httpx
        import openai
        print("✅ 关键模块导入测试通过")
        return True
    except ImportError as e:
        print(f"❌ 模块导入失败：{e}")
        return False

def main():
    """主安装流程"""
    print("=" * 60)
    print("🚀 智能交易代理系统 - 安装向导")
    print("=" * 60)
    
    # 检查Python版本
    if not check_python_version():
        return False
    
    # 创建目录
    create_directories()
    
    # 安装依赖
    if not install_requirements():
        return False
    
    # 测试导入
    if not test_imports():
        return False
    
    # 检查配置
    config_ok = check_config()
    
    print("\n" + "=" * 60)
    print("📋 安装完成！")
    print("=" * 60)
    
    if config_ok:
        print("✅ 系统已准备就绪！")
        print("\n🚀 快速开始：")
        print("   python main.py        # 运行主程序")
        print("   python chat.py        # 启动聊天模式")
        print("   python quick_test.py  # 快速测试")
    else:
        print("⚠️  请先配置API密钥，然后运行：")
        print("   python main.py")
    
    print("\n📖 详细说明请查看：README.md")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
