@echo off
chcp 65001 >nul
echo ========================================
echo 智能交易代理系统 - Windows安装脚本
echo ========================================
echo.

echo 正在检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误：未找到Python，请先安装Python 3.8+
    echo 下载地址：https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python环境检查通过
echo.

echo 正在运行安装脚本...
python setup.py

if errorlevel 1 (
    echo.
    echo ❌ 安装失败，请检查错误信息
    pause
    exit /b 1
)

echo.
echo 🎉 安装完成！
echo.
echo 下一步：
echo 1. 编辑 config/config.yaml 文件，配置你的API密钥
echo 2. 运行 python main.py 启动系统
echo.
pause



