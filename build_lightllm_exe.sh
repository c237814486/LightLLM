#!/bin/bash

# LightLLM 可执行文件构建脚本
# 解决与TTS框架的lightllm命名冲突问题

set -e

echo "开始构建 LightLLM 可执行文件..."

# 检查PyInstaller是否安装
if ! command -v pyinstaller &> /dev/null; then
    echo "错误: PyInstaller 未安装"
    echo "请运行: pip install pyinstaller"
    exit 1
fi

# 检查必要的依赖
echo "检查依赖..."
python -c "import torch, transformers, flash_attn, deepspeed" 2>/dev/null || {
    echo "错误: 缺少必要的依赖包"
    echo "请确保已安装: torch, transformers, flash_attn, deepspeed"
    exit 1
}

# 清理之前的构建
echo "清理之前的构建..."
rm -rf build/ dist/

# 创建构建目录
mkdir -p build dist

# 运行PyInstaller
echo "开始PyInstaller构建..."
pyinstaller light_llm_tts.spec

# 检查构建结果
if [ -d "dist/llm_tts_server" ]; then
    echo "构建成功!"
else
    echo "构建失败!"
    exit 1
fi
