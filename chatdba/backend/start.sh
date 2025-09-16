#!/bin/bash

# ChatDBA 启动脚本

echo "启动 ChatDBA 数据库 AI 助手..."

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python -m venv venv
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source venv/bin/activate

# 安装依赖
echo "安装依赖包..."
export http_proxy=http://127.0.0.1:1087;export https_proxy=http://127.0.0.1:1087;
pip install -r requirements.txt

# 检查环境变量
if [ ! -f ".env" ]; then
    echo "警告: 未找到 .env 文件，请复制 .env.example 并配置环境变量"
    echo "正在创建默认 .env 文件..."
    cp .env.example .env
    echo "请编辑 .env 文件配置您的数据库连接和 OpenAI API 密钥"
fi

# 启动服务
echo "启动 FastAPI 服务..."
python main.py