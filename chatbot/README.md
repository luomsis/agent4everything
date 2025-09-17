# Chatbot Project

一个基于 LangGraph 和 SQLite 的聊天机器人应用。

## 功能特性

- 使用 LangGraph 构建对话流程
- SQLite 数据库持久化会话状态
- 支持 DeepSeek 语言模型
- 可配置的对话修剪和提示模板

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行应用

```bash
python chat.py
```

## 项目结构

```
chatbot/
├── chat.py          # 主聊天应用
├── llm_env.py       # 语言模型配置
├── requirements.txt # 项目依赖
└── README.md        # 项目说明
```

## 配置

- 数据库连接: 通过环境变量 `DB_URI` 配置 SQLite 数据库路径
- 默认使用本地 SQLite 数据库文件 `chatbot.db`