# ChatDBA - Database AI Assistant

基于 LangChain + RAG 的数据库运维 AI 智能助手，支持自然语言查询数据库。

## 功能特性

- 🗣️ 自然语言转 SQL 查询
- 🧠 基于 RAG 的智能上下文理解
- 🔒 安全的只读查询执行
- 📊 结果解释和可视化
- 🌐 前后端分离架构

## 技术栈

### 后端 (Python)
- FastAPI - Web 框架
- LangChain - LLM 集成框架
- 多模型支持: OpenAI, DeepSeek, Azure OpenAI
- ChromaDB - 向量数据库 (RAG)
- SQLAlchemy - 数据库连接

### 前端 (JavaScript)
- 原生 HTML/CSS/JavaScript
- 响应式设计
- REST API 集成

## 安装部署

### 1. 克隆项目
```bash
git clone <repository-url>
cd chatdpa
```

### 2. 后端设置
```bash
cd backend

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\\Scripts\\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，设置您的配置
```

### 3. 环境变量配置
在 `.env` 文件中设置：

```env
# 选择模型提供商 (openai, deepseek, azure)
LLM_PROVIDER=openai

# OpenAI 配置
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo

# DeepSeek 配置
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_MODEL=deepseek-chat

# Azure OpenAI 配置
AZURE_OPENAI_API_KEY=your_azure_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_DEPLOYMENT=your-deployment-name

# 数据库配置
DATABASE_URL=postgresql://user:password@localhost:5432/dbname
DB_TYPE=postgres
CHROMA_DB_PATH=./chroma_db
HOST=0.0.0.0
PORT=8000
```

### 4. 启动后端
```bash
python main.py
```

后端服务将在 http://localhost:8000 启动

### 5. 前端访问
直接在浏览器中打开 `frontend/index.html` 文件，或使用 HTTP 服务器：

```bash
cd frontend
# 使用 Python 内置服务器
python -m http.server 3000
```

访问 http://localhost:3000

## API 接口

### 查询接口
- `POST /api/v1/query` - 处理自然语言查询
- `GET /api/v1/schema` - 获取数据库模式
- `GET /api/v1/health` - 健康检查

### 请求示例
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "natural_language_query": "显示最近7天注册的活跃用户",
    "max_results": 50
  }'
```

## 多模型配置示例

### OpenAI 配置
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-openai-key
OPENAI_MODEL=gpt-3.5-turbo
```

### DeepSeek 配置
```env
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-your-deepseek-key
DEEPSEEK_MODEL=deepseek-chat
```

### Azure OpenAI 配置
```env
LLM_PROVIDER=azure
AZURE_OPENAI_API_KEY=your-azure-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
```

### SQLite 数据库配置
```env
# 使用 SQLite 数据库
DB_TYPE=sqlite
SQLITE_DB_PATH=./data/chatdba.db

# 或者使用 PostgreSQL
# DB_TYPE=postgres
# DATABASE_URL=postgresql://user:password@localhost:5432/dbname

# 或者使用 MySQL
# DB_TYPE=mysql
# DATABASE_URL=mysql://user:password@localhost:3306/dbname
```

## SQLite 示例数据

当使用 SQLite 时，系统会自动创建以下示例表和数据：

### 用户表 (users)
- id, name, email, created_at, active
- 包含4个示例用户

### 产品表 (products)
- id, name, price, category, stock_quantity
- 包含4个示例产品

### 订单表 (orders)
- id, user_id, product_id, quantity, total_price, order_date
- 包含4个示例订单

## 安全特性

- 🔐 只允许 SELECT 查询
- 🚫 阻止所有数据修改操作
- 🔍 SQL 注入防护
- ✅ 查询安全性验证

## 支持的数据库

- PostgreSQL
- MySQL
- SQLite
- 其他 SQLAlchemy 支持的数据库

## 开发

### 项目结构
```
chatdpa/
├── backend/
│   ├── app/
│   ├── models/
│   ├── routers/
│   ├── services/
│   ├── main.py
│   ├── requirements.txt
│   └── .env
├── frontend/
│   └── index.html
└── README.md
```

### 添加新功能
1. 在 `services/` 中添加新服务
2. 在 `routers/` 中添加 API 路由
3. 在 `models/` 中定义数据模型
4. 更新前端界面

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！