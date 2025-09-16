import os
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class DatabaseService:
    def __init__(self):
        self.engine = None
        self.db_type = os.getenv("DB_TYPE", "postgres")
        self.connect()

    def connect(self):
        try:
            if self.db_type == "sqlite":
                # SQLite 配置
                sqlite_path = os.getenv("SQLITE_DB_PATH", "./data/chatdba.db")
                # 确保目录存在
                os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
                database_url = f"sqlite:///{sqlite_path}"
            else:
                # PostgreSQL/MySQL 配置
                database_url = os.getenv("DATABASE_URL")
                if not database_url:
                    raise ValueError("DATABASE_URL environment variable is required for non-SQLite databases")

            self.engine = create_engine(database_url)
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info(f"Connected to {self.db_type} database successfully")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def get_schema(self) -> Dict[str, List[Dict[str, Any]]]:
        """Extract database schema information"""
        try:
            inspector = inspect(self.engine)
            schema = {}

            for table_name in inspector.get_table_names():
                columns = []
                for column in inspector.get_columns(table_name):
                    columns.append({
                        "name": column["name"],
                        "type": str(column["type"]),
                        "nullable": column["nullable"],
                        "primary_key": column.get("primary_key", False)
                    })
                schema[table_name] = columns

            return schema
        except SQLAlchemyError as e:
            logger.error(f"Schema extraction failed: {e}")
            raise

    def execute_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute SQL query safely"""
        try:
            with self.engine.connect() as conn:
                if parameters:
                    result = conn.execute(text(query), parameters)
                else:
                    result = conn.execute(text(query))

                # Convert result to list of dictionaries
                rows = []
                for row in result:
                    rows.append(dict(row._mapping))

                return rows
        except SQLAlchemyError as e:
            logger.error(f"Query execution failed: {e}")
            raise

    def is_safe_query(self, query: str) -> bool:
        """Check if query is safe (read-only)"""
        dangerous_keywords = ["insert", "update", "delete", "drop", "truncate", "alter", "create"]
        query_lower = query.lower()

        # Allow only SELECT queries
        if not query_lower.strip().startswith("select"):
            return False

        # Check for dangerous keywords
        for keyword in dangerous_keywords:
            if keyword in query_lower:
                return False

        return True

    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def initialize_sqlite_database(self):
        """Initialize SQLite database with sample data"""
        if self.db_type != "sqlite":
            return

        try:
            with self.engine.connect() as conn:
                # 创建示例表
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        email TEXT UNIQUE NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        active BOOLEAN DEFAULT 1
                    )
                """))

                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS products (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        price DECIMAL(10, 2) NOT NULL,
                        category TEXT NOT NULL,
                        stock_quantity INTEGER DEFAULT 0
                    )
                """))

                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS orders (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER,
                        product_id INTEGER,
                        quantity INTEGER NOT NULL,
                        total_price DECIMAL(10, 2) NOT NULL,
                        order_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (id),
                        FOREIGN KEY (product_id) REFERENCES products (id)
                    )
                """))

                # 插入示例数据
                conn.execute(text("""
                    INSERT OR IGNORE INTO users (name, email, active) VALUES
                    ('张三', 'zhangsan@example.com', 1),
                    ('李四', 'lisi@example.com', 1),
                    ('王五', 'wangwu@example.com', 0),
                    ('赵六', 'zhaoliu@example.com', 1)
                """))

                conn.execute(text("""
                    INSERT OR IGNORE INTO products (name, price, category, stock_quantity) VALUES
                    ('笔记本电脑', 5999.99, '电子产品', 50),
                    ('智能手机', 2999.99, '电子产品', 100),
                    ('办公椅', 899.99, '家具', 30),
                    ('咖啡机', 1299.99, '家电', 20)
                """))

                conn.execute(text("""
                    INSERT OR IGNORE INTO orders (user_id, product_id, quantity, total_price) VALUES
                    (1, 1, 1, 5999.99),
                    (1, 2, 2, 5999.98),
                    (2, 3, 1, 899.99),
                    (4, 4, 1, 1299.99)
                """))

                conn.commit()
                logger.info("SQLite database initialized with sample data")

        except Exception as e:
            logger.error(f"Failed to initialize SQLite database: {e}")
            raise