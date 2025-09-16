import os
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from typing import Dict, List, Any
import logging
import re

from models.llm_config import LLMConfig, LLMProvider

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.config = self._load_config()
        self.llm = self._initialize_llm()

    def _load_config(self) -> LLMConfig:
        """从环境变量加载LLM配置"""
        provider = os.getenv("LLM_PROVIDER", "openai")

        return LLMConfig(
            provider=LLMProvider(provider),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),

            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
            deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),

            azure_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),

            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1000")),
            timeout=int(os.getenv("LLM_TIMEOUT", "30"))
        )

    def _initialize_llm(self):
        """根据配置初始化LLM"""
        try:
            if self.config.provider == LLMProvider.OPENAI:
                if not self.config.openai_api_key:
                    raise ValueError("OpenAI API key is required")

                return ChatOpenAI(
                    openai_api_key=self.config.openai_api_key,
                    model_name=self.config.openai_model,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout,
                    base_url=self.config.openai_base_url
                )

            elif self.config.provider == LLMProvider.DEEPSEEK:
                if not self.config.deepseek_api_key:
                    raise ValueError("DeepSeek API key is required")

                return ChatOpenAI(
                    openai_api_key=self.config.deepseek_api_key,
                    model_name=self.config.deepseek_model,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout,
                    base_url=self.config.deepseek_base_url
                )

            elif self.config.provider == LLMProvider.AZURE_OPENAI:
                if not all([self.config.azure_api_key, self.config.azure_endpoint, self.config.azure_deployment]):
                    raise ValueError("Azure OpenAI requires api_key, endpoint, and deployment")

                return AzureChatOpenAI(
                    azure_deployment=self.config.azure_deployment,
                    azure_endpoint=self.config.azure_endpoint,
                    api_key=self.config.azure_api_key,
                    api_version="2023-05-15",
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout
                )

            else:
                raise ValueError(f"Unsupported LLM provider: {self.config.provider}")

        except Exception as e:
            logger.error(f"LLM initialization failed: {e}")
            raise

    async def generate_sql_query(self, natural_language_query: str, context: Dict[str, Any]) -> str:
        """Generate SQL query from natural language using context"""
        try:
            system_prompt = self._create_system_prompt(context)

            response = await self.llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Generate a SQL query for: {natural_language_query}")
            ])

            sql_query = response.content.strip()

            # Clean up the query
            sql_query = self._clean_sql_query(sql_query)

            if self._is_query_safe(sql_query):
                return sql_query
            else:
                raise ValueError("Generated query failed safety validation")

        except Exception as e:
            logger.error(f"SQL generation failed: {e}")
            raise

    async def explain_results(self, query: str, results: List[Dict[str, Any]],
                             natural_language_query: str) -> str:
        """Explain query results in natural language"""
        try:
            system_prompt = f"""
You are a database expert explaining query results in natural language.

Original question: {natural_language_query}
Generated SQL: {query}

Your task is to explain the results clearly and concisely.
"""

            sample_results = results[:5] if len(results) > 5 else results
            human_message = f"""
Query Results (sample):
{self._format_results(sample_results)}

Total rows returned: {len(results)}

Please explain these results in natural language:
"""

            response = await self.llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_message)
            ])

            return response.content.strip()

        except Exception as e:
            logger.error(f"Result explanation failed: {e}")
            return "Unable to generate explanation for the results."

    def _create_system_prompt(self, context: Dict[str, Any]) -> str:
        """Create system prompt with schema and context"""
        schema_info = "\n".join([
            f"{info['content']}" for info in context["schema_info"]
        ])

        best_practices = "\n".join([
            f"{info['content']}" for info in context["best_practices"]
        ])

        return f"""
You are an expert SQL database assistant. Generate safe, efficient SQL queries.

DATABASE SCHEMA:
{context['full_schema']}

RELEVANT SCHEMA INFORMATION:
{schema_info}

BEST PRACTICES:
{best_practices}

RULES:
1. ONLY generate SELECT queries
2. Use parameterized queries with ? placeholders
3. Include proper JOIN conditions
4. Add appropriate WHERE clauses
5. Use LIMIT for large result sets
6. Include comments explaining the query
7. Return ONLY the SQL query
8. If request is ambiguous, return "ERROR: Request requires clarification"

Example:
-- Query to get active users
SELECT id, name, email FROM users WHERE active = ? LIMIT 10;
"""

    def _clean_sql_query(self, query: str) -> str:
        """Clean and validate SQL query"""
        # Remove markdown code blocks
        query = re.sub(r'```sql\n?|```', '', query)
        query = query.strip()

        # Ensure it starts with SELECT
        if not query.upper().startswith("SELECT"):
            # Try to find SELECT in the query
            select_match = re.search(r'SELECT', query, re.IGNORECASE)
            if select_match:
                query = query[select_match.start():]
            else:
                raise ValueError("Query must be a SELECT statement")

        return query

    def _is_query_safe(self, query: str) -> bool:
        """Validate query safety"""
        dangerous_patterns = [
            r'DROP\s+TABLE',
            r'DELETE\s+FROM',
            r'UPDATE\s+[^\s]+\s+SET',
            r'INSERT\s+INTO',
            r'TRUNCATE',
            r'EXEC\s*\(',
            r'EXECUTE\s*\(',
            r'--\s*[^\n]*$',
            r'/\*[\s\S]*?\*/'
        ]

        query_upper = query.upper()

        # Must be a SELECT query
        if not query_upper.startswith("SELECT"):
            return False

        # Check for dangerous patterns
        for pattern in dangerous_patterns:
            if re.search(pattern, query_upper, re.IGNORECASE):
                return False

        return True

    def _format_results(self, results: List[Dict[str, Any]]) -> str:
        """Format results for display"""
        if not results:
            return "No results found"

        formatted = []
        for i, row in enumerate(results, 1):
            row_str = ", ".join([f"{k}: {v}" for k, v in row.items()])
            formatted.append(f"Row {i}: {row_str}")

        return "\n".join(formatted)