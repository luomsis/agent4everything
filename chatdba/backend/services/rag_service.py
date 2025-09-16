import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
import logging

from ..models.llm_config import LLMConfig, LLMProvider

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self.config = self._load_config()
        self.embeddings = self._initialize_embeddings()
        self.vector_store = None
        self.collection_name = "database_docs"
        self.initialize()

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
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT")
        )

    def _initialize_embeddings(self):
        """根据配置初始化Embeddings"""
        try:
            if self.config.provider == LLMProvider.OPENAI:
                if not self.config.openai_api_key:
                    raise ValueError("OpenAI API key is required for embeddings")

                return OpenAIEmbeddings(
                    openai_api_key=self.config.openai_api_key,
                    model="text-embedding-ada-002",
                    base_url=self.config.openai_base_url
                )

            elif self.config.provider == LLMProvider.DEEPSEEK:
                # DeepSeek 使用 OpenAI 兼容的 embeddings
                if not self.config.deepseek_api_key:
                    raise ValueError("DeepSeek API key is required for embeddings")

                return OpenAIEmbeddings(
                    openai_api_key=self.config.deepseek_api_key,
                    model="text-embedding-ada-002",
                    base_url=f"{self.config.deepseek_base_url}/embeddings"
                )

            elif self.config.provider == LLMProvider.AZURE_OPENAI:
                if not all([self.config.azure_api_key, self.config.azure_endpoint]):
                    raise ValueError("Azure OpenAI requires api_key and endpoint for embeddings")

                return OpenAIEmbeddings(
                    openai_api_key=self.config.azure_api_key,
                    openai_api_base=self.config.azure_endpoint,
                    openai_api_type="azure",
                    openai_api_version="2023-05-15",
                    deployment="text-embedding-ada-002"
                )

            else:
                # 默认使用 OpenAI
                return OpenAIEmbeddings(
                    openai_api_key=os.getenv("OPENAI_API_KEY"),
                    model="text-embedding-ada-002"
                )

        except Exception as e:
            logger.error(f"Embeddings initialization failed: {e}")
            raise

    def initialize(self):
        """Initialize ChromaDB vector store"""
        try:
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=os.getenv("CHROMA_DB_PATH", "./chroma_db")
            )
            logger.info("RAG system initialized successfully")
        except Exception as e:
            logger.error(f"RAG initialization failed: {e}")
            raise

    def add_database_documentation(self, schema: Dict[str, List[Dict[str, Any]]]):
        """Add database schema and documentation to vector store"""
        try:
            documents = []

            # Create documents from schema
            for table_name, columns in schema.items():
                column_descriptions = []
                for col in columns:
                    desc = f"{col['name']} ({col['type']})"
                    if col['primary_key']:
                        desc += " - PRIMARY KEY"
                    if not col['nullable']:
                        desc += " - NOT NULL"
                    column_descriptions.append(desc)

                content = f"""
Table: {table_name}
Columns:
{chr(10).join(column_descriptions)}

Usage: This table contains data about {table_name.replace('_', ' ')}.
Best practices: Use appropriate indexes on frequently queried columns.
"""
                documents.append(Document(
                    page_content=content,
                    metadata={"type": "schema", "table": table_name}
                ))

            # Add general database documentation
            general_docs = [
                Document(
                    page_content="""
Database Best Practices:
- Always use parameterized queries to prevent SQL injection
- Create indexes on frequently queried columns
- Regular backups are essential for data safety
- Monitor query performance and optimize slow queries
- Use transactions for atomic operations
""",
                    metadata={"type": "best_practices", "category": "general"}
                ),
                Document(
                    page_content="""
SQL Query Patterns:
- SELECT: Retrieve data from tables
- JOIN: Combine data from multiple tables
- WHERE: Filter results based on conditions
- GROUP BY: Aggregate data
- ORDER BY: Sort results
- LIMIT: Restrict number of rows returned
""",
                    metadata={"type": "query_patterns", "category": "general"}
                )
            ]
            documents.extend(general_docs)

            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            split_docs = text_splitter.split_documents(documents)

            # Add to vector store
            self.vector_store.add_documents(split_docs)
            logger.info(f"Added {len(split_docs)} document chunks to RAG system")

        except Exception as e:
            logger.error(f"Failed to add documentation: {e}")
            raise

    def search_relevant_info(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant database documentation"""
        try:
            results = self.vector_store.similarity_search(query, k=max_results)
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": getattr(doc, 'score', 0.0)
                }
                for doc in results
            ]
        except Exception as e:
            logger.error(f"RAG search failed: {e}")
            return []

    def get_query_context(self, natural_language_query: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Get context for query generation"""
        try:
            # Search for schema information
            schema_results = self.search_relevant_info(natural_language_query, 3)

            # Search for best practices
            best_practice_results = self.search_relevant_info(
                f"best practices {natural_language_query}", 2
            )

            return {
                "schema_info": schema_results,
                "best_practices": best_practice_results,
                "full_schema": schema
            }
        except Exception as e:
            logger.error(f"Context generation failed: {e}")
            return {"schema_info": [], "best_practices": [], "full_schema": schema}