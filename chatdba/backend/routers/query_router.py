from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging

from ..models.schemas import QueryRequest, QueryResponse
from ..services.database_service import DatabaseService
from ..services.rag_service import RAGService
from ..services.llm_service import LLMService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["query"])

# Initialize services
db_service = DatabaseService()
rag_service = RAGService()
llm_service = LLMService()

# Initialize SQLite database if needed
if db_service.db_type == "sqlite":
    db_service.initialize_sqlite_database()

# Load schema and initialize RAG
schema = db_service.get_schema()
rag_service.add_database_documentation(schema)

@router.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest) -> QueryResponse:
    """Process natural language query and return SQL results"""
    try:
        # Get context from RAG
        context = rag_service.get_query_context(request.natural_language_query, schema)

        # Generate SQL query
        sql_query = await llm_service.generate_sql_query(
            request.natural_language_query, context
        )

        # Execute query safely
        if not db_service.is_safe_query(sql_query):
            raise HTTPException(status_code=400, detail="Generated query is not safe")

        results = db_service.execute_query(sql_query)

        # Explain results
        explanation = await llm_service.explain_results(
            sql_query, results, request.natural_language_query
        )

        return QueryResponse(
            generated_sql=sql_query,
            execution_result=results[:request.max_results],
            explanation=explanation,
            success=True
        )

    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        return QueryResponse(
            generated_sql="",
            explanation=f"Error: {str(e)}",
            success=False,
            error_message=str(e)
        )

@router.get("/schema")
async def get_schema() -> Dict[str, Any]:
    """Get database schema"""
    return schema

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "database_connected": db_service.test_connection(),
        "llm_configured": llm_service.config.openai_api_key or llm_service.config.deepseek_api_key or llm_service.config.azure_api_key,
        "llm_provider": llm_service.config.provider.value,
        "rag_initialized": True
    }