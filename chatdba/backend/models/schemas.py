from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class QueryRequest(BaseModel):
    natural_language_query: str
    max_results: Optional[int] = 100

class QueryResponse(BaseModel):
    generated_sql: str
    execution_result: Optional[List[Dict[str, Any]]] = None
    explanation: str
    success: bool
    error_message: Optional[str] = None

class DatabaseSchema(BaseModel):
    tables: Dict[str, List[Dict[str, Any]]]

class HealthCheck(BaseModel):
    status: str
    database_connected: bool
    llm_configured: bool
    rag_initialized: bool