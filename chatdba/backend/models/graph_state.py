from typing import TypedDict, Optional, List, Dict, Any
from langchain_core.messages import BaseMessage


class QueryState(TypedDict):
    """State for query processing workflow"""
    natural_language_query: str
    max_results: int
    schema: Optional[Dict[str, Any]]
    rag_context: Optional[Dict[str, Any]]
    generated_sql: Optional[str]
    execution_result: Optional[List[Dict[str, Any]]]
    explanation: Optional[str]
    error: Optional[str]
    success: bool


class UploadState(TypedDict):
    """State for file upload processing workflow"""
    files: List[Dict[str, Any]]
    file_contents: List[bytes]
    extracted_texts: List[str]
    processed_documents: List[Dict[str, Any]]
    results: List[Dict[str, Any]]
    error: Optional[str]
    success: bool


class AgentState(TypedDict):
    """General agent state for conversation flows"""
    messages: List[BaseMessage]
    user_input: str
    context: Optional[Dict[str, Any]]
    response: Optional[str]
    error: Optional[str]
    success: bool


class WorkflowState(TypedDict):
    """Composite state for managing multiple workflows"""
    query_state: Optional[QueryState]
    upload_state: Optional[UploadState]
    agent_state: Optional[AgentState]
    workflow_type: str  # "query", "upload", "chat"
    status: str  # "pending", "processing", "completed", "failed"
    metadata: Dict[str, Any]