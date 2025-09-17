from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage

from models.graph_state import QueryState
from services.rag_service import get_rag_service
from services.llm_service import LLMService
from services.database_service import DatabaseService
import logging

logger = logging.getLogger(__name__)


class QueryWorkflow:
    def __init__(self):
        self.rag_service = get_rag_service()
        self.llm_service = LLMService()
        self.db_service = DatabaseService()
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the query processing workflow graph"""
        workflow = StateGraph(QueryState)

        # Add nodes
        workflow.add_node("get_schema", self.get_schema)
        workflow.add_node("get_rag_context", self.get_rag_context)
        workflow.add_node("generate_sql", self.generate_sql)
        workflow.add_node("execute_query", self.execute_query)
        workflow.add_node("explain_results", self.explain_results)
        workflow.add_node("handle_error", self.handle_error)

        # Define edges
        workflow.set_entry_point("get_schema")

        workflow.add_edge("get_schema", "get_rag_context")
        workflow.add_edge("get_rag_context", "generate_sql")
        workflow.add_conditional_edges(
            "generate_sql",
            self.check_sql_validity,
            {
                "valid": "execute_query",
                "invalid": "handle_error"
            }
        )
        workflow.add_conditional_edges(
            "execute_query",
            self.check_execution_success,
            {
                "success": "explain_results",
                "error": "handle_error"
            }
        )
        workflow.add_edge("explain_results", END)
        workflow.add_edge("handle_error", END)

        return workflow.compile()

    async def get_schema(self, state: QueryState) -> QueryState:
        """Get database schema"""
        try:
            schema = self.db_service.get_schema()
            return {"schema": schema}
        except Exception as e:
            logger.error(f"Failed to get schema: {e}")
            return {"error": f"Schema extraction failed: {str(e)}", "success": False}

    async def get_rag_context(self, state: QueryState) -> QueryState:
        """Get RAG context for query generation"""
        try:
            if not state.get("schema"):
                return {"error": "No schema available", "success": False}

            context = self.rag_service.get_query_context(
                state["natural_language_query"],
                state["schema"]
            )
            return {"rag_context": context}
        except Exception as e:
            logger.warning(f"RAG context failed: {e}")
            # Continue without RAG context
            return {"rag_context": {"schema_info": [], "best_practices": [], "full_schema": state.get("schema", {})}}

    async def generate_sql(self, state: QueryState) -> QueryState:
        """Generate SQL query using LLM"""
        try:
            if not state.get("rag_context"):
                return {"error": "No context available for SQL generation", "success": False}

            sql_query = await self.llm_service.generate_sql_query(
                state["natural_language_query"],
                state["rag_context"]
            )
            return {"generated_sql": sql_query}
        except Exception as e:
            logger.error(f"SQL generation failed: {e}")
            return {"error": f"SQL generation failed: {str(e)}", "success": False}

    async def execute_query(self, state: QueryState) -> QueryState:
        """Execute the generated SQL query"""
        try:
            if not state.get("generated_sql"):
                return {"error": "No SQL query to execute", "success": False}

            # Validate query safety
            if not self.db_service.is_safe_query(state["generated_sql"]):
                return {"error": "Generated query is not safe", "success": False}

            results = self.db_service.execute_query(state["generated_sql"])
            return {
                "execution_result": results[:state.get("max_results", 50)],
                "success": True
            }
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return {"error": f"Query execution failed: {str(e)}", "success": False}

    async def explain_results(self, state: QueryState) -> QueryState:
        """Explain query results using LLM"""
        try:
            if not state.get("execution_result") or not state.get("generated_sql"):
                return state

            explanation = await self.llm_service.explain_results(
                state["generated_sql"],
                state["execution_result"],
                state["natural_language_query"]
            )
            return {"explanation": explanation}
        except Exception as e:
            logger.warning(f"Result explanation failed: {e}")
            return {"explanation": "Unable to generate explanation for the results."}

    async def handle_error(self, state: QueryState) -> QueryState:
        """Handle errors in the workflow"""
        error_msg = state.get("error", "Unknown error occurred")
        logger.error(f"Query workflow error: {error_msg}")
        return {"error": error_msg, "success": False}

    def check_sql_validity(self, state: QueryState) -> str:
        """Check if generated SQL is valid"""
        if state.get("error") or not state.get("generated_sql"):
            return "invalid"

        # Basic SQL validation
        sql = state["generated_sql"].upper()
        if not sql.startswith("SELECT") or "ERROR:" in sql:
            return "invalid"

        return "valid"

    def check_execution_success(self, state: QueryState) -> str:
        """Check if query execution was successful"""
        if state.get("error") or not state.get("success", False):
            return "error"
        return "success"

    async def process_query(self, natural_language_query: str, max_results: int = 50) -> Dict[str, Any]:
        """Process a natural language query through the workflow"""
        initial_state = {
            "natural_language_query": natural_language_query,
            "max_results": max_results,
            "success": False
        }

        try:
            final_state = await self.graph.ainvoke(initial_state)

            return {
                "generated_sql": final_state.get("generated_sql"),
                "execution_result": final_state.get("execution_result"),
                "explanation": final_state.get("explanation"),
                "success": final_state.get("success", False),
                "error": final_state.get("error")
            }
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {
                "success": False,
                "error": f"Workflow execution failed: {str(e)}"
            }