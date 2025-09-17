from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from models.graph_state import UploadState
from services.rag_service import get_rag_service
from utils.text_extraction import extract_text_from_file
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class UploadWorkflow:
    def __init__(self):
        self.rag_service = get_rag_service()
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the file upload processing workflow graph"""
        workflow = StateGraph(UploadState)

        # Add nodes
        workflow.add_node("validate_files", self.validate_files)
        workflow.add_node("extract_text", self.extract_text)
        workflow.add_node("create_documents", self.create_documents)
        workflow.add_node("add_to_vector_store", self.add_to_vector_store)
        workflow.add_node("handle_error", self.handle_error)

        # Define edges
        workflow.set_entry_point("validate_files")

        workflow.add_edge("validate_files", "extract_text")
        workflow.add_edge("extract_text", "create_documents")
        workflow.add_conditional_edges(
            "create_documents",
            self.check_documents_created,
            {
                "success": "add_to_vector_store",
                "error": "handle_error"
            }
        )
        workflow.add_conditional_edges(
            "add_to_vector_store",
            self.check_upload_success,
            {
                "success": END,
                "error": "handle_error"
            }
        )
        workflow.add_edge("handle_error", END)

        return workflow.compile()

    async def validate_files(self, state: UploadState) -> UploadState:
        """Validate uploaded files"""
        try:
            files = state.get("files", [])
            if not files:
                return {"error": "No files provided", "success": False}

            validated_files = []
            for file_info in files:
                filename = file_info.get("filename")
                content = file_info.get("content")

                if not filename or not content:
                    continue

                file_extension = os.path.splitext(filename)[1].lower()
                validated_files.append({
                    "filename": filename,
                    "content": content,
                    "extension": file_extension,
                    "size": len(content)
                })

            return {"files": validated_files}
        except Exception as e:
            logger.error(f"File validation failed: {e}")
            return {"error": f"File validation failed: {str(e)}", "success": False}

    async def extract_text(self, state: UploadState) -> UploadState:
        """Extract text from uploaded files"""
        try:
            files = state.get("files", [])
            if not files:
                return {"error": "No files to process", "success": False}

            extracted_texts = []
            for file_info in files:
                try:
                    text_content = extract_text_from_file(
                        file_info["content"],
                        file_info["extension"]
                    )
                    extracted_texts.append({
                        "filename": file_info["filename"],
                        "text": text_content,
                        "extension": file_info["extension"]
                    })
                except Exception as e:
                    logger.warning(f"Failed to extract text from {file_info['filename']}: {e}")
                    extracted_texts.append({
                        "filename": file_info["filename"],
                        "text": f"Failed to extract text: {str(e)}",
                        "extension": file_info["extension"],
                        "error": str(e)
                    })

            return {"extracted_texts": extracted_texts}
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return {"error": f"Text extraction failed: {str(e)}", "success": False}

    async def create_documents(self, state: UploadState) -> UploadState:
        """Create LangChain documents from extracted text"""
        try:
            extracted_texts = state.get("extracted_texts", [])
            if not extracted_texts:
                return {"error": "No extracted text to process", "success": False}

            from langchain.schema import Document
            from langchain.text_splitter import RecursiveCharacterTextSplitter

            documents = []
            for text_info in extracted_texts:
                if text_info.get("error"):
                    continue

                document = Document(
                    page_content=text_info["text"],
                    metadata={
                        "filename": text_info["filename"],
                        "upload_time": datetime.now().isoformat(),
                        "file_type": text_info["extension"],
                        "source": "upload"
                    }
                )
                documents.append(document)

            # Split documents for better chunking
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            split_docs = text_splitter.split_documents(documents)

            return {"processed_documents": split_docs}
        except Exception as e:
            logger.error(f"Document creation failed: {e}")
            return {"error": f"Document creation failed: {str(e)}", "success": False}

    async def add_to_vector_store(self, state: UploadState) -> UploadState:
        """Add documents to vector store"""
        try:
            documents = state.get("processed_documents", [])
            if not documents:
                return {"error": "No documents to add", "success": False}

            if not self.rag_service.vector_store:
                return {
                    "error": "Vector store not available",
                    "success": False,
                    "results": [{
                        "filename": doc.metadata.get("filename", "unknown"),
                        "status": "error",
                        "message": "Vector store not available"
                    } for doc in documents]
                }

            # Add to vector store
            self.rag_service.vector_store.add_documents(documents)

            results = []
            for doc in documents:
                results.append({
                    "filename": doc.metadata.get("filename", "unknown"),
                    "status": "success",
                    "message": "File processed successfully",
                    "chunk_size": len(doc.page_content)
                })

            logger.info(f"Added {len(documents)} document chunks to RAG system")
            return {
                "results": results,
                "success": True
            }
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {e}")
            return {"error": f"Failed to add documents: {str(e)}", "success": False}

    async def handle_error(self, state: UploadState) -> UploadState:
        """Handle errors in the workflow"""
        error_msg = state.get("error", "Unknown error occurred")
        logger.error(f"Upload workflow error: {error_msg}")

        # Create error results for all files
        files = state.get("files", [])
        results = []
        for file_info in files:
            results.append({
                "filename": file_info.get("filename", "unknown"),
                "status": "error",
                "message": error_msg
            })

        return {
            "error": error_msg,
            "success": False,
            "results": results
        }

    def check_documents_created(self, state: UploadState) -> str:
        """Check if documents were successfully created"""
        if state.get("error") or not state.get("processed_documents"):
            return "error"
        return "success"

    def check_upload_success(self, state: UploadState) -> str:
        """Check if upload was successful"""
        if state.get("error") or not state.get("success", False):
            return "error"
        return "success"

    async def process_upload(self, files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process file upload through the workflow"""
        initial_state = {
            "files": files,
            "success": False
        }

        try:
            final_state = await self.graph.ainvoke(initial_state)

            return {
                "processed_files": len(files),
                "successful": len([r for r in final_state.get("results", []) if r.get("status") == "success"]),
                "failed": len([r for r in final_state.get("results", []) if r.get("status") == "error"]),
                "results": final_state.get("results", []),
                "success": final_state.get("success", False),
                "error": final_state.get("error")
            }
        except Exception as e:
            logger.error(f"Upload workflow execution failed: {e}")

            # Create error results for all files
            results = []
            for file_info in files:
                results.append({
                    "filename": file_info.get("filename", "unknown"),
                    "status": "error",
                    "message": f"Workflow execution failed: {str(e)}"
                })

            return {
                "processed_files": len(files),
                "successful": 0,
                "failed": len(files),
                "results": results,
                "success": False,
                "error": f"Workflow execution failed: {str(e)}"
            }