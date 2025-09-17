from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import logging
import os
from datetime import datetime

from services.upload_workflow import UploadWorkflow
from utils.text_extraction import extract_text_from_file

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["upload"])

# Upload workflow instance
upload_workflow = None

def get_upload_workflow():
    global upload_workflow
    if upload_workflow is None:
        upload_workflow = UploadWorkflow()
    return upload_workflow

# Supported file types for knowledge base
SUPPORTED_EXTENSIONS = {
    '.txt', '.pdf', '.doc', '.docx', '.md',
    '.csv', '.json', '.xml', '.html', '.htm'
}

@router.post("/upload/knowledge")
async def upload_knowledge_file(file: UploadFile = File(...)):
    """Upload a knowledge base file for RAG system"""
    try:
        # Validate file type
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_extension}. Supported types: {', '.join(SUPPORTED_EXTENSIONS)}"
            )

        # Read file content
        content = await file.read()

        # Use workflow for processing
        workflow = get_upload_workflow()
        result = await workflow.process_upload([
            {
                "filename": file.filename,
                "content": content,
                "extension": file_extension
            }
        ])

        if result["success"]:
            return JSONResponse(
                status_code=200,
                content={
                    "message": "File uploaded successfully",
                    "filename": file.filename,
                    "file_type": file_extension,
                    "size": len(content),
                    "processed": True,
                    "result": result["results"][0] if result["results"] else {}
                }
            )
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Upload failed"))

    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")

@router.post("/upload/knowledge/batch")
async def upload_knowledge_batch(files: List[UploadFile] = File(...)):
    """Upload multiple knowledge base files"""
    try:
        file_data = []
        for file in files:
            # Validate file type
            file_extension = os.path.splitext(file.filename)[1].lower()
            if file_extension not in SUPPORTED_EXTENSIONS:
                continue

            content = await file.read()
            file_data.append({
                "filename": file.filename,
                "content": content,
                "extension": file_extension
            })

        if not file_data:
            raise HTTPException(status_code=400, detail="No valid files provided")

        # Use workflow for batch processing
        workflow = get_upload_workflow()
        result = await workflow.process_upload(file_data)

        return result

    except Exception as e:
        logger.error(f"Batch upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch upload failed: {str(e)}")

@router.get("/upload/supported-types")
async def get_supported_file_types():
    """Get list of supported file types for upload"""
    return {
        "supported_extensions": list(SUPPORTED_EXTENSIONS),
        "max_file_size": "10MB",  # You can configure this
        "description": "Knowledge base files for RAG system"
    }