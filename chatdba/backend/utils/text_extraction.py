from typing import Optional
import tempfile
import os
from fastapi import HTTPException

# Try to import text extraction libraries
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import docx
except ImportError:
    docx = None

def extract_text_from_pdf(content: bytes) -> str:
    """Extract text from PDF file"""
    if PyPDF2 is None:
        raise HTTPException(status_code=400, detail="PDF processing requires PyPDF2 package")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(content)
            tmp_file.flush()

        with open(tmp_file.name, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"

        os.unlink(tmp_file.name)
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF extraction failed: {str(e)}")

def extract_text_from_docx(content: bytes) -> str:
    """Extract text from DOCX file"""
    if docx is None:
        raise HTTPException(status_code=400, detail="DOCX processing requires python-docx package")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
            tmp_file.write(content)
            tmp_file.flush()

        doc = docx.Document(tmp_file.name)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])

        os.unlink(tmp_file.name)
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"DOCX extraction failed: {str(e)}")

def extract_text_from_file(content: bytes, file_extension: str) -> str:
    """Extract text from various file types"""
    if file_extension == '.pdf':
        return extract_text_from_pdf(content)
    elif file_extension == '.docx':
        return extract_text_from_docx(content)
    elif file_extension in {'.txt', '.md', '.csv', '.json', '.xml', '.html', '.htm'}:
        try:
            return content.decode('utf-8')
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="File encoding not supported")
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type for text extraction: {file_extension}")