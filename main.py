from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sqlalchemy.orm import Session
from document_processor import DocumentProcessor
from db import SessionLocal, init_db, Document
from pydantic import BaseModel
from rag_system import RAGSystem
import logging
from dotenv import load_dotenv
from typing import List, Dict, Generator
import os
import re
import asyncio
import tempfile
import pymupdf as fitz
import pdfplumber
from datetime import datetime
import gc

# Environment and Logging Setup
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI App Initialization
app = FastAPI()
init_db()
processor = DocumentProcessor()
rag = RAGSystem(api_key=os.getenv("GEMINI_API_KEY"))


class QueryRequest(BaseModel):
    query: str


def clean_text(text: str) -> str:
    return ' '.join(text.split())


def validate_chunk(chunk: str) -> bool:
    if len(chunk) < 50:
        return False
    alnum_count = sum(1 for c in chunk if c.isalnum())
    if alnum_count / len(chunk) < 0.4:
        return False
    stopwords = {"the", "and", "of", "to", "in", "a", "is"}
    if not any(word in chunk.lower() for word in stopwords):
        return False
    return True


def semantic_chunking(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", ", "],
        keep_separator=True
    )
    return splitter.split_text(text)


def process_page_layout(page, page_num: int, file_path: str) -> List[Dict]:
    """Process a single page using layout chunking"""
    chunks = []
    text = page.extract_text(layout=True, x_tolerance=2, y_tolerance=2) or ""
    heading_pattern = r'(\n\s*(?:[A-Z][\w\s]+|\d+\.\d+)\s*\n[=\-~]+\n)'
    sections = re.split(heading_pattern, text)

    if sections and not sections[0].strip():
        sections = sections[1:]

    for i in range(0, len(sections), 2):
        if i + 1 >= len(sections):
            break
        section_content = sections[i] + sections[i + 1]
        chunk_content = section_content.strip()

        if validate_chunk(chunk_content):
            chunks.append({
                "content": chunk_content,
                "metadata": {
                    "source": file_path,
                    "page": page_num + 1,
                    "section": sections[i].strip()[:100]
                }
            })
    return chunks


def process_page_hybrid(page, page_num: int, file_path: str, chunk_size: int) -> List[Dict]:
    """Process a single page using hybrid chunking"""
    chunks = []
    text = page.extract_text(layout=True) or ""
    heading_pattern = r'(\n\s*(?:[A-Z][\w\s]+|[IVX]+\.\d+|\d+\.\d+)\s*\n)'
    sections = re.split(heading_pattern, text)

    for i in range(1, len(sections), 2):
        if i + 1 >= len(sections):
            continue
        header = sections[i].strip()
        content = sections[i + 1].strip()
        combined = f"{header}\n{content}"

        if len(combined) > chunk_size * 1.5:
            sub_chunks = semantic_chunking(combined, chunk_size)
            for j, chunk_text in enumerate(sub_chunks):
                if validate_chunk(chunk_text):
                    chunks.append({
                        "content": chunk_text,
                        "metadata": {
                            "source": file_path,
                            "page": page_num + 1,
                            "section": header,
                            "sub_section": f"Part {j + 1}"
                        }
                    })
        else:
            if validate_chunk(combined):
                chunks.append({
                    "content": combined,
                    "metadata": {
                        "source": file_path,
                        "page": page_num + 1,
                        "section": header
                    }
                })
    return chunks


def process_pdf(file_path: str, strategy: str = "hybrid", chunk_size: int = 1000, chunk_overlap: int = 200) -> Generator[Dict, None, None]:
    """Process PDF page by page to reduce memory usage"""
    try:
        # Try with pdfplumber first
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                if strategy == "layout":
                    yield from process_page_layout(page, page_num, file_path)
                elif strategy == "hybrid":
                    yield from process_page_hybrid(page, page_num, file_path, chunk_size)
                else:
                    text = page.extract_text(layout=False) or ""
                    for chunk in semantic_chunking(text, chunk_size, chunk_overlap):
                        if validate_chunk(chunk):
                            yield {
                                "content": chunk,
                                "metadata": {
                                    "source": file_path,
                                    "page": page_num + 1,
                                    "strategy": strategy
                                }
                            }
                # Clear memory after each page
                del page
                gc.collect()

    except Exception as e:
        logger.error(f"PDF processing error: {e}")
        # Fallback to PyMuPDF
        with fitz.open(file_path) as doc:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text("text")
                for chunk in semantic_chunking(text, chunk_size, chunk_overlap):
                    if validate_chunk(chunk):
                        yield {
                            "content": chunk,
                            "metadata": {
                                "source": file_path,
                                "page": page_num + 1,
                                "strategy": strategy
                            }
                        }
                # Clear memory after each page
                del page
                gc.collect()


async def process_pdf_async(file_path: str, strategy: str, chunk_size: int, chunk_overlap: int) -> List[Dict]:
    """Process PDF asynchronously with incremental saving"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: list(process_pdf(file_path, strategy, chunk_size, chunk_overlap)))


def save_chunks_to_db(docs: List[Dict]):
    """Save chunks in smaller batches to reduce memory pressure"""
    session = SessionLocal()
    batch_size = 20  # Reduced batch size
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        for doc in batch:
            embedding = processor.embed_model.encode(doc["content"]).tolist()
            session.add(Document(
                content=clean_text(doc["content"]),
                embedding=embedding,
                metadata=doc["metadata"]
            ))
        try:
            session.commit()
        except Exception as e:
            logger.error(f"Database commit error: {e}")
            session.rollback()
        finally:
            # Clear session to prevent memory bloat
            session.expunge_all()
    session.close()


@app.post("/upload/")
async def upload_pdf(
    file: UploadFile = File(...),
    strategy: str = Query("hybrid", enum=["semantic", "layout", "hybrid"]),
    chunk_size: int = Query(1000, ge=100, le=5000),
    chunk_overlap: int = Query(200, ge=10)
) -> JSONResponse:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail="Uploaded file is not a PDF")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            temp_path = tmp_file.name
            contents = await file.read()
            tmp_file.write(contents)

        # Process and save in smaller chunks
        docs = []
        chunk_generator = process_pdf(
            temp_path, strategy, chunk_size, chunk_overlap)
        for doc in chunk_generator:
            docs.append(doc)
            # Save in batches to prevent memory buildup
            if len(docs) >= 100:
                save_chunks_to_db(docs)
                docs = []
                gc.collect()

        # Save any remaining docs
        if docs:
            save_chunks_to_db(docs)

        logger.info(f"Generated {len(docs)} valid chunks from {file.filename}")
        os.remove(temp_path)
        return JSONResponse(content={"chunks_processed": len(docs)}, status_code=200)

    except Exception as e:
        logger.exception("Processing failed")
        raise HTTPException(
            status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/query/")
def query_document(request: QueryRequest):
    try:
        query_embed = processor.embed_model.encode(request.query).tolist()
        session = SessionLocal()
        try:
            results = session.query(Document).order_by(
                Document.embedding.cosine_distance(query_embed)
            ).limit(3).all()

            context = "\n".join([doc.content for doc in results])
            prompt = f"Context:\n{context}\n\nQuestion: {request.query}\nAnswer:"
            answer = rag.generate_response(prompt)
            return {"answer": answer}
        finally:
            session.close()
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
