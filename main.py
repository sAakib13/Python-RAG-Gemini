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
from typing import List, Dict
import os
import re
import asyncio
import tempfile
import fitz
import pdfplumber
from datetime import datetime

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
        logger.debug("Chunk too short")
        return False
    alnum_count = sum(1 for c in chunk if c.isalnum())
    if alnum_count / len(chunk) < 0.4:
        logger.debug("Low alphanumeric ratio")
        return False
    stopwords = {"the", "and", "of", "to", "in", "a", "is"}
    if not any(word in chunk.lower() for word in stopwords):
        logger.debug("Stopwords missing — likely non-content")
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


def layout_chunking(file_path: str) -> List[Dict]:
    chunks = []
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text(
                layout=True, x_tolerance=2, y_tolerance=2) or ""
            heading_pattern = r'(\n\s*(?:[A-Z][\w\s]+|\d+\.\d+)\s*\n[=\-~]+\n)'
            sections = re.split(heading_pattern, text)
            if sections and not sections[0].strip():
                sections = sections[1:]
            for i in range(0, len(sections), 2):
                if i + 1 >= len(sections):
                    break
                section_content = sections[i] + sections[i + 1]
                chunks.append({
                    "content": section_content.strip(),
                    "metadata": {
                        "source": file_path,
                        "page": page_num + 1,
                        "section": sections[i].strip()[:100]
                    }
                })
    return chunks


def hybrid_chunking(file_path: str, chunk_size: int = 1000) -> List[Dict]:
    chunks = []
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
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
                    for j, chunk in enumerate(sub_chunks):
                        chunks.append({
                            "content": chunk,
                            "metadata": {
                                "source": file_path,
                                "page": page_num + 1,
                                "section": header,
                                "sub_section": f"Part {j + 1}"
                            }
                        })
                else:
                    chunks.append({
                        "content": combined,
                        "metadata": {
                            "source": file_path,
                            "page": page_num + 1,
                            "section": header
                        }
                    })
    return chunks


def process_pdf(file_path: str, strategy: str = "hybrid", chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict]:
    try:
        with pdfplumber.open(file_path) as pdf:
            full_text = "\n".join(page.extract_text(
                layout=False) or "" for page in pdf.pages)
    except Exception as e:
        logger.error(f"Text extraction failed: {e}")
        with fitz.open(file_path) as doc:
            full_text = "\n".join(page.get_text("text") for page in doc)

    if strategy == "semantic":
        chunks = semantic_chunking(full_text, chunk_size, chunk_overlap)
        chunks = [{"content": c} for c in chunks]
    elif strategy == "layout":
        chunks = layout_chunking(file_path)
    elif strategy == "hybrid":
        chunks = hybrid_chunking(file_path, chunk_size)
    else:
        raise ValueError(f"Invalid strategy: {strategy}")

    for chunk in chunks:
        chunk.setdefault("metadata", {})
        chunk["metadata"].update({
            "source": file_path,
            "strategy": strategy,
            "timestamp": datetime.utcnow().isoformat()
        })

    return [c for c in chunks if validate_chunk(c["content"])]


async def process_pdf_async(file_path: str, strategy: str, chunk_size: int, chunk_overlap: int) -> List[Dict]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: process_pdf(file_path, strategy, chunk_size, chunk_overlap))


def save_chunks_to_db(docs: List[Dict]):
    session = SessionLocal()
    batch_size = 50
    for i in range(0, len(docs), batch_size):
        for doc in docs[i:i + batch_size]:
            embedding = processor.embed_model.encode(doc["content"]).tolist()
            session.add(Document(
                content=clean_text(doc["content"]),
                embedding=embedding,
                metadata=doc["metadata"]
            ))
        session.commit()
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

        docs = await process_pdf_async(temp_path, strategy, chunk_size, chunk_overlap)
        logger.info(f"Generated {len(docs)} valid chunks from {file.filename}")
        save_chunks_to_db(docs)

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
