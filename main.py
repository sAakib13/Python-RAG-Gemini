from fastapi import FastAPI, UploadFile, File
from document_processor import DocumentProcessor
from db import SessionLocal, init_db, Document
from pydantic import BaseModel
from rag_system import RAGSystem
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

init_db()
processor = DocumentProcessor()
rag = RAGSystem(api_key="YOUR_GEMINI_API_KEY")

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        with open(f"/tmp/{file.filename}", "wb") as f:
            f.write(contents)
        docs = processor.process_pdf(f"/tmp/{file.filename}")
        session = SessionLocal()
        for doc in docs:
            document = Document(content=doc["content"], embedding=doc["embedding"], doc_metadata=doc["doc_metadata"])
            session.add(document)
        session.commit()
        session.close()
        response_data = {"message": f"Processed {len(docs)} chunks"}
        logger.info(f"Response: {response_data}")  # Log the response
        return response_data
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        return {"error": str(e)} # Return JSON error response



class QueryRequest(BaseModel):
    query: str

@app.post("/query/")
def query_document(request: QueryRequest):
    query = request.query
    # Implement your search and generation logic here
    answer = "This is a dummy answer for: " + query
    return {"answer": answer}