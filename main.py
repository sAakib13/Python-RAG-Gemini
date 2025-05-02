from fastapi import FastAPI, UploadFile, File, HTTPException
from document_processor import DocumentProcessor
from db import SessionLocal, init_db, Document
from pydantic import BaseModel
from rag_system import RAGSystem
import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize components
init_db()
processor = DocumentProcessor()
rag = RAGSystem(api_key=os.getenv("GEMINI_API_KEY"))  # Get from environment

class QueryRequest(BaseModel):
    query: str

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        contents = await file.read()
        file_path = f"/tmp/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(contents)
        
        # Process PDF
        docs = processor.process_pdf(file_path)
        
        # Store in database
        session = SessionLocal()
        try:
            for doc in docs:
                document = Document(
                    content=doc["content"],
                    embedding=doc["embedding"],
                    doc_metadata=doc["doc_metadata"]
                )
                session.add(document)
            session.commit()
        finally:
            session.close()
        
        return {"message": f"Processed {len(docs)} chunks"}
    
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/")
def query_document(request: QueryRequest):
    try:
        # Encode query
        query_embed = processor.embed_model.encode(request.query).tolist()
        
        # Find similar documents
        session = SessionLocal()
        try:
            # Cosine similarity search
            results = session.query(Document).order_by(
                Document.embedding.cosine_distance(query_embed)
            ).limit(3).all()
            
            # Combine context
            context = "\n".join([doc.content for doc in results])
            
            # Generate answer
            prompt = f"Context:\n{context}\n\nQuestion: {request.query}\nAnswer:"
            answer = rag.generate_response(prompt)
            
            return {"answer": answer}
        finally:
            session.close()
    
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))