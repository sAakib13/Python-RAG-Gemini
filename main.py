from fastapi import FastAPI, UploadFile, File, HTTPException
from document_processor import DocumentProcessor
from db import SessionLocal, init_db, Document
from pydantic import BaseModel
from rag_system import RAGSystem
import os
import logging
from dotenv import load_dotenv
import tempfile

# ------------------------------
# Environment and Logging Setup
# ------------------------------

# Load environment variables from a .env file
load_dotenv()

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------
# FastAPI App Initialization
# ------------------------------

app = FastAPI()

# Initialize the database
init_db()

# Initialize PDF processor and RAG (Retrieval-Augmented Generation) system
processor = DocumentProcessor()
rag = RAGSystem(api_key=os.getenv("GEMINI_API_KEY"))  # Secure API key loading

# ------------------------------
# Request Model
# ------------------------------


class QueryRequest(BaseModel):
    query: str  # Input query from user


# ------------------------------
# Endpoint: Upload PDF Document
# ------------------------------

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file, extract and embed its content, and store it in the database.
    """
    try:
        # Read uploaded file content
        contents = await file.read()
        file_path = os.path.join(tempfile.gettempdir(), file.filename)

        # Write file temporarily for processing
        with open(file_path, "wb") as f:
            f.write(contents)

        # Process and split the PDF into document chunks
        docs = processor.process_pdf(file_path)

        session = SessionLocal()
        try:
            for doc in docs:
                # Sanitize document content to remove null characters
                clean_content = doc["content"].replace("\x00", " ")

                # Create Document instance for DB
                document = Document(
                    content=clean_content,
                    embedding=doc["embedding"],
                    doc_metadata=doc["doc_metadata"]
                )
                session.add(document)

            # Commit to save all documents
            session.commit()
        except Exception as db_error:
            # Rollback in case of DB errors
            session.rollback()
            raise db_error
        finally:
            session.close()
            os.remove(file_path)  # Clean up temporary file

        return {"message": f"Processed {len(docs)} chunks"}

    except Exception as e:
        logger.exception("Upload failed")
        raise HTTPException(500, detail=str(e))


# ------------------------------
# Endpoint: Query Embedded Documents
# ------------------------------

@app.post("/query/")
def query_document(request: QueryRequest):
    """
    Accept a user query, retrieve relevant documents using embeddings,
    and generate an answer using the RAG system.
    """
    try:
        # Convert user query to vector embedding
        query_embed = processor.embed_model.encode(request.query).tolist()

        session = SessionLocal()
        try:
            # Find top 3 most similar documents based on cosine similarity
            results = session.query(Document).order_by(
                Document.embedding.cosine_distance(query_embed)
            ).limit(3).all()

            # Concatenate content of retrieved documents
            context = "\n".join([doc.content for doc in results])

            # Create RAG prompt and generate answer
            prompt = f"Context:\n{context}\n\nQuestion: {request.query}\nAnswer:"
            answer = rag.generate_response(prompt)

            return {"answer": answer}
        finally:
            session.close()

    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
