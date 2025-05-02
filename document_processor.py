from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        # 384-dimensional model
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

    def process_pdf(self, file_path):
        try:
            # Extract text
            text = ""
            with open(file_path, "rb") as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"
            
            # Split and embed
            chunks = self.splitter.split_text(text)
            embeddings = self.embed_model.encode(chunks)
            
            return [{
                "content": chunk,
                "embedding": emb.tolist(),
                "doc_metadata": {"source": file_path}
            } for chunk, emb in zip(chunks, embeddings)]
        
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            raise