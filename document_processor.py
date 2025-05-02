from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self):
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def process_pdf(self, file_path):
        text = ""
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        chunks = self.splitter.split_text(text)
        embeddings = self.embed_model.encode(chunks)
        return [{
            "content": chunk,
            "embedding": emb.tolist(),
            "doc_metadata": {"source": file_path}
        } for chunk, emb in zip(chunks, embeddings)]
