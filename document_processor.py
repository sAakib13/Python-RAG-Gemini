from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Processes PDF documents by extracting text, chunking it,
    and generating embeddings for use in downstream retrieval tasks.
    """

    def __init__(self):
        """
        Initializes the document processor with:
        - A SentenceTransformer model for embeddings.
        - A text splitter for breaking large documents into smaller chunks.
        """
        # SentenceTransformer model: 384-dimensional embeddings
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Splitter configuration for chunking text with overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

    def process_pdf(self, file_path):
        """
        Extracts text from a PDF, splits it into chunks, and generates embeddings.

        Args:
            file_path (str): Path to the PDF file.

        Returns:
            List[Dict]: A list of dictionaries, each containing:
                - 'content': Chunked text.
                - 'embedding': Embedding vector for the chunk.
                - 'doc_metadata': Metadata including file source.

        Raises:
            Exception: Any issue during reading, processing, or embedding.
        """
        try:
            # Extract text from the entire PDF
            text = ""
            with open(file_path, "rb") as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"

            # Split the full text into manageable chunks
            chunks = self.splitter.split_text(text)

            # Generate embeddings for each chunk
            embeddings = self.embed_model.encode(chunks)

            # Return structured data: content + vector + metadata
            return [{
                "content": chunk,
                "embedding": emb.tolist(),
                "doc_metadata": {"source": file_path}
            } for chunk, emb in zip(chunks, embeddings)]

        except Exception as e:
            # Log and raise any errors encountered during processing
            logger.error(f"Processing error: {str(e)}")
            raise
