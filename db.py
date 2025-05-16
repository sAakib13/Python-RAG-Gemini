from sqlalchemy import create_engine, Column, Integer, String, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the database URL from environment variables
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set")

# Set up SQLAlchemy engine and session factory
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Declare the base class for model definitions
Base = declarative_base()


class Document(Base):
    """
    SQLAlchemy model representing a document chunk with its embedding and metadata.

    Attributes:
        id (int): Primary key identifier.
        content (str): The text content of the document chunk.
        embedding (Vector): 384-dimensional vector for semantic similarity.
        doc_metadata (dict): Additional metadata about the source/document.
    """
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(String)
    embedding = Column(Vector(384))  # Compatible with 'all-MiniLM-L6-v2'
    doc_metadata = Column(JSON)


def init_db():
    """
    Initializes the database by creating all tables defined in the models.
    """
    Base.metadata.create_all(bind=engine)
