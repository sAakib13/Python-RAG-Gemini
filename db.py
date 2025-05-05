from sqlalchemy import create_engine, Column, Integer, String, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector
import os
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Verify database URL exists
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    content = Column(String)
    embedding = Column(Vector(384))  # Match sentence-transformers dimension
    doc_metadata = Column(JSON)


def init_db():
    Base.metadata.create_all(bind=engine)
