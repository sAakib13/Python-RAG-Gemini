from sqlalchemy import create_engine, Column, Integer, String, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector
import os

from urllib.parse import quote
from dotenv import load_dotenv

raw_password = os.getenv("DB_PASSWORD")  # Store password separately in .env
encoded_password = quote(raw_password)
DATABASE_URL = f"postgresql://{os.getenv('DB_USER')}:{encoded_password}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}?sslmode=require"
DATABASE_URL = os.getenv("DATABASE_URL")

load_dotenv()
# Get from environment

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