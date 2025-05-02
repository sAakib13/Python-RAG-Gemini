from sqlalchemy import create_engine, Column, Integer, String, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector

DATABASE_URL = "postgresql://neondb_owner:npg_5CUXDgQKM2bc@ep-quiet-rice-a17hh4qs-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    content = Column(String)
    embedding = Column(Vector(768))
    doc_metadata = Column(JSON)  # Renamed the column here

def init_db():
    Base.metadata.create_all(bind=engine)