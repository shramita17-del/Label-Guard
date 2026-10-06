import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Setup PostgreSQL Connection (can be overridden via environment variables)
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/labelguard")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
