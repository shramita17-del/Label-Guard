import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Setup PostgreSQL / SQLite Connection (can be overridden via environment variables)
# Uses SQLite as safe fallback if no PostgreSQL DB is active
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./labelguard.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
