from sqlalchemy import Column, Integer, String, JSON
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class ChecklistRule(Base):
    __tablename__ = "checklist_rules"
    id = Column(Integer, primary_key=True, index=True)
    category = Column(String, index=True)  
    check_id = Column(String, unique=True, index=True)
    name = Column(String)
    rule_ref = Column(String)
    severity = Column(String)
    requirement = Column(String)

class RuleBase(Base):
    """Stores the deeper legal-text dataset (with GSR amendment history)"""
    __tablename__ = "rules_base"
    id = Column(Integer, primary_key=True, index=True)
    rule_ref = Column(String, unique=True, index=True)
    legal_text = Column(String)
    amendment_history = Column(JSON)
