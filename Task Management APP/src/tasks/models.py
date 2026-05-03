from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey
from src.utils.db import Base

''' Represents a task table in the database.'''
class TaskModel(Base):
    __tablename__ = "user_tasks"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    is_completed = Column(Boolean, default=False)  
    
    user_id = Column(Integer,ForeignKey("user_table.id", ondelete="CASCADE"),nullable=False  )