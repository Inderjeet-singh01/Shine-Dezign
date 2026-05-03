from sqlalchemy import Column, Integer, String, Boolean
from src.utils.db import Base

class UserModel(Base):
    __tablename__ = "user_table"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    email = Column(String(150), nullable=False)   
    username = Column(String(150), nullable=False)
    hash_password = Column(String(150), nullable=False)
