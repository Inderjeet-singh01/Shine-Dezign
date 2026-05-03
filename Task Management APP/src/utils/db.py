from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from src.utils.setting import settings 

''' This module is responsible for setting up the database connection'''
Base = declarative_base()


''' Create the database engine using the connection string from the settings'''
engine = create_engine(url=settings.DB_CONNECTION)

''' Create a session factory that will be used to create database sessions'''
SessionLocal = sessionmaker(bind=engine) 
 

''' This Function is used to get a database session'''
def get_db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()