from src.logs.logger import logger
from src.utils.setting import Settings
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

'''This Class Checks & Validates dependencies before starting the app'''
class Checkings():

    # (settings.py) Checking and validating data. 
    settings = Settings()
    logger.info("🔧Loading environment variables from .env file.")

    # (db.py) Create the database engine using the connection string from the settings 
    engine = create_engine(url=settings.DB_CONNECTION)
    logger.info("🔌Database connection established.")

    # (db.py) Create a session factory that will be used to create database sessions
    SessionLocal = sessionmaker(bind=engine) 
    logger.info("🛠️Database session factory created.") 






