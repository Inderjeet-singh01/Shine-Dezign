from pydantic_settings import BaseSettings, SettingsConfigDict

from src.logs.logger import logger

'''This class is responsible for loading environment variables from the .env file
   and providing them as attributes.'''

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    DB_CONNECTION: str
    SECRET_KEY : str
    ALGORITHM : str
    ACCESS_TOKEN_EXPIRE_MINUTES : int 
    
settings = Settings() 