import os
from dotenv import load_dotenv  # type: ignore

load_dotenv()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))


class Settings:
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./blog.db")

    TEMPLATES_DIR = os.path.join(BASE_DIR, os.getenv("TEMPLATES_DIR", "app/templates"))
    STATIC_DIR = os.path.join(BASE_DIR, os.getenv("STATIC_DIR", "app/static"))
    UPLOAD_DIR = os.path.join(BASE_DIR, os.getenv("UPLOAD_DIR", "app/uploads"))

    SESSION_SECRET_KEY = os.getenv(
        "SESSION_SECRET_KEY",
        "change-this-session-secret-in-production",
    )

    JWT_SECRET_KEY = os.getenv(
        "JWT_SECRET_KEY",
        "change-this-jwt-secret-in-production",
    )
    JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))


settings = Settings()
