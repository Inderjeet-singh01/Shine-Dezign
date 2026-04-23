import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from app.core.config import settings
from app.core.logger import logger
from app.db.database import Base, engine
from app.routes import auth, pages, posts

app = FastAPI(title="Blog Project")

logger.info("🚀 Blog App Starting...")

app.add_middleware(
    SessionMiddleware,
    secret_key=settings.SESSION_SECRET_KEY,
    session_cookie="blog_session",
    max_age=60 * 60 * 24,
    same_site="lax",
    https_only=False,
)

if not os.path.exists(settings.UPLOAD_DIR):
    os.makedirs(settings.UPLOAD_DIR)

Base.metadata.create_all(bind=engine)

app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")
app.include_router(auth.router)
app.include_router(posts.router)
app.include_router(pages.router)
