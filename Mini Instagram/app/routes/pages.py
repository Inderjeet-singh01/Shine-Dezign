from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.core.config import settings
from app.core.logger import logger

router = APIRouter(tags=["Pages"])

templates = Jinja2Templates(directory=settings.TEMPLATES_DIR)


@router.get("/", response_class=HTMLResponse)
def home_page(request: Request):
    logger.info("[PAGE] Home page opened")
    return templates.TemplateResponse(request=request, name="home.html", context={})


@router.get("/register-page", response_class=HTMLResponse)
def register_page(request: Request):
    logger.info("[PAGE] Register page opened")
    return templates.TemplateResponse(request=request, name="register.html", context={})


@router.get("/login-page", response_class=HTMLResponse)
def login_page(request: Request):
    logger.info("[PAGE] Login page opened")
    return templates.TemplateResponse(request=request, name="login.html", context={})


@router.get("/create-post-page", response_class=HTMLResponse)
def create_post_page(request: Request):
    user_id = request.session.get("user_id")
    if not user_id:
        logger.warning("[PAGE] Unauthorized access to create post page")
        return RedirectResponse(url="/login-page", status_code=302)

    logger.info(f"[PAGE] Create post page opened for user_id={user_id}")
    return templates.TemplateResponse(request=request, name="create_post.html", context={})


@router.get("/posts-page", response_class=HTMLResponse)
def posts_page(request: Request):
    logger.info("[PAGE] Posts page opened")
    return templates.TemplateResponse(request=request, name="posts.html", context={})


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard_page(request: Request):
    user_id = request.session.get("user_id")
    if not user_id:
        logger.warning("[PAGE] Unauthorized access to dashboard")
        return RedirectResponse(url="/login-page", status_code=302)

    logger.info(f"[PAGE] Dashboard opened for user_id={user_id}")
    return templates.TemplateResponse(request=request, name="dashboard.html", context={})
