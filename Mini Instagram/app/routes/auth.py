from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logger import logger
from app.core.security import create_access_token, hash_password, verify_password
from app.db.database import get_db
from app.dependencies.auth import get_current_user_from_session, get_current_user_from_token
from app.models.user import User
from app.schemas.auth import LoginSchema, RegisterSchema, TokenResponse

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/register")
def register_user(data: RegisterSchema, db: Session = Depends(get_db)):
    logger.info(f"[REGISTER] Request received for username={data.username}, email={data.email}")

    existing_user = db.query(User).filter(
        (User.username == data.username) | (User.email == data.email)
    ).first()

    if existing_user:
        logger.warning("[REGISTER] Failed because username/email already exists")
        raise HTTPException(status_code=400, detail="Username or email already exists")

    user = User(
        username=data.username,
        email=data.email,
        password=hash_password(data.password),
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    logger.info(f"[REGISTER] Success for user_id={user.id}")
    return {"message": "User registered successfully"}


@router.post("/login-session", response_model=TokenResponse)
def login_with_session(
    data: LoginSchema,
    request: Request,
    db: Session = Depends(get_db),
):
    logger.info(f"[SESSION LOGIN] Attempt for identity={data.identity}")

    user = db.query(User).filter(
        (User.username == data.identity) | (User.email == data.identity)
    ).first()

    if not user:
        logger.warning(f"[SESSION LOGIN] User not found for identity={data.identity}")
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not verify_password(data.password, user.password):
        logger.warning(f"[SESSION LOGIN] Wrong password for user_id={user.id}")
        raise HTTPException(status_code=401, detail="Invalid credentials")

    request.session["user_id"] = user.id
    request.session["username"] = user.username

    access_token = create_access_token(
        data={"sub": str(user.id), "username": user.username},
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    )

    logger.info(f"[SESSION LOGIN] Success for user_id={user.id}")

    return TokenResponse(
        access_token=access_token,
        user_id=user.id,
        username=user.username,
        message="Login successful",
    )


@router.post("/token", response_model=TokenResponse)
def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    logger.info(f"[TOKEN LOGIN] Attempt for username_or_email={form_data.username}")

    user = db.query(User).filter(
        (User.username == form_data.username) | (User.email == form_data.username)
    ).first()

    if not user:
        logger.warning("[TOKEN LOGIN] User not found")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )

    if not verify_password(form_data.password, user.password):
        logger.warning(f"[TOKEN LOGIN] Wrong password for user_id={user.id}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )

    access_token = create_access_token(
        data={"sub": str(user.id), "username": user.username},
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    )

    logger.info(f"[TOKEN LOGIN] Token issued for user_id={user.id}")
    return TokenResponse(
        access_token=access_token,
        user_id=user.id,
        username=user.username,
        message="Token created successfully",
    )


@router.get("/me")
def get_logged_in_user(user: User = Depends(get_current_user_from_session)):
    logger.info(f"[SESSION PROFILE] Returning profile for user_id={user.id}")
    return {
        "user_id": user.id,
        "username": user.username,
        "email": user.email,
    }


@router.get("/me-token")
def get_logged_in_user_from_token(user: User = Depends(get_current_user_from_token)):
    logger.info(f"[TOKEN PROFILE] Returning profile for user_id={user.id}")
    return {
        "user_id": user.id,
        "username": user.username,
        "email": user.email,
    }


@router.post("/logout-session")
def logout_session(request: Request):
    user_id = request.session.get("user_id")
    logger.info(f"[LOGOUT] Logout requested for session user_id={user_id}")
    request.session.clear()
    logger.info(f"[LOGOUT] Session cleared for user_id={user_id}")
    return {"message": "Logged out from session successfully"}
