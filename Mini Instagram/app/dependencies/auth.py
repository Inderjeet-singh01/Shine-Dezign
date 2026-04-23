from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError  # type: ignore
from sqlalchemy.orm import Session

from app.core.logger import logger
from app.core.security import decode_access_token
from app.db.database import get_db
from app.models.user import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


def get_current_user_from_session(request: Request, db: Session = Depends(get_db)) -> User:
    user_id = request.session.get("user_id")

    logger.info(f"[SESSION CHECK] Checking session user_id={user_id}")

    if not user_id:
        logger.warning("[SESSION CHECK] Failed because user not logged in")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Login required",
        )

    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        logger.warning(f"[SESSION CHECK] Failed because user_id={user_id} not found in DB")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid session user",
        )

    logger.info(f"[SESSION CHECK] Success for user_id={user.id}")
    return user


def get_current_user_from_token(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    logger.info("[TOKEN VERIFY] Incoming token verification request")

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = decode_access_token(token)
        user_id = payload.get("sub")

        logger.info(f"[TOKEN VERIFY] Payload received with sub={user_id}")

        if user_id is None:
            logger.warning("[TOKEN VERIFY] Failed because sub is missing in token payload")
            raise credentials_exception

    except JWTError as exc:
        logger.error(f"[TOKEN VERIFY] JWT verification failed: {exc}")
        raise credentials_exception

    user = db.query(User).filter(User.id == int(user_id)).first()

    if not user:
        logger.warning(f"[TOKEN VERIFY] Failed because user_id={user_id} not found in DB")
        raise credentials_exception

    logger.info(f"[TOKEN VERIFY] Success for user_id={user.id}")
    return user
