from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt  # type: ignore
from pwdlib import PasswordHash  # type: ignore

from app.core.config import settings
from app.core.logger import logger

password_hash = PasswordHash.recommended()


def hash_password(password: str) -> str:
    logger.info("[SECURITY] Hashing password for new registration")
    return password_hash.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    logger.info("[SECURITY] Verifying password")
    return password_hash.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})

    logger.info(f"[TOKEN CREATE] Creating JWT token with expiry={expire.isoformat()}")
    return jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )


def decode_access_token(token: str) -> dict:
    try:
        logger.info("[TOKEN DECODE] Decoding JWT token")
        return jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
    except JWTError as exc:
        logger.warning(f"[TOKEN DECODE] Invalid JWT token: {exc}")
        raise
