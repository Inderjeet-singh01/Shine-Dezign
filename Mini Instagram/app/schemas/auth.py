from typing import Optional

from pydantic import BaseModel, EmailStr


class RegisterSchema(BaseModel):
    username: str
    email: EmailStr
    password: str


class LoginSchema(BaseModel):
    identity: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: Optional[int] = None
    username: Optional[str] = None
    message: Optional[str] = None
