from fastapi import HTTPException , Request
from pwdlib import PasswordHash
import jwt
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError  
from datetime import datetime, timedelta

from src.user.dtos import UserSchema, UserLoginSchema
from sqlalchemy.orm import Session
from src.logs.logger import logger
from src.user.models import UserModel
from src.utils.setting import settings

password_hash = PasswordHash.recommended()

def get_password_hash(password):
    return password_hash.hash(password) 

def verify_password(plain_password, hashed_password):
    return password_hash.verify(plain_password, hashed_password)  


# Controller function to handle user registration logic
def register_user(body: UserSchema, db: Session):
    logger.info(f"Registering user with username: {body.username}")

    is_user = db.query(UserModel).filter(UserModel.username == body.username).first()
    if is_user:
        logger.warning(f"Username {body.username} already exists")
        raise HTTPException(status_code=400, detail="Username already exists")
    
    is_user = db.query(UserModel).filter(UserModel.email == body.email).first()
    if is_user:
        logger.warning(f"Email {body.email} already exists")
        raise HTTPException(status_code=400, detail="Email already exists")
    
    hash_password = get_password_hash(body.password)
    logger.info(f"Password hashed for user {body.username}")

    new_user = UserModel(
        name=body.name, 
        username=body.username,
        hash_password=hash_password,
        email=body.email
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    logger.info(f"User {body.username} registered successfully")

    return new_user


#Controller function to handle user login logic
def login_user(body: UserLoginSchema, db: Session):
    logger.info(f"Attempting to log in user with username: {body.username}")

    user = db.query(UserModel).filter(UserModel.username == body.username).first()
    if not user:
        logger.warning(f"User with username {body.username} not found")
        raise HTTPException(status_code=404, detail="User not found")
    
    if not verify_password(body.password, user.hash_password):
        logger.warning(f"Incorrect password for user {body.username}")
        raise HTTPException(status_code=400, detail="Incorrect password") 
    
    exp_time = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    token = jwt.encode({"user_id":user.id, "username":user.username, "exp":exp_time}, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

    logger.info(f"User {body.username} logged in successfully")
    return {"access_token":token}


#Controller function to check if the user is authenticated
def  is_authenticated(request:Request, db: Session):
    try:
        logger.info("Checking authentication status")

        token = request.headers.get("Authorization")
        if not token:
            logger.warning("Authorization header missing")
            raise HTTPException(status_code=401, detail="Authorization header missing")
        
        token = token.split(" ")[-1] 

        data = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]) 

        user_id = data.get("user_id")

        user = db.query(UserModel).filter(UserModel.id == user_id).first()
        if not user:
            logger.warning(f"User with ID {user_id} not found")
            raise HTTPException(status_code=404, detail="User not found")
        
        return user
    
    except ExpiredSignatureError:
        logger.warning("Token has expired")
        raise HTTPException(status_code=401, detail="Token has expired")
        