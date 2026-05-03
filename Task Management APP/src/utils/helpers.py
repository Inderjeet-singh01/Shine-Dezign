from src.logs.logger import logger
from src.user.models import UserModel
from src.utils.setting import settings
from sqlalchemy.orm import Session
from fastapi import HTTPException, Request, Depends
import jwt
from jwt.exceptions import ExpiredSignatureError
from src.utils.db import get_db


def  is_authenticated(request:Request, db: Session=Depends(get_db)):
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
        