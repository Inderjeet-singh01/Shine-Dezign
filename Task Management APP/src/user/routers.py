from fastapi import APIRouter, Depends, status, Request

from sqlalchemy.orm import Session
from src.user.dtos import UserSchema, UserResponseSchema, UserLoginSchema
from src.utils.db import get_db 
from src.user import controller
from src.logs.logger import logger


user_routes = APIRouter(prefix="/users", tags=["Users"])


# Route to register a new user 
@user_routes.post("/register", response_model=UserResponseSchema ,status_code=status.HTTP_201_CREATED)
def register(body: UserSchema, db:Session = Depends(get_db)):
    logger.info(f"Received request to register user with username: {body.username}")

    result = controller.register_user(body, db)

    logger.info(f"User {body.username} registration process completed") 
    return result
 

# Route to login a user
@user_routes.post("/login", status_code=status.HTTP_200_OK)
def login(body: UserLoginSchema, db: Session = Depends(get_db)):
    logger.info(f"Received login request for username: {body.username}")

    result = controller.login_user(body, db)

    logger.info(f"User {body.username} login process completed")
    return result    


@user_routes.get("/is-auth", response_model=UserResponseSchema, status_code=status.HTTP_200_OK)
def is_auth(request:Request, db: Session = Depends(get_db)):
    logger.info("Received request to check authentication status")

    result = controller.is_authenticated(request, db)

    logger.info("Authentication status check completed")    
    return result