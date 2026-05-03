from fastapi import FastAPI

from src.utils.db import engine, Base
from src.logs.logger import logger
from src.tasks.routers import task_routes
from src.user.routers import user_routes
from src.utils.checking import Checkings

# Initialize the Checkings class to set up database connection and session factory
Checking = Checkings()  


''' This is the main entry point of the application.
It initializes the FastAPI app and sets up the database connection.'''

Base.metadata.create_all(bind=engine)
logger.info("📊Database initialized.")


# Initialize FastAPI app and include task routes
app = FastAPI(
    title="Task Manager API",
    description="A simple API to create, read, update and delete tasks.",
    version="1.0.0"
)

app.include_router(task_routes)     # Include the task routes in the main application
app.include_router(user_routes)      # Include the user routes in the main application

@app.get("/")
def home():
    return {"message": "Task Manager App Working✅"}