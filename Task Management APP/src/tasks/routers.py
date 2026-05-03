from fastapi import APIRouter, HTTPException, Depends, status 
from typing import List

from sqlalchemy.orm import Session

from src.logs.logger import logger
from src.tasks import controller
from src.tasks.dtos import TaskSchema, TaskResponseSchema
from src.utils.db import get_db
from src.utils.helpers import is_authenticated
from src.user.models import UserModel


task_routes = APIRouter(prefix="/tasks", tags=["tasks"]) 


# Define the route for creating a new task
@task_routes.post("/create-task", response_model=TaskResponseSchema, status_code=status.HTTP_201_CREATED )
def create_task( body:TaskSchema, db:Session =Depends(get_db), user:UserModel = Depends(is_authenticated) ):
    logger.info("🧾Received Request to create a new task with title: %s", body.title)

    result = controller.create_task(body, db, user)

    logger.info("✅ Successfully created a new task with title: %s", body.title)
    return result


# Define the route for retrieving all tasks
@task_routes.get("/all-tasks", response_model=List[TaskResponseSchema], status_code=status.HTTP_200_OK)
def get_all_tasks( db:Session=Depends(get_db) , user:UserModel = Depends(is_authenticated) ):
    logger.info("🧾Received Request to retrieve all tasks from the database.")

    result = controller.get_tasks(db, user)

    logger.info("✅ Successfully retrieved all tasks from the database.")
    return result


# Define the route for retrieving a Single T ask by its ID
@task_routes.get("/one-task/{task_id}", response_model=TaskResponseSchema, status_code=status.HTTP_200_OK) 
def get_one_task(task_id: int, db:Session=Depends(get_db), user:UserModel = Depends(is_authenticated) ):
    logger.info("🧾Received Request to retrieve task with id '%d' from the database.", task_id)

    result = controller.get_one_task(task_id, db)

    logger.info("✅ Successfully retrieved task with id '%d'", task_id)
    return result


# Define the route for updating a task by its ID
@task_routes.put("/update-task/{task_id}", response_model=TaskResponseSchema, status_code=status.HTTP_201_CREATED)
def update_task(task_id: int, body: TaskSchema, db:Session=Depends(get_db), user:UserModel = Depends(is_authenticated) ):
    logger.info("🧾Received Request to update task with id '%d' in the database.", task_id)

    result = controller.update_task(task_id, body, db, user)

    logger.info("✅ Successfully updated task with id '%d'", task_id)
    return result

# Define the route for deleting a task by its ID    
@task_routes.delete("/delete-task/{task_id}", response_model=None, status_code=status.HTTP_204_NO_CONTENT)
def delete_task(task_id: int, db:Session=Depends(get_db), user:UserModel = Depends(is_authenticated) ):
    logger.info("🧾Received Request to delete task with id '%d' from the database.", task_id)

    result = controller.delete_task(task_id, db, user)

    logger.info("✅ Successfully deleted task with id '%d'", task_id)
    return result