from fastapi import  HTTPException

from src.tasks.dtos import TaskSchema
from sqlalchemy.orm import Session
from src.tasks.models import TaskModel
from src.logs.logger import logger
from src.user.models import UserModel


#Contains the business logic for CreateTask Route
def create_task(body :TaskSchema, db: Session, user:UserModel):
    logger.info("📝Creating a new task with title: %s", body.title)

    # Convert user input to Dict from Json(reveived from req)
    data = body.model_dump()

    # TaskModel is object of Class TaskModel(models.py) which is used to create a new task in the database.
    new_task = TaskModel(title=data["title"], 
                         description=data["description"],
                         is_completed=data["is_completed"],
                         user_id=user.id )
    db.add(new_task)    
    db.commit() 
    db.refresh(new_task)

    logger.info("📝Task created successfully with title: %s", body.title)
    return new_task


#Contains the business logic for GetAllTasks Route
def get_tasks(db: Session, user:UserModel ):
    logger.info("📋Retrieving all tasks from the database.")
    tasks= db.query(TaskModel).filter(TaskModel.user_id == user.id).all()
    logger.info("📋Retrieved all tasks from the database.")

    return tasks
 

#Contains the business logic for GetOneTask Route
def get_one_task(task_id: int, db: Session):
    logger.info("📋Retrieving task with id '%d' from the database.", task_id)

    #task that mached the id provided in the request
    one_task= db.query(TaskModel).filter(TaskModel.id == task_id).first()

    if not one_task:
        logger.warning("⚠️Task with id '%d' not found in the database.", task_id)
        raise HTTPException(status_code=404, detail="Task not found")
    
    logger.info("📋Retrieved task with id '%d' from the database.", task_id)
    
    return one_task
    


#Contains the business logic for UpdateTask Route
def update_task(task_id: int, body: TaskSchema, db: Session, user:UserModel):
    logger.info("🔄 Updating task with id '%d' in the database.", task_id)

    one_task:TaskModel = db.query(TaskModel).filter(TaskModel.id == task_id).first()

    if not one_task:
        logger.warning("⚠️ Task with id '%d' not found in the database.", task_id)
        raise HTTPException(status_code=404, detail="Task not found")

    if one_task.user_id != user.id:
        logger.warning("⚠️ Task with id '%d', you are not allowed to update this Task.", task_id)
        raise HTTPException(status_code=401, detail="You are not allowed to update this task")  

    data = body.model_dump(exclude_unset=True)

    for key, value in data.items():
        setattr(one_task, key, value)

    db.add(one_task)
    db.commit()
    db.refresh(one_task)

    logger.info("✅ Updated task with id '%d' in the database.", task_id)

    return one_task



#Contains the business logic for DeleteTask Route
def delete_task(task_id: int, db: Session, user:UserModel):
    logger.info("🗑️Deleting task with id '%d' from the database.", task_id)

    #task that mached the id provided in the request
    one_task = db.query(TaskModel).filter(TaskModel.id == task_id).first()

    if not one_task:
        logger.warning("⚠️Task with id '%d' not found in the database.", task_id)
        raise HTTPException(status_code=404, detail="Task not found")

    if one_task.user_id != user.id:
        logger.warning("⚠️ Task with id '%d', you are not allowed to delete this Task.", task_id)
        raise HTTPException(status_code=401, detail="You are not allowed to delete this task")

    db.delete(one_task)
    db.commit()

    logger.info("🗑️Deleted task with id '%d' from the database.", task_id)
    
    return None