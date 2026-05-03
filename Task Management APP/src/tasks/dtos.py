# Schema for Task, used to validate data sent by client and data sent by database.
from pydantic import BaseModel
from typing import Optional


'''Checks the data sent by the client, matches format of data before send to database.
also checks format of data sent by database before send to client.'''

class TaskSchema(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    is_completed: Optional[bool] = None


#Checks format of data sent by database before send to client.
#With the help of this we only send data to user that we want to see the user,if we dont want to show all data that entered in database we can use this schema to send only required data to user.
class TaskResponseSchema(BaseModel):
    id: int
    title: Optional[str] 
    description: Optional[str]
    is_completed: Optional[bool] 
    user_id: int | None = 0
