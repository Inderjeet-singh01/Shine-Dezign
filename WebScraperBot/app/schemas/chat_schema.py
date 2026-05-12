from typing import Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = Field(default=None)

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Enter your question here",
                "session_id": ""
            }
        }


class ChatResponse(BaseModel):
    session_id: str
    answer: str