from pydantic import BaseModel


class PostResponse(BaseModel):
    image: str | None
    content: str
    username: str
    user_id: int