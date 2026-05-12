from fastapi import APIRouter

from app.schemas.chat_schema import ChatRequest, ChatResponse
from app.services.chat_service import generate_answer
from app.services.memory_service import get_history, get_all_chats

router = APIRouter()


@router.post("/", response_model=ChatResponse)
def chat(request: ChatRequest):
    return generate_answer(
        query=request.query,
        session_id=request.session_id,
    )
 

@router.get("/history")
def all_chat_history():
    return {
        "total_sessions": len(get_all_chats()),
        "all_chats": get_all_chats(),
    }


@router.get("/history/{session_id}")
def session_chat_history(session_id: str):
    return {
        "session_id": session_id,
        "history": get_history(session_id),
    }
