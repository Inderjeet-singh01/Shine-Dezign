from uuid import uuid4
from typing import Dict, List

# Temporary in-memory chat store.
# Data will be cleared when the server restarts.
memory_store: Dict[str, List[dict]] = {}


def create_session() -> str:
    session_id = str(uuid4())
    memory_store[session_id] = []
    return session_id


def ensure_session(session_id: str | None) -> str:
    if not session_id:
        return create_session()

    if session_id not in memory_store:
        memory_store[session_id] = []

    return session_id


def save_message(session_id: str, role: str, content: str) -> None:
    
    if session_id not in memory_store:
        memory_store[session_id] = []

    memory_store[session_id].append({
        "role": role,
        "content": content,
    })


def get_history(session_id: str) -> List[dict]:
    return memory_store.get(session_id, [])


def get_all_chats() -> Dict[str, List[dict]]:
    return memory_store


def format_history(session_id: str, limit: int = 6) -> str:
    history = get_history(session_id)

    if not history:
        return "No previous conversation."

    recent_history = history[-limit:]
    return "\n".join(
        f"{message['role']}: {message['content']}"
        for message in recent_history
    )
