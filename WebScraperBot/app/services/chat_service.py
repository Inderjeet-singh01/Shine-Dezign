from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.logger import logger
from app.prompts.chat_prompt import chat_prompt
from app.services.scraper_service import scrape_website
from app.services.memory_service import ensure_session, save_message, format_history
from app.services.memory_service import *

# Grok only: this uses xAI's OpenAI-compatible endpoint.
# Later, if you want to change provider/model, update this block only.
llm = ChatOpenAI(
    model=settings.grok_model,
    api_key=settings.grok_api_key,
    base_url="https://api.groq.com/openai/v1",
    temperature=0.2,
)

chain = chat_prompt | llm


def generate_answer(query: str, session_id: str | None = None):
    if not session_id or session_id.strip() == "":
        session_id = create_session()

    website_data = scrape_website(settings.default_scrape_url)
    history = format_history(session_id)

    response = chain.invoke({
        "website_data": website_data,
        "history": history,
        "query": query
    })

    answer = response.content

    save_message(session_id, "user", query)
    save_message(session_id, "assistant", answer)

    return {
        "session_id": session_id,
        "answer": answer
    }