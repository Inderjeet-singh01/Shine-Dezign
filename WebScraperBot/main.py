from fastapi import FastAPI
from app.routers.chat_router import router as chat_router
from app.core.logger import logger

app = FastAPI(
    title="WebScraperBot",
    description="FastAPI + LangChain + Grok website scraper chatbot",
    version="1.0.0",
)

app.include_router(chat_router, prefix="/chat", tags=["Chat"])

logger.info("🚀Starting WebScraperBot")

@app.get("/")
def root():
    return {"message": "WebScraperBot is running"}


@app.get("/health")
def health_check():
    return {"status": "ok"}
