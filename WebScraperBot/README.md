# WebScraperBot

WebScraperBot is an AI-powered website chatbot built using FastAPI, LangChain, Groq LLM, Requests, and BeautifulSoup.

The project scrapes website content, processes readable text, stores chat memory using session IDs, and allows users to ask questions related to the scraped website.

The chatbot supports:
- Multi-page website scraping
- Session-based memory
- AI-generated answers using Groq LLM
- Clean and modular FastAPI architecture

---

# Features

- AI-powered chatbot
- Website scraping using BeautifulSoup
- Multi-page internal link crawling
- Session-based memory management
- FastAPI backend
- Groq LLM integration
- Clean project structure
- Logging support
- Environment variable support using Pydantic Settings

---

# Tech Stack

- Python
- FastAPI
- LangChain
- Groq API
- BeautifulSoup4
- Requests
- Pydantic Settings
- Uvicorn

---

# Project Structure

```txt
WebScraperBot/
├── main.py
├── requirements.txt
├── .env.example
└── app/
    ├── core/
    │   ├── config.py
    │   └── logger.py
    ├── routers/
    │   └── chat_router.py
    ├── schemas/
    │   └── chat_schema.py
    ├── services/
    │   ├── scraper_service.py
    │   ├── memory_service.py
    │   └── chat_service.py
    └── prompts/
        └── chat_prompt.py
```

---

# How It Works

1. User sends a message to the chatbot API.
2. The scraper service scrapes website content.
3. BeautifulSoup extracts readable text from HTML.
4. Multi-page crawling collects internal page content.
5. LangChain sends the website data and user question to Groq LLM.
6. AI generates a contextual answer.
7. Session memory stores conversation history.

---

# Single Page vs Multi-Page Scraping

The project originally used a single-page scraper.

Single-page scraper:
- Scraped only one webpage
- Did not follow internal links
- Limited website understanding

The old single-page scraping logic is still preserved in `scraper_service.py` as commented code for learning and comparison purposes.

Current implementation:
- Supports multi-page scraping
- Extracts internal links
- Crawls multiple pages
- Improves AI response quality

---

# Example Use Cases

- Documentation chatbot
- AI website assistant
- FAQ assistant
- Technical documentation Q&A
- Educational chatbot

---

# Example Websites

Recommended websites for testing:

- https://docs.python.org/3/
- https://python.langchain.com/docs/introduction/

---

# API Endpoint

## Chat Endpoint

```http
POST /chat
```

### Request Body

```json
{
  "message": "What is middleware in FastAPI?",
  "session_id": null
}
```

### Response

```json
{
  "session_id": "generated-session-id",
  "response": "Middleware in FastAPI..."
}
```

---

# Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
DEFAULT_SCRAPE_URL=https://fastapi.tiangolo.com/
GROQ_MODEL=llama-3.1-8b-instant
```


---

# Author

Inderjeet Singh