# WebScraperBot Run Guide

FastAPI + LangChain + Grok/xAI based website scraper chatbot.

## 1. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

## 2. Install packages

```bash
pip install -r requirements.txt
```

## 3. Create `.env`

```bash
cp .env.example .env
```

Update `.env`:

```env
XAI_API_KEY=xai-your-api-key-here
DEFAULT_SCRAPE_URL=https://openai.com
GROK_MODEL=grok-3-mini
```

Use only xAI/Grok API key from https://console.x.ai.

## 4. Run server

```bash
uvicorn main:app --reload
```

Open Swagger:

```txt
http://127.0.0.1:8000/docs
```

## 5. First chat request

Do not send session_id first time.

```json
{
  "query": "latest model of chatgpt"
}
```

Response will contain session_id:

```json
{
  "session_id": "generated-session-id",
  "answer": "..."
}
```

## 6. Continue same conversation

```json
{
  "session_id": "generated-session-id",
  "query": "its price?"
}
```

## 7. View one session history

```txt
GET /chat/history/{session_id}
```

## 8. View all chat history

```txt
GET /chat/history
```

## Important

Current chat memory is temporary RAM memory. If server restarts, chat history is cleared.
