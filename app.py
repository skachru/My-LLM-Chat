"""
Simple ChatGPT-style chat app - backend (FastAPI + OpenRouter + SQLite).

Run it:
    uvicorn app:app --reload
Then open http://localhost:8000 in your browser.

How it works, in one paragraph:
The model API is stateless - it remembers nothing between calls. So we store
every conversation in a small SQLite database (chats.db). Each time you send a
message, we save it, load the recent history of that conversation, send the
whole history to the model, stream the reply back to the browser word by word,
and finally save the reply too.
"""

import json
import os
import sqlite3
import time
from pathlib import Path

import httpx
import openai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from openai import OpenAI
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# 1. Settings (read from the .env file - never hard-code keys in code)
# ---------------------------------------------------------------------------
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "").strip()
DEFAULT_SYSTEM_PROMPT = os.getenv("DEFAULT_SYSTEM_PROMPT", "You are a helpful assistant.")
# Only the most recent N messages are sent to the model, so long chats
# don't overflow the model's context window or keep getting more expensive.
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "20"))

if not OPENROUTER_API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY is missing. Copy .env.example to .env and put your key in it."
    )

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    timeout=90,  # seconds)
)


BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "chats.db"
MAX_IMAGES_PER_MESSAGE = 4
MAX_IMAGE_DATA_URL_LENGTH = 8_000_000  # about a 5-6 MB image once base64-encoded

app = FastAPI(title="Simple Chat App")


# ---------------------------------------------------------------------------
# 2. Database helpers (SQLite = one local file, no server to install)
# ---------------------------------------------------------------------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # lets us read columns by name: row["title"]
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def db_execute(sql, params=()):
    """Run an INSERT/UPDATE/DELETE. Returns the new row's id for INSERTs."""
    conn = get_db()
    try:
        cur = conn.execute(sql, params)
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def db_fetch_all(sql, params=()):
    conn = get_db()
    try:
        return [dict(row) for row in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def db_fetch_one(sql, params=()):
    rows = db_fetch_all(sql, params)
    return rows[0] if rows else None


def init_db():
    conn = get_db()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                title         TEXT NOT NULL,
                model         TEXT NOT NULL,
                system_prompt TEXT NOT NULL,
                created_at    REAL NOT NULL,
                updated_at    REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL
                                REFERENCES conversations(id) ON DELETE CASCADE,
                role            TEXT NOT NULL,          -- 'user' or 'assistant'
                text            TEXT NOT NULL,
                images          TEXT NOT NULL DEFAULT '[]',  -- JSON list of data URLs
                created_at      REAL NOT NULL
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


init_db()


def get_conversation_or_404(conv_id):
    conv = db_fetch_one("SELECT * FROM conversations WHERE id = ?", (conv_id,))
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


# ---------------------------------------------------------------------------
# 3. Shapes of the data the browser sends us (FastAPI validates these)
# ---------------------------------------------------------------------------
class NewConversation(BaseModel):
    model: str | None = None
    system_prompt: str | None = None


class ConversationUpdate(BaseModel):
    title: str | None = None
    model: str | None = None
    system_prompt: str | None = None


class NewMessage(BaseModel):
    text: str = ""
    images: list[str] = []  # data URLs, e.g. "data:image/png;base64,iVBOR..."


# ---------------------------------------------------------------------------
# 4. Turning stored messages into the format the model API expects
# ---------------------------------------------------------------------------
def to_api_message(row):
    images = json.loads(row["images"])
    if not images:
        return {"role": row["role"], "content": row["text"]}

    # A message with images is a list of "parts": text plus one part per image.
    parts = []
    if row["text"]:
        parts.append({"type": "text", "text": row["text"]})
    for url in images:
        parts.append({"type": "image_url", "image_url": {"url": url}})
    return {"role": row["role"], "content": parts}


def friendly_error(e):
    """Turn API errors into messages a person can act on."""
    if isinstance(e, openai.AuthenticationError):
        return "The API key was rejected. Check OPENROUTER_API_KEY in your .env file."
    if isinstance(e, openai.NotFoundError):
        return "Model not found. Check the model ID (format: provider/model-name)."
    if isinstance(e, openai.RateLimitError):
        return "Rate limit reached. Wait a moment and try again."
    if isinstance(e, openai.APIConnectionError):
        return "Could not reach OpenRouter. Check your internet connection."
    if isinstance(e, openai.APIStatusError):
        if e.status_code == 402:
            return "Out of OpenRouter credits. Top up your account and try again."
        return f"OpenRouter returned an error ({e.status_code}): {e.message}"
    return f"Unexpected error: {e}"


# ---------------------------------------------------------------------------
# 5. API endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def home():
    """Serve the chat page."""
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/config")
def get_config():
    return {"default_model": DEFAULT_MODEL,"default_system_prompt": DEFAULT_SYSTEM_PROMPT,}


_models_cache = {"models": None, "fetched_at": 0.0}


@app.get("/api/models")
def list_models():
    """List the models available on OpenRouter (cached for an hour)."""
    if _models_cache["models"] and time.time() - _models_cache["fetched_at"] < 3600:
        return {"models": _models_cache["models"]}

    try:
        response = httpx.get("https://openrouter.ai/api/v1/models", timeout=20)
        response.raise_for_status()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Could not load the model list: {e}")

    models = []
    for m in response.json().get("data", []):
        input_types = (m.get("architecture") or {}).get("input_modalities") or []
        models.append(
            {
                "id": m["id"],
                "name": m.get("name") or m["id"],
                "supports_images": "image" in input_types,
            }
        )
    models.sort(key=lambda m: m["name"].lower())

    _models_cache["models"] = models
    _models_cache["fetched_at"] = time.time()
    return {"models": models}


@app.get("/api/conversations")
def list_conversations():
    """All conversations, newest first - this fills the sidebar."""
    return db_fetch_all(
        "SELECT id, title, model, updated_at FROM conversations ORDER BY updated_at DESC"
    )


@app.post("/api/conversations")
def create_conversation(body: NewConversation):
    model = (body.model or DEFAULT_MODEL).strip()
    if not model:
        raise HTTPException(status_code=400, detail="Pick a model first.")
    system_prompt = body.system_prompt if body.system_prompt is not None else DEFAULT_SYSTEM_PROMPT

    now = time.time()
    conv_id = db_execute(
        "INSERT INTO conversations (title, model, system_prompt, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        ("New chat", model, system_prompt, now, now),
    )
    return get_conversation_or_404(conv_id)


@app.get("/api/conversations/{conv_id}")
def get_conversation(conv_id: int):
    conv = get_conversation_or_404(conv_id)
    messages = db_fetch_all(
        "SELECT id, role, text, images, created_at FROM messages "
        "WHERE conversation_id = ? ORDER BY id",
        (conv_id,),
    )
    for m in messages:
        m["images"] = json.loads(m["images"])
    conv["messages"] = messages
    return conv


@app.patch("/api/conversations/{conv_id}")
def update_conversation(conv_id: int, body: ConversationUpdate):
    get_conversation_or_404(conv_id)
    if body.title is not None:
        db_execute("UPDATE conversations SET title = ? WHERE id = ?", (body.title.strip() or "New chat", conv_id))
    if body.model is not None and body.model.strip():
        db_execute("UPDATE conversations SET model = ? WHERE id = ?", (body.model.strip(), conv_id))
    if body.system_prompt is not None:
        db_execute("UPDATE conversations SET system_prompt = ? WHERE id = ?", (body.system_prompt, conv_id))
    return get_conversation_or_404(conv_id)


@app.delete("/api/conversations/{conv_id}")
def delete_conversation(conv_id: int):
    get_conversation_or_404(conv_id)
    db_execute("DELETE FROM conversations WHERE id = ?", (conv_id,))  # messages go too (CASCADE)
    return {"deleted": conv_id}


@app.post("/api/conversations/{conv_id}/messages")
def send_message(conv_id: int, msg: NewMessage):
    """Save the user's message, then stream the model's reply back as plain text."""
    conv = get_conversation_or_404(conv_id)

    text = msg.text.strip()
    if not text and not msg.images:
        raise HTTPException(status_code=400, detail="Message is empty.")
    if len(msg.images) > MAX_IMAGES_PER_MESSAGE:
        raise HTTPException(status_code=400, detail=f"Up to {MAX_IMAGES_PER_MESSAGE} images per message.")
    for url in msg.images:
        if not url.startswith("data:image/"):
            raise HTTPException(status_code=400, detail="Images must be uploaded image files.")
        if len(url) > MAX_IMAGE_DATA_URL_LENGTH:
            raise HTTPException(status_code=400, detail="An image is too large (keep it under about 5 MB).")

    # Step 1: save the user's message.
    now = time.time()
    db_execute(
        "INSERT INTO messages (conversation_id, role, text, images, created_at) VALUES (?, ?, ?, ?, ?)",
        (conv_id, "user", text, json.dumps(msg.images), now),
    )
    # Name the chat after its first message, like ChatGPT does.
    if conv["title"] == "New chat":
        title = text[:50] or "Image chat"
        db_execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conv_id))

    # Step 2: build what we send - system prompt + the most recent messages.
    recent = db_fetch_all(
        "SELECT role, text, images FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
        (conv_id, MAX_HISTORY_MESSAGES),
    )
    recent.reverse()  # oldest first
    api_messages = []
    if conv["system_prompt"].strip():
        api_messages.append({"role": "system", "content": conv["system_prompt"]})
    api_messages += [to_api_message(row) for row in recent]

    # Step 3: stream the reply. This function runs while the browser is reading.
    def generate():
        reply_parts = []
        try:
            stream = client.chat.completions.create(
                model=conv["model"],
                messages=api_messages,
                stream=True,
            )
            for chunk in stream:
                # Some chunks carry no text (or no choices at all) - skip them.
                if chunk.choices and chunk.choices[0].delta.content:
                    piece = chunk.choices[0].delta.content
                    reply_parts.append(piece)
                    yield piece
        except Exception as e:
            yield f"\n\n⚠️ {friendly_error(e)}"
        finally:
            # Step 4: save the reply (even a partial one, if the user stopped it).
            reply = "".join(reply_parts)
            if reply:
                db_execute(
                    "INSERT INTO messages (conversation_id, role, text, images, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (conv_id, "assistant", reply, "[]", time.time()),
                )
            db_execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (time.time(), conv_id))

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")
