# My Chat — a simple ChatGPT-style app

FastAPI backend + one HTML page. Chats are saved in a local SQLite file (`chats.db`).
Models come from OpenRouter.

## Features

* Streaming replies (text appears word by word)
* Sidebar of past chats, saved across restarts
* Model picker (searchable list loaded from OpenRouter, or type any model ID)
* System prompt per chat
* Image upload (up to 4 images, 5 MB each)

## Files

```
chat-app/
├── app.py             # backend: API endpoints, database, calls to OpenRouter
├── static/index.html  # the chat screen
├── requirements.txt   # Python packages
├── .env.example       # copy to .env and add your key
└── .gitignore         # keeps .env and chats.db out of Git
```

## Setup (Windows, PowerShell) — needs Python 3.10 or newer

```powershell
cd chat-app
python -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
notepad .env          # paste your NEW OpenRouter key, save, close
uvicorn app:app --reload
```

Open **http://localhost:8000**.

If PowerShell blocks the activate script, run this once, then try again:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

(Mac/Linux: use `source .venv/bin/activate` and `cp` instead of `copy`.)

## Using it

1. Pick a model in the top bar (start typing to search, e.g. "gemini" or "claude").
The note beside it tells you whether the model accepts images.
2. Optional: click **System prompt** to set instructions for this chat.
3. Type and press Enter. Use 📎 to attach images.

## API endpoints (also browsable at http://localhost:8000/docs)

|Method|Path|What it does|
|-|-|-|
|GET|`/api/config`|Default model and system prompt|
|GET|`/api/models`|Models available on OpenRouter|
|GET|`/api/conversations`|List chats (newest first)|
|POST|`/api/conversations`|Start a chat|
|GET|`/api/conversations/{id}`|One chat with its messages|
|PATCH|`/api/conversations/{id}`|Change title, model or system prompt|
|DELETE|`/api/conversations/{id}`|Delete a chat|
|POST|`/api/conversations/{id}/messages`|Send a message; reply streams back|

## Known limits (fine for local use, fix before sharing)

* No login — anyone who can reach the server can read all chats. Keep it on your laptop.
* Only the last 20 messages are sent to the model, so very long chats "forget" early turns.
* Images are stored inside `chats.db`, so the file grows with image-heavy chats.
* No "stop generating" button yet.
* Errors from the model appear at the end of the reply with a ⚠️.

