# ब्रह्मज्ञान (BrahmaGyaan) — a ChatGPT-style chat app

FastAPI backend + one HTML page. Chats are saved in Postgres (works locally and on
serverless hosts like Vercel, which don't allow writing to a local SQLite file).
Models come from OpenRouter.

## Features

* Streaming replies (text appears word by word)
* Sidebar of past chats, saved across restarts
* Model picker (searchable list loaded from OpenRouter, or type any model ID)
* System prompt per chat
* Image upload (up to 4 images, 5 MB each)
* Dark/light themes with a toggle that remembers your choice
* Responsive down to phones, with touch-sized controls

## Files

```
chat-app/
├── app.py             # backend: API endpoints, database, calls to OpenRouter
├── static/index.html  # the chat screen
├── requirements.txt   # Python packages
├── vercel.json        # tells Vercel how to run the FastAPI app
├── .env.example       # copy to .env and add your key + database URL
└── .gitignore         # keeps .env out of Git
```

## Setup (Windows, PowerShell) — needs Python 3.10 or newer

```powershell
cd chat-app
python -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
notepad .env          # paste your NEW OpenRouter key and a Postgres DATABASE_URL, save, close
uvicorn app:app --reload
```

Open **http://localhost:8000**.

If PowerShell blocks the activate script, run this once, then try again:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

(Mac/Linux: use `source .venv/bin/activate` and `cp` instead of `copy`.)

Need a free Postgres database for local dev? [Neon](https://neon.tech) or
[Supabase](https://supabase.com) both give you a connection string in under a
minute — paste it into `DATABASE_URL`.

## Deploying to Vercel

1. Push this repo to GitHub, then import it in Vercel ("Add New Project").
2. In the Vercel project's **Storage** tab, create a Postgres database and
   connect it to the project — this injects a `POSTGRES_URL` env var
   automatically. (Or set `DATABASE_URL` yourself under **Settings → Environment
   Variables** if you're using another Postgres provider.)
3. Also set `OPENROUTER_API_KEY` under **Settings → Environment Variables**
   (your `.env` file is gitignored, so it never reaches Vercel on its own).
4. Redeploy.

**Known limitation:** Vercel's Python functions buffer the full response
before returning it, so replies will appear all at once instead of streaming
in word by word, even though the code streams internally. This only affects
the Vercel deployment — streaming still works when run locally with
`uvicorn`.

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
* Images are stored as base64 text in Postgres, so the database grows with image-heavy chats.
* No "stop generating" button yet.
* Errors from the model appear at the end of the reply with a ⚠️.

