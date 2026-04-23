# Chat UI

A Hebrew RTL chat interface for asking halacha (Jewish law) questions, powered by OpenAI GPT.

🔗 **Live demo: [halacha-chat.vercel.app](https://halacha-chat.vercel.app)**

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/egozi/rag_shul.git
cd rag_shul
```

**2. Add your OpenAI API key**
```bash
cp chat-ui/.env.example chat-ui/.env
```
Open `chat-ui/.env` and replace the placeholder with your key:
```
OPENAI_API_KEY=sk-...
```
Get a key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

**3. Install Python dependencies**
```bash
pip install -r requirements.txt
```

---

## Option A — Run locally (no extra tools)

```bash
python run_chat.py
```
Open [http://localhost:3000](http://localhost:3000)

---

## Option B — Run with Vercel CLI

Install the Vercel CLI once:
```bash
npm install -g vercel
```
Then run from the `chat-ui` folder:
```bash
cd chat-ui
vercel dev
```
Open [http://localhost:3000](http://localhost:3000)

> Use this option if you plan to deploy to Vercel — it simulates the production environment exactly.

---

## Deploy to Vercel

```bash
cd chat-ui
vercel --prod
```

> First time? Create a project at [vercel.com](https://vercel.com) and add `OPENAI_API_KEY` under **Settings → Environment Variables**.

---

## Files

| File | Purpose |
|------|---------|
| `run_chat.py` | Local dev server (run from repo root) |
| `requirements.txt` | Python dependencies |
| `chat-ui/index.html` | Chat UI |
| `chat-ui/api/chat.py` | Serverless function that calls OpenAI |
| `chat-ui/.env.example` | API key template — never commit a real key |
| `chat-ui/vercel.json` | Vercel configuration |
