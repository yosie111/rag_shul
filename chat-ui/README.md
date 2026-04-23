# Chat UI

A Hebrew RTL chat interface for asking halacha (Jewish law) questions, powered by OpenAI GPT.

🔗 **Live demo: [halacha-chat.vercel.app](https://halacha-chat.vercel.app)**

---

## Setup (both options require this)

**1. Clone the repo**
```bash
git clone https://github.com/egozi/rag_shul.git
cd rag_shul/chat-ui
```

**2. Add your OpenAI API key**
```bash
cp .env.example .env
```
Open `.env` and replace the placeholder with your key:
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
python server.py
```
Open [http://localhost:3000](http://localhost:3000)

---

## Option B — Run with Vercel CLI

Install the Vercel CLI once:
```bash
npm install -g vercel
```
Then run:
```bash
vercel dev
```
Open [http://localhost:3000](http://localhost:3000)

> Use this option if you plan to deploy to Vercel — it simulates the production environment exactly.

---

## Deploy to Vercel

1. Create a project at [vercel.com](https://vercel.com)
2. Go to **Settings → Environment Variables** and add `OPENAI_API_KEY`
3. Run:
```bash
vercel --prod
```

---

## Files

| File | Purpose |
|------|---------|
| `index.html` | Chat UI |
| `api/chat.py` | Serverless function that calls OpenAI |
| `server.py` | Local server for Option A |
| `requirements.txt` | Python dependencies |
| `.env.example` | API key template — never commit a real key |
| `vercel.json` | Vercel configuration |
