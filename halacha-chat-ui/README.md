# Halacha Chat UI

דף צ'אט בעברית לשאלות בהלכה, עם קריאה ל-OpenAI דרך פונקציית שרת `/api/chat`.

## קבצים

- `index.html` - מסך הצ'אט.
- `api/chat.py` - endpoint שמדבר עם OpenAI.
- `requirements.txt` - תלות Python לפריסה.
- `.env.example` - משתני סביבה לדוגמה.

## הפעלה

בפריסה יש להגדיר:

```bash
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini
```

ה-frontend שולח את היסטוריית השיחה ל-`/api/chat`, והשרת מוסיף system prompt הלכתי לפני הקריאה למודל.
