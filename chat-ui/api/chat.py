from http.server import BaseHTTPRequestHandler
import json
import os

from openai import OpenAI


SYSTEM_PROMPT = """
אתה עוזר לימוד הלכה בעברית.
ענה במשפט אחד בלבד. אל תרחיב אלא אם נשאלת במפורש.
כאשר אפשר, ציין מקור אחד רלוונטי בלבד (שולחן ערוך, משנה ברורה וכד').
אם השאלה דורשת פסיקה למעשה, הוסף: "יש להתייעץ עם רב."
אל תמציא מקור אם אינך בטוח בו.
""".strip()

ALLOWED_ROLES = {"user", "assistant"}
MAX_MESSAGES = 12
MAX_CONTENT_CHARS = 4000


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_POST(self):
        try:
            payload = self._read_json()
            messages = self._clean_messages(payload.get("messages", []))

            if not messages:
                self._send_json(400, {"error": "לא התקבלה שאלה לשליחה."})
                return

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                self._send_json(500, {"error": "חסר OPENAI_API_KEY בסביבת השרת."})
                return

            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                temperature=0.35,
                max_tokens=1200,
            )

            reply = response.choices[0].message.content or ""
            self._send_json(200, {"reply": reply.strip()})
        except json.JSONDecodeError:
            self._send_json(400, {"error": "בקשת JSON לא תקינה."})
        except Exception as exc:
            self._send_json(500, {"error": f"שגיאה מול OpenAI: {exc}"})

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw or "{}")

    def _clean_messages(self, messages):
        clean = []
        if not isinstance(messages, list):
            return clean

        for message in messages[-MAX_MESSAGES:]:
            if not isinstance(message, dict):
                continue

            role = message.get("role")
            content = message.get("content")
            if role not in ALLOWED_ROLES or not isinstance(content, str):
                continue

            content = content.strip()
            if content:
                clean.append({"role": role, "content": content[:MAX_CONTENT_CHARS]})

        return clean

    def _send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
