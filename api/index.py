# api/index.py
import os
import csv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# If your logic lives in support_ticket_chatbot.py, import it here
# from support_ticket_chatbot import answer  # example

app = FastAPI()

# Load data once at cold start (Serverless best practice)
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "support_tickets.csv")

tickets = []
with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    tickets = list(reader)

@app.get("/")
def health():
    return {"ok": True, "count": len(tickets)}

@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")
    # --- Your existing logic here ---
    # result = answer(prompt, tickets)  # <- call your function(s)
    # For demo purposes:
    result = f"Echo: {prompt}. Tickets loaded: {len(tickets)}"
    return JSONResponse({"reply": result})
