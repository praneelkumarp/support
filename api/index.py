# api/index.py
import os
import json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import pandas as pd

# If your logic lives in support_ticket_chatbot.py, import it here
# from support_ticket_chatbot import answer  # example

app = FastAPI()

# Load data once at cold start (Serverless best practice)
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "support_tickets.csv")
df = pd.read_csv(CSV_PATH)

@app.get("/")
def health():
    return {"ok": True, "count": len(df)}

@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")
    # --- Your existing logic here ---
    # result = answer(prompt, df)  # <- call your function(s)
    # For demo purposes:
    result = f"Echo: {prompt}. Tickets loaded: {len(df)}"
    return JSONResponse({"reply": result})
