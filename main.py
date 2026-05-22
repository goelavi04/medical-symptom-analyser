# main.py
# FastAPI backend — receives symptoms from frontend, returns AI analysis

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag import analyse_symptoms
# Import our RAG pipeline function from rag.py

app = FastAPI()

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── REQUEST MODEL ─────────────────────────────────────────────────────────
class SymptomRequest(BaseModel):
    symptoms: str
# BaseModel = Pydantic model — defines the shape of incoming JSON
# When frontend sends {"symptoms": "I have a fever"}, FastAPI validates it automatically

# ── SERVE FRONTEND ────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")

# ── MAIN ENDPOINT ─────────────────────────────────────────────────────────
@app.post("/analyse")
async def analyse(request: SymptomRequest):
    """
    Receives symptom text from the frontend.
    Runs it through the RAG pipeline.
    Returns the AI analysis as JSON.
    """
    result = analyse_symptoms(request.symptoms)
    return result