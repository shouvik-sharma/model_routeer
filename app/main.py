from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from agent.graph import QuantMetronAgent

app = FastAPI(title="QuantMetron API")
agent = QuantMetronAgent()

class OnboardRequest(BaseModel):
    first_time: bool
    path_len: str # "1-week", "1-month", "3-month"
    quant_level: str # "Beginner", "Advanced", "Pro"
    options_level: str # "Beginner", "Advanced", "Pro"

class DailyRequest(BaseModel):
    user_id: str

class FeedbackRequest(BaseModel):
    user_id: str
    day: int
    confidence: int
    note: Optional[str] = None
    paper_pnl: Optional[float] = None

@app.get("/")
async def root():
    return {"message": "QuantMetron API is running"}

@app.post("/onboard")
async def onboard(req: OnboardRequest):
    # Use a dummy user_id for now or generate one
    user_id = "user_123" 
    msg = agent.onboard_user(user_id, req.path_len, req.quant_level, req.options_level)
    return {"status": "success", "user_id": user_id, "message": msg}

@app.post("/daily")
async def daily(req: DailyRequest):
    lesson = agent.get_daily_lesson(req.user_id)
    if "error" in lesson:
        raise HTTPException(status_code=404, detail=lesson["error"])
    return lesson

@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    msg = agent.record_feedback(req.user_id, req.day, req.confidence, req.note, req.paper_pnl)
    return {"status": "success", "message": msg}
