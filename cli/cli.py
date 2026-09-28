import typer
import requests
from typing import Optional

app = typer.Typer()
API_URL = "http://127.0.0.1:8000"

@app.command()
def onboard(
    first_time: bool = typer.Option(True, help="Are you a first-time user?"),
    path: str = typer.Option("1-week", help="Path length: 1-week, 1-month, 3-month"),
    quant: str = typer.Option("Beginner", help="Quant level: Beginner, Advanced, Pro"),
    options: str = typer.Option("Beginner", help="Options level: Beginner, Advanced, Pro"),
):
    \"\"\"Onboard a new user to QuantMetron\"\"\"
    payload = {
        "first_time": first_time,
        "path_len": path,
        "quant_level": quant,
        "options_level": options,
    }
    try:
        response = requests.post(f"{API_URL}/onboard", json=payload)
        typer.echo(response.json().get("message", "Onboarding failed"))
    except requests.exceptions.ConnectionError:
        typer.echo("Error: API server not running. Start it with 'uvicorn app.main:app --reload'")

@app.command()
def daily(user_id: str):
    \"\"\"Get today's lesson and simulation\"\"\"
    try:
        response = requests.post(f"{API_URL}/daily", json={"user_id": user_id})
        data = response.json()
        typer.echo(f"Topic: {data.get('topic')}")
        typer.echo(f"Explanation: {data.get('explanation')}")
    except requests.exceptions.ConnectionError:
        typer.echo("Error: API server not running.")

@app.command()
def feedback(
    user_id: str, 
    day: int, 
    confidence: int, 
    note: Optional[str] = None, 
    pnl: Optional[float] = None
):
    \"\"\"Provide feedback for the day's lesson\"\"\"
    payload = {
        "user_id": user_id,
        "day": day,
        "confidence": confidence,
        "note": note,
        "paper_pnl": pnl,
    }
    try:
        response = requests.post(f"{API_URL}/feedback", json=payload)
        typer.echo(response.json().get("message", "Feedback failed"))
    except requests.exceptions.ConnectionError:
        typer.echo("Error: API server not running.")

if __name__ == "__main__":
    app()
