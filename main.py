"""
TransitGuard AI - FastAPI Backend
Receives incident data from the CV pipeline and writes it to Supabase.
The React dashboard subscribes to Supabase Realtime to get live updates.
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client

# Load environment variables from .env
load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(title="TransitGuard AI Backend")

# Allow the React dashboard (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for hackathon - restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)


class IncidentPayload(BaseModel):
    """Shape of the data sent from cv_pipeline.py for each detected accident."""
    camera_id: str = "CAM_001"
    location: str = "Camera-01 / Main Road Junction"
    frame_number: int
    accident_type: str          # "vehicle-vehicle" or "vehicle-pedestrian"
    severity_score: int
    severity_level: str         # "MINOR" | "MAJOR" | "CRITICAL"
    response_level: str         # "LOW" | "MODERATE" | "MAJOR" | "CRITICAL"
    notify: list[str]           # e.g. ["Police", "Ambulance"]
    incident_summary: str


@app.get("/")
def root():
    return {"status": "TransitGuard AI backend running"}


@app.post("/incident")
def create_incident(payload: IncidentPayload):
    """
    Called by cv_pipeline.py whenever an accident is detected.
    Inserts the incident into Supabase, and creates a responder_status
    row for each authority that needs to be notified.
    """
    try:
        incident_data = payload.model_dump()
        result = supabase.table("incidents").insert(incident_data).execute()

        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to insert incident")

        incident_id = result.data[0]["id"]

        for authority in payload.notify:
            supabase.table("responder_status").insert({
                "incident_id": incident_id,
                "authority_type": authority,
                "status": "alerted",
            }).execute()

        return {
            "status": "success",
            "incident_id": incident_id,
            "notified": payload.notify,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/incidents")
def get_incidents(limit: int = 50):
    """Get recent incidents (for testing / dashboard initial load)."""
    result = (
        supabase.table("incidents")
        .select("*")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data


@app.patch("/responder_status/{status_id}")
def update_responder_status(status_id: str, status: str):
    """
    Update a responder's status (e.g. 'acknowledged', 'en_route', 'on_scene').
    """
    result = (
        supabase.table("responder_status")
        .update({"status": status})
        .eq("id", status_id)
        .execute()
    )
    return result.data


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)