\# TransitGuard AI



Agentic autonomous emergency response system - Far Away 2026 Hackathon



\## Current Progress

\- CV pipeline: OpenCV + YOLOv8 reads CCTV footage frame-by-frame

\- Detects vehicles (car, truck, bus, motorcycle) and pedestrians

\- Tracks objects across frames to identify accident events

\- Tuned to filter false positives



\## Files

\- cv\_pipeline.py - main entry point, runs YOLO on video

\- collision\_detection.py - bounding box overlap (IoU) logic

\- accident\_detector.py - frame-to-frame tracking + accident detection



\## Next Steps

\- Severity scoring

\- LangChain dispatcher agent

\- FastAPI backend + Supabase

\- React dashboard

