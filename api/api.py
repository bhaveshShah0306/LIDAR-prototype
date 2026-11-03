"""Backend for the dashboard: serves rides, per-frame data, events and a real-time replay.

You know REST APIs from .NET; FastAPI is the Python equivalent of a minimal-API controller:
a decorated function per route, type hints instead of attributes, JSON returned automatically.

Run (from the project root):
    uv add fastapi "uvicorn[standard]" duckdb
    uv run uvicorn api.api:app --reload
Then open http://127.0.0.1:8000/docs  (Swagger UI, like in .NET)

Done when: /docs lists every route, each returns data for one ride, and the WebSocket streams a
whole ride at real-time speed (check it with a small Python client using the `websockets` package).

Python you'll practise: decorators, type hints, async/await, path/query parameters, WebSockets.
"""
import asyncio
import json
from pathlib import Path

import duckdb
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

RIDES = Path(__file__).resolve().parent.parent / "data" / "rides"

app = FastAPI(title="Rear Alert API")
app.add_middleware(  # lets the React dev server (Vite, port 5173) call this API
    CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["*"], allow_headers=["*"],
)


def ride_dir(ride: str) -> Path:
    """Folder of one ride, or a 404 if it doesn't exist."""
    path = RIDES / ride
    if not (path / "meta.json").exists():
        raise HTTPException(status_code=404, detail=f"No ride {ride}")
    return path


@app.get("/rides")
def list_rides() -> list[dict]:
    """Every ride's meta.json, newest first.

    TODO: for each folder in RIDES with a meta.json, json.loads it; sort by "ride" descending; return.
    """
    raise NotImplementedError("TODO: list_rides")


@app.get("/rides/{ride}")
def get_ride(ride: str) -> dict:
    """meta.json plus a small summary (tracks per class, how often each light was on).

    TODO:
      1. meta = json.loads((ride_dir(ride) / "meta.json").read_text())
      2. Use duckdb.sql(f"SELECT ... FROM '{path}'").df() on video_tracks.parquet for tracks per class.
      3. Return {**meta, "tracks_per_class": {...}}
    """
    raise NotImplementedError("TODO: get_ride")


@app.get("/rides/{ride}/frames")
def get_frames(ride: str, start: float = 0.0, end: float = 10.0) -> list[dict]:
    """Objects between `start` and `end` seconds: what the 3D view draws.

    Query parameters come straight from the function arguments: /rides/x/frames?start=5&end=8

    TODO:
      1. Read video_geometry.parquet with DuckDB, WHERE t BETWEEN start AND end.
         Use a parameter instead of pasting numbers into the SQL string:
           duckdb.execute("SELECT ... FROM read_parquet(?) WHERE t BETWEEN ? AND ?", [str(path), start, end])
      2. Return .df().to_dict(orient="records")

    Check yourself: why use ? parameters instead of an f-string with start/end? (Same reason as in SQL Server.)
    """
    raise NotImplementedError("TODO: get_frames")


@app.get("/rides/{ride}/events")
def get_events(ride: str) -> list[dict]:
    """Moments where a light came on: start time, end time, side, max level (for the event list).

    TODO: read video_risk.parquet (or video_rules.parquet until risk.py is done) and merge
          consecutive frames with a light on into one event per side.
    """
    raise NotImplementedError("TODO: get_events")


@app.get("/rides/{ride}/video")
def get_video(ride: str):
    """The ride video for the dashboard's <video> element."""
    return FileResponse(ride_dir(ride) / "video.mp4", media_type="video/mp4")


@app.websocket("/rides/{ride}/replay")
async def replay(ws: WebSocket, ride: str):
    """Stream the ride frame by frame at real-time speed: objects + both light levels.

    TODO:
      1. await ws.accept()
      2. Load geometry + light levels for the ride (DuckDB or pandas), grouped by frame.
      3. For each frame in order:
           await ws.send_json({"t": ..., "objects": [...], "left": ..., "right": ...})
           await asyncio.sleep(time until the next frame)   # 0.1 s at 10 fps
      4. Close when done: await ws.close()

    Check yourself: why `await asyncio.sleep` and not `time.sleep` inside an async function?
    """
    raise NotImplementedError("TODO: replay")
