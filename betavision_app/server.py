"""
server.py - the local web server (FastAPI). Serves the single page in static/
and a small JSON API:

  POST /api/upload                    upload a video, then follow the camera in the background
  GET  /api/jobs/{id}/frame/{i}       frame i as JPEG for the picker (?w=240 for a thumbnail)
  POST /api/jobs/{id}/holds           picker state -> detected holds + volumes (preview)
  POST /api/jobs/{id}/start           start the analysis in the background
  DELETE /api/jobs/{id}               the user started over: free the job's memory and files
  GET  /api/jobs/{id}                 status / progress / results (the page polls this)
  GET  /api/jobs/{id}/video           annotated H.264 video (inline)
  GET  /api/jobs/{id}/download        the same video as a download
  GET  /api/jobs/{id}/csv             per-frame data
  POST /api/jobs/{id}/compare         upload a second climb of the route and compare it (background)
  GET  /api/jobs/{id}/compare/video   the side-by-side comparison video (H.264)
  GET  /api/jobs/{id}/compare/check   the route drawn on the second video (registration check)
"""

import re
import shutil
from pathlib import Path
from typing import Any

import cv2
from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import jobs

STATIC = Path(__file__).resolve().parent / "static"
MODELS = ("lite", "full", "heavy")

app = FastAPI(title="BetaVision", docs_url=None, redoc_url=None)
worker = jobs.Worker()


@app.exception_handler(jobs.PickerError)
def bad_picker(_: Request, e: jobs.PickerError):
    return JSONResponse({"detail": str(e)}, status_code=400)


@app.exception_handler(jobs.JobBusy)
def busy(_: Request, e: jobs.JobBusy):
    return JSONResponse({"detail": str(e)}, status_code=409)


def get_job(job_id):
    job = worker.get(job_id)
    if job is None:
        raise HTTPException(404, "That climb isn't here any more. The app may have restarted, so upload it again.")
    return job


class StartRequest(BaseModel):
    picker: dict | None = None      # None = skip holds, body tracking only
    model: str = "full"
    notes: str = ""                 # sticky notes, one per line (as in a --notes file)
    lock: bool = True               # fine-tune each hold per frame (CLI: unless --no-lock)
    off_route: bool = True          # flag possible off-route touches (CLI: unless --no-off-route)


@app.post("/api/upload")
def upload(file: UploadFile = File(...), still: bool = Form(False)):
    if Path(file.filename or "").suffix.lower() not in jobs.VIDEO_TYPES:
        raise HTTPException(400, "Please choose an .mp4, .mov or .avi video.")
    job = worker.new_job(file.filename)
    with open(job.input, "wb") as out:
        shutil.copyfileobj(file.file, out, 1024 * 1024)
    try:
        job.open(still)
    except ValueError as e:
        job.state = "error"
        worker.remove(job.id)                               # don't keep a job that never opened
        raise HTTPException(400, str(e))
    return job.status()


@app.get("/api/jobs/{job_id}/frame/{i}")
def frame(job_id: str, i: int, w: int | None = None):
    data = get_job(job_id).picture_jpeg(i, w)
    if data is None:
        raise HTTPException(404, "Frame not found")
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


@app.post("/api/jobs/{job_id}/holds")
def holds(job_id: str, picker: Any = Body(...)):
    job = get_job(job_id)
    if job.state in ("uploading", "preparing"):
        raise HTTPException(409, "Still reading the video. Try again in a moment.")
    return job.preview(picker)


@app.post("/api/jobs/{job_id}/start")
def start(job_id: str, req: StartRequest):
    job = get_job(job_id)
    if job.state not in ("picking", "error"):               # quick answer; Worker.start re-checks atomically
        raise HTTPException(409, "This climb is already being analyzed.")
    try:
        notes = jobs.parse_notes(req.notes)
    except ValueError as e:
        raise HTTPException(400, f"Sticky notes: {e}")
    picker = job.clean_picker(req.picker) if req.picker else None     # 400 with a readable message
    worker.start(job, picker, req.model if req.model in MODELS else "full", notes, req.lock, req.off_route)
    return job.status()


@app.delete("/api/jobs/{job_id}")
def delete(job_id: str):
    """The page's Start over: free the job (it may be gone already, which is fine)."""
    worker.remove(job_id)
    return {"ok": True}


@app.get("/api/jobs/{job_id}")
def status(job_id: str):
    return get_job(job_id).status()


def output(job_id, attr):
    job = get_job(job_id)
    if job.state != "done":
        raise HTTPException(409, "The analysis isn't finished yet.")
    return job, getattr(job, attr)


@app.get("/api/jobs/{job_id}/video")
def video(job_id: str):
    _, path = output(job_id, "output")
    return FileResponse(path, media_type="video/mp4")


@app.get("/api/jobs/{job_id}/download")
def download(job_id: str):
    job, path = output(job_id, "output")
    return FileResponse(path, media_type="video/mp4", filename=f"{job.name}_analysis.mp4")


@app.get("/api/jobs/{job_id}/csv")
def csv(job_id: str):
    job, path = output(job_id, "csv")
    return FileResponse(path, media_type="text/csv", filename=f"{job.name}_analysis.csv")


@app.post("/api/jobs/{job_id}/compare")
def compare(job_id: str, file: UploadFile = File(...)):
    job = get_job(job_id)
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in jobs.VIDEO_TYPES:
        raise HTTPException(400, "Please choose an .mp4, .mov or .avi video.")
    if job.state != "done":
        raise HTTPException(409, "Analyze this climb first, then compare.")
    if (job.compare or {}).get("state") in jobs.COMPARE_BUSY:
        raise HTTPException(409, "A comparison for this climb is already running.")
    if not getattr(job, "holds", None):
        raise HTTPException(409, "Mark the route's holds first (start over and don't skip them): the comparison "
                                 "uses the route to find where each climb starts and ends and which hold each "
                                 "hand and foot is on.")
    stem = re.sub(r"[^\w\-]+", "_", Path(file.filename).stem).strip("_") or "climb_b"
    if stem == job.name:
        stem += "_b"
    dest = job.dir / "compare" / f"{stem}{suffix}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out, 1024 * 1024)
    cap = cv2.VideoCapture(str(dest))
    ok = cap.isOpened() and cap.read()[0]
    cap.release()
    if not ok:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "Couldn't read that video. Try exporting it again as .mp4.")
    jobs.start_compare(job, dest)
    return job.status()


def compare_output(job_id, key):
    job = get_job(job_id)
    if (job.compare or {}).get("state") != "done" or key not in job.compare_files:
        raise HTTPException(409, "The comparison isn't finished yet.")
    return job.compare_files[key]


@app.get("/api/jobs/{job_id}/compare/video")
def compare_video(job_id: str):
    return FileResponse(compare_output(job_id, "video"), media_type="video/mp4")


@app.get("/api/jobs/{job_id}/compare/check")
def compare_check(job_id: str):
    return FileResponse(compare_output(job_id, "check"), media_type="image/jpeg")


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
