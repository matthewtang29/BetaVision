"""
server.py - the local web server (FastAPI). Serves the single page in static/
and a small JSON API:

  POST   /api/upload                    upload a video, then follow the camera in the background
  GET    /api/jobs/{id}/frame/{i}       frame i as JPEG for the picker (?w=240 for a thumbnail)
  POST   /api/jobs/{id}/holds           picker state -> detected holds + volumes (preview)
  POST   /api/jobs/{id}/start           start the analysis in the background
  GET    /api/jobs/{id}                 status / progress / results (the page polls this)
  DELETE /api/jobs/{id}                 delete the video and everything made from it
  GET    /api/jobs/{id}/video           annotated H.264 video (inline)
  GET    /api/jobs/{id}/download        the same video as a download
  GET    /api/jobs/{id}/csv             per-frame data

Only this computer's own pages may use the API: requests from other websites
(Origin) or other host names (Host, against DNS rebinding) are refused.
"""

import shutil
import traceback
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, File, Form, HTTPException, Path as PathParam, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from . import jobs

STATIC = Path(__file__).resolve().parent / "static"
MODELS = ("lite", "full", "heavy")
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
allowed_hosts = set(LOCAL_HOSTS)     # __main__ adds --host if you serve on another address

app = FastAPI(title="BetaVision", docs_url=None, redoc_url=None, openapi_url=None)
worker = jobs.Worker()


def allow_host(host):
    host = host.strip("[]").lower()
    allowed_hosts.add("*" if host in ("0.0.0.0", "::") else host)   # all addresses: any host name


def _hostname(value):
    try:
        return (urlsplit(f"//{value}").hostname or "").lower()
    except ValueError:
        return ""


@app.middleware("http")
async def only_this_computer(request: Request, call_next):
    """A web page you visit elsewhere could otherwise send videos to the app while it runs.
    Host must be this computer (blocks DNS rebinding); Origin, if sent, must be this page itself."""
    host = request.headers.get("host", "")
    if "*" not in allowed_hosts and _hostname(host) not in allowed_hosts:
        return JSONResponse({"detail": "Unknown host name."}, status_code=400)
    origin = request.headers.get("origin")
    cross = request.headers.get("sec-fetch-site") == "cross-site"
    if (origin and urlsplit(origin).netloc.lower() != host.lower()) or (cross and request.method not in ("GET", "HEAD")):
        return JSONResponse({"detail": "Requests from other websites aren't allowed."}, status_code=403)
    return await call_next(request)


def get_job(job_id):
    job = worker.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "That climb isn't here any more. The app may have restarted, so upload it again.")
    return job


# ------------------------------------------------------------------ the picker's state (see jobs.Job.build_route)
Coord = Annotated[int, Field(ge=-100_000, le=100_000)]
FrameNo = Annotated[int, Field(ge=0, le=10_000_000)]
Point = Annotated[list[Coord], Field(min_length=2, max_length=3)]     # [x, y] or [x, y, frame]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Click(Strict):
    x: Coord
    y: Coord
    frame: FrameNo | None = None


class ManualVolume(Strict):
    kind: Literal["click", "poly"]
    x: Coord | None = None
    y: Coord | None = None
    points: Annotated[list[Annotated[list[Coord], Field(min_length=2, max_length=2)]],
                      Field(min_length=3, max_length=500)] | None = None
    frame: FrameNo | None = None


class PickerState(Strict):
    frame: FrameNo = 0
    frames: Annotated[list[FrameNo], Field(max_length=200)] | None = None
    tol: Annotated[int, Field(ge=1, le=100)] = 22
    min_size: Annotated[int, Field(ge=0, le=100)] = 3
    auto: Annotated[list[Click], Field(max_length=200)] = []
    manual: Annotated[list[Click], Field(max_length=500)] = []
    deleted: Annotated[list[Point], Field(max_length=1000)] = []
    starts: Annotated[list[Point], Field(max_length=2)] = []
    end: Point | None = None
    vols_manual: Annotated[list[ManualVolume], Field(max_length=200)] = []
    vols_deleted: Annotated[list[Point], Field(max_length=500)] = []


def picker_dict(job, picker):
    """The validated state as the plain dict build_route expects, after checking it fits this video."""
    st = picker.model_dump(exclude_none=True)
    if job.reader is None:
        raise HTTPException(409, "That video was closed. Upload it again.")
    n = job.reader.total
    frames = [st["frame"], *st.get("frames", [])]
    frames += [c["frame"] for key in ("auto", "manual", "vols_manual") for c in st[key] if "frame" in c]
    frames += [p[2] for key in ("deleted", "starts", "vols_deleted") for p in st[key] if len(p) > 2]
    if any(f >= n for f in frames):
        raise HTTPException(400, f"The video only has {n} frames.")
    for v in st["vols_manual"]:
        if (v["kind"] == "poly") != ("points" in v) or (v["kind"] == "click" and ("x" not in v or "y" not in v)):
            raise HTTPException(400, "A volume needs either a click (x, y) or its corners (points).")
    return st


class StartRequest(Strict):
    picker: PickerState | None = None   # None = skip holds, body tracking only
    model: Literal["lite", "full", "heavy"] = "full"
    notes: Annotated[str, Field(max_length=20_000)] = ""   # sticky notes, one per line (as in a --notes file)
    lock: bool = True                   # fine-tune each hold per frame (CLI: unless --no-lock)


# ------------------------------------------------------------------ routes
@app.post("/api/upload")
def upload(request: Request, file: UploadFile = File(...), still: bool = Form(False)):
    limit = jobs.MAX_UPLOAD_MB * 1024 * 1024
    if Path(file.filename or "").suffix.lower() not in jobs.VIDEO_TYPES:
        raise HTTPException(400, "Please choose an .mp4, .mov, .m4v or .avi video.")
    if (file.size or 0) > limit:
        raise HTTPException(413, too_big())
    job = worker.new_job(file.filename)
    try:
        written = 0
        with open(job.input, "wb") as out:
            while chunk := file.file.read(1024 * 1024):
                written += len(chunk)
                if written > limit:
                    raise HTTPException(413, too_big())
                out.write(chunk)
        job.open(still)
    except ValueError as e:
        worker.discard(job)
        raise HTTPException(400, str(e))
    except HTTPException:
        worker.discard(job)
        raise
    except Exception as e:
        traceback.print_exc()
        worker.discard(job)
        raise HTTPException(500, f"Couldn't open that video ({e}).")
    return job.status()


def too_big():
    return (f"That video is over {jobs.MAX_UPLOAD_MB / 1024:g} GB. Trim it to the climb, "
            "or export it at 1080p, and try again.")


@app.middleware("http")
async def refuse_huge_uploads(request: Request, call_next):
    """Turn away an oversized upload before it's received (the browser says how big it is)."""
    if request.url.path == "/api/upload":
        try:
            size = int(request.headers.get("content-length", 0))
        except ValueError:
            size = 0
        if size > jobs.MAX_UPLOAD_MB * 1024 * 1024 + 1024 * 1024:     # + room for the form's own fields
            return JSONResponse({"detail": too_big()}, status_code=413)
    return await call_next(request)


@app.get("/api/jobs/{job_id}/frame/{i}")
def frame(job_id: str, i: Annotated[int, PathParam(ge=0)], w: Annotated[int | None, Query(ge=16, le=4096)] = None):
    data = get_job(job_id).picture_jpeg(i, w)
    if data is None:
        raise HTTPException(404, "Frame not found")
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


@app.post("/api/jobs/{job_id}/holds")
def holds(job_id: str, picker: PickerState):
    job = get_job(job_id)
    if job.state != "picking":
        raise HTTPException(409, "This climb isn't ready for marking.")
    st = picker_dict(job, picker)
    try:
        return job.preview(st)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(400, f"Couldn't outline the route there ({e}).")


@app.post("/api/jobs/{job_id}/start")
def start(job_id: str, req: StartRequest):
    job = get_job(job_id)
    if job.state not in ("picking", "error"):
        raise HTTPException(409, "This climb is already being analyzed.")
    try:
        notes = jobs.parse_notes(req.notes)
    except ValueError as e:
        raise HTTPException(400, f"Sticky notes: {e}")
    st = picker_dict(job, req.picker) if req.picker else None
    try:
        worker.start(job, st, req.model, notes, req.lock)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(400, f"Couldn't prepare the route for analysis ({e}).")
    return job.status()


@app.get("/api/jobs/{job_id}")
def status(job_id: str):
    return get_job(job_id).status()


@app.delete("/api/jobs/{job_id}")
def delete(job_id: str):
    job = get_job(job_id)
    if job.state in ("queued", "running"):
        raise HTTPException(409, "Wait for the analysis to finish, then delete it.")
    worker.discard(job)
    return {"deleted": job_id}


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


@app.get("/api/limits")
def limits():
    """So the page can check a file's size before uploading it."""
    return {"max_upload_mb": jobs.MAX_UPLOAD_MB, "max_minutes": jobs.MAX_MINUTES, "video_types": sorted(jobs.VIDEO_TYPES)}


class Static(StaticFiles):
    """The page + a small 404 page; the service worker must never be cached stale."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        if path in ("sw.js", "index.html", ".", ""):
            response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/", Static(directory=STATIC, html=True), name="static")
