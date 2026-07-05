from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI


ROOT = Path(__file__).resolve().parent
CACHE_ROOT = ROOT / "cache"
FRAME_CACHE_ROOT = CACHE_ROOT / "frames"
MAX_UPLOAD_BYTES = 60 * 1024 * 1024
DEFAULT_MODEL = "gemini-3.1-pro"

FRAME_CACHE_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="PetWatch AI Backend")
app.mount("/cache", StaticFiles(directory=str(CACHE_ROOT)), name="cache")


def load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise HTTPException(
            status_code=500,
            detail=f"{name} is required for video frame extraction. Install ffmpeg first.",
        )


def run_command(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or "Command failed"
        raise HTTPException(status_code=400, detail=message[:800]) from exc
    return completed.stdout.strip()


def get_duration(video_path: Path) -> float:
    output = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
    )
    try:
        duration = float(output)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Unable to read video duration.") from exc

    if duration <= 0:
        raise HTTPException(status_code=400, detail="Uploaded video has no readable duration.")
    return duration


def create_frame_cache(filename: str | None) -> tuple[str, Path]:
    raw_stem = Path(filename or "upload").stem
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw_stem).strip(".-") or "upload"
    safe_stem = safe_stem[:48]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cache_id = f"{timestamp}-{safe_stem}-{uuid4().hex[:8]}"
    frame_dir = FRAME_CACHE_ROOT / cache_id
    frame_dir.mkdir(parents=True, exist_ok=False)
    return cache_id, frame_dir


def extract_frames(video_path: Path, frame_dir: Path) -> list[Path]:
    require_binary("ffmpeg")
    require_binary("ffprobe")

    duration = get_duration(video_path)
    sample_times = [
        max(0.1, duration * 0.2),
        max(0.1, duration * 0.5),
        max(0.1, duration * 0.8),
    ]

    frames: list[Path] = []
    for index, sample_time in enumerate(sample_times, start=1):
        frame_path = frame_dir / f"frame_{index}.jpg"
        run_command(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{min(sample_time, max(duration - 0.1, 0.1)):.3f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-vf",
                "scale=768:-2",
                "-q:v",
                "4",
                str(frame_path),
            ]
        )
        if not frame_path.exists() or frame_path.stat().st_size == 0:
            raise HTTPException(status_code=400, detail="Unable to extract frames from this video.")
        frames.append(frame_path)

    return frames


def frame_cache_response(cache_id: str, frames: list[Path]) -> dict[str, Any]:
    return {
        "cacheId": cache_id,
        "cachedFrameDir": str((FRAME_CACHE_ROOT / cache_id).resolve()),
        "frameFiles": [str(frame.resolve()) for frame in frames],
        "frameUrls": [f"/cache/frames/{cache_id}/{frame.name}" for frame in frames],
    }


def image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_prompt(pet_type: str, scenario: str) -> str:
    return f"""
You are PetWatch AI, an early-warning assistant for pet owners.
Analyze three still frames sampled from a short pet video. The pet type is {pet_type}.
The user-selected demo context is "{scenario}", but prioritize what is visible in the images.

Look only for visible signals: gait or posture asymmetry, guarded movement, resting breathing
appearance, activity level, repetitive licking/scratching, and obvious discomfort. Do not diagnose,
do not name diseases, and do not claim certainty from three frames.

Return only valid JSON with this exact shape:
{{
  "overall": "Normal | Monitor | Vet Recommended",
  "className": "risk-normal | risk-monitor | risk-vet",
  "summary": "One owner-friendly sentence.",
  "cards": [
    {{"title": "Gait Risk", "status": "Low | Monitor | High", "className": "risk-normal | risk-monitor | risk-vet", "text": "Short visible-signal explanation."}},
    {{"title": "Breathing Risk", "status": "Low | Monitor | High", "className": "risk-normal | risk-monitor | risk-vet", "text": "Short visible-signal explanation."}},
    {{"title": "Behavior Risk", "status": "Low | Monitor | High", "className": "risk-normal | risk-monitor | risk-vet", "text": "Short visible-signal explanation."}}
  ],
  "next": "One practical next step for the owner, including vet contact when appropriate."
}}
""".strip()


def call_poe(frames: list[Path], pet_type: str, scenario: str) -> str:
    api_key = os.getenv("POE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing POE_API_KEY environment variable.")

    timeout = float(os.getenv("POE_TIMEOUT_SECONDS", "45"))
    client = OpenAI(api_key=api_key, base_url="https://api.poe.com/v1", timeout=timeout)
    model = os.getenv("POE_MODEL", DEFAULT_MODEL)

    content: list[dict[str, Any]] = [{"type": "text", "text": build_prompt(pet_type, scenario)}]
    for index, frame in enumerate(frames, start=1):
        content.append({"type": "text", "text": f"Frame {index} of 3:"})
        content.append({"type": "image_url", "image_url": {"url": image_data_url(frame)}})

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=0.2,
            stream=False,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Poe API request failed: {exc}") from exc

    message = completion.choices[0].message.content
    if not message:
        raise HTTPException(status_code=502, detail="Poe returned an empty response.")
    return message


def extract_json(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    candidates = [stripped]

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fence:
        candidates.insert(0, fence.group(1))

    brace = re.search(r"\{.*\}", stripped, re.DOTALL)
    if brace:
        candidates.append(brace.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def normalize_class(class_name: str | None, status: str | None = None) -> str:
    value = (class_name or status or "").lower()
    if "vet" in value or "high" in value:
        return "risk-vet"
    if "monitor" in value or "medium" in value:
        return "risk-monitor"
    return "risk-normal"


def normalize_result(raw_text: str, pet_type: str, scenario: str, filename: str) -> dict[str, Any]:
    parsed = extract_json(raw_text)
    if not parsed:
        return {
            "pet": pet_type,
            "scenario": scenario,
            "source": filename,
            "overall": "Monitor",
            "className": "risk-monitor",
            "summary": "The model returned a free-form response instead of structured risk cards.",
            "cards": [
                {
                    "title": "Model Notes",
                    "status": "Monitor",
                    "className": "risk-monitor",
                    "text": raw_text[:700],
                }
            ],
            "next": "Review the model notes and contact a licensed veterinarian for urgent or persistent concerns.",
            "model": os.getenv("POE_MODEL", DEFAULT_MODEL),
        }

    overall = str(parsed.get("overall") or "Monitor")
    class_name = normalize_class(str(parsed.get("className") or ""), overall)
    cards = parsed.get("cards") if isinstance(parsed.get("cards"), list) else []
    normalized_cards = []

    for card in cards[:3]:
        if not isinstance(card, dict):
            continue
        status = str(card.get("status") or "Monitor")
        normalized_cards.append(
            {
                "title": str(card.get("title") or "Visible Signal"),
                "status": status,
                "className": normalize_class(str(card.get("className") or ""), status),
                "text": str(card.get("text") or "No explanation returned by the model."),
            }
        )

    if not normalized_cards:
        normalized_cards = [
            {
                "title": "Visible Signal",
                "status": overall,
                "className": class_name,
                "text": str(parsed.get("summary") or "The model did not return detailed risk cards."),
            }
        ]

    return {
        "pet": pet_type,
        "scenario": scenario,
        "source": filename,
        "overall": overall,
        "className": class_name,
        "summary": str(parsed.get("summary") or "Analysis complete."),
        "cards": normalized_cards,
        "next": str(parsed.get("next") or "Keep monitoring and contact a veterinarian if concerns persist."),
        "model": os.getenv("POE_MODEL", DEFAULT_MODEL),
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/analyze-video")
async def analyze_video(
    video: UploadFile = File(...),
    pet_type: str = Form("Dog"),
    scenario: str = Form("Normal daily check"),
) -> dict[str, Any]:
    load_dotenv()

    if not video.content_type or not video.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Please upload a valid video file.")

    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    cache_id, frame_dir = create_frame_cache(video.filename)
    with tempfile.TemporaryDirectory(prefix="petwatch_") as temp_root:
        temp_path = Path(temp_root)
        upload_path = temp_path / f"upload{suffix}"

        total = 0
        with upload_path.open("wb") as handle:
            while chunk := await video.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Video is too large. Keep uploads under 60 MB.")
                handle.write(chunk)

        frames = extract_frames(upload_path, frame_dir)
        raw_text = call_poe(frames, pet_type, scenario)
        result = normalize_result(raw_text, pet_type, scenario, video.filename or "uploaded video")
        result["frameCount"] = len(frames)
        result.update(frame_cache_response(cache_id, frames))
        return result
