"""
app.py — FastAPI backend for the Temporal Deepfake Detector.

Upload a video file and get back a REAL / FAKE verdict with a confidence score.
The trained model and face extractor are loaded once at startup and reused
across requests (inference is serialised with a lock since the model is shared).

Run
───
  uvicorn app:app --host 0.0.0.0 --port 8000
  # or simply:  python app.py

Endpoints
─────────
  GET  /            → minimal HTML upload page
  GET  /health      → service / model status
  POST /predict     → multipart upload (field name: "file") → JSON verdict
"""

from __future__ import annotations

import shutil
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

import config
from data_loader import FaceExtractor, _build_val_transform
from inference import crops_to_windows, load_face_crops, load_model, run_inference
from utils import get_device

# ── Allowed upload types ──────────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB

# Default sliding-window stride (frames). Mirrors the inference.py CLI default.
DEFAULT_STRIDE = 10

# ── Shared, lazily-initialised state ───────────────────────────────────────────
_STATE: dict = {
    "device": None,
    "model": None,
    "extractor": None,
    "transform": None,
}
# Serialise model inference: a single shared nn.Module is not thread-safe and
# requests are dispatched to a threadpool by Starlette (sync endpoint).
_INFER_LOCK = threading.Lock()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load the model + face extractor once when the server starts."""
    device = get_device()
    print("[App] Loading model …")
    model = load_model(config.BEST_MODEL_PATH, device)
    print("[App] Initialising face extractor …")
    extractor = FaceExtractor(device=str(device))

    _STATE.update(
        device=device,
        model=model,
        extractor=extractor,
        transform=_build_val_transform(),
    )
    print("[App] Ready ✅")
    yield
    _STATE.clear()


app = FastAPI(
    title="Deepfake Detector API",
    description="Upload a video to classify it as REAL or FAKE.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the React dev server (and other local origins) to call the API directly.
# In dev, Vite also proxies /predict and /health to this server, so CORS is a
# belt-and-suspenders measure for when the frontend is served from another host.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Core prediction (blocking / CPU-bound) ──────────────────────────────────────

def _predict_video(video_path: Path, threshold: float, stride: int) -> dict:
    """Run the full inference pipeline on a saved video file."""
    extractor = _STATE["extractor"]
    model = _STATE["model"]
    device = _STATE["device"]
    transform = _STATE["transform"]

    crops = load_face_crops(str(video_path), extractor)
    if not crops:
        raise HTTPException(
            status_code=422,
            detail="No faces detected in the video. Cannot classify.",
        )

    windows = crops_to_windows(crops, config.SEQ_LEN, stride, transform)

    # The shared model is not thread-safe — serialise the forward pass.
    with _INFER_LOCK:
        mean_prob, all_probs = run_inference(model, windows, device)

    is_fake = mean_prob >= threshold
    confidence = mean_prob if is_fake else 1.0 - mean_prob

    return {
        "verdict": "FAKE" if is_fake else "REAL",
        "is_fake": bool(is_fake),
        "fake_probability": round(float(mean_prob), 4),
        "confidence": round(float(confidence), 4),
        "threshold": threshold,
        "faces_detected": len(crops),
        "windows_analysed": len(all_probs),
        "per_window_min": round(float(min(all_probs)), 4) if all_probs else None,
        "per_window_max": round(float(max(all_probs)), 4) if all_probs else None,
    }


# ── Routes ──────────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    """Report whether the model is loaded and which checkpoint is in use."""
    return {
        "status": "ok" if _STATE.get("model") is not None else "loading",
        "device": str(_STATE.get("device")),
        "checkpoint": str(config.BEST_MODEL_PATH),
        "checkpoint_exists": config.BEST_MODEL_PATH.exists(),
        "seq_len": config.SEQ_LEN,
    }


@app.post("/predict")
def predict(
    file: UploadFile = File(...),
    threshold: float = config.INFER_THRESHOLD,
    stride: int = DEFAULT_STRIDE,
):
    """
    Accept a multipart video upload and return a REAL / FAKE verdict.

    Form fields:
      file       — the video file (required)
      threshold  — fake-probability cutoff (optional, default config.INFER_THRESHOLD)
      stride     — sliding-window stride in frames (optional, default 10)
    """
    if _STATE.get("model") is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. "
                   f"Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )
    if not 0.0 < threshold < 1.0:
        raise HTTPException(status_code=400, detail="threshold must be in (0, 1).")
    if stride < 1:
        raise HTTPException(status_code=400, detail="stride must be >= 1.")

    # Stream the upload to a temp file (enforcing the size cap as we go).
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = Path(tmp.name)
            written = 0
            while chunk := file.file.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
                    )
                tmp.write(chunk)

        if written == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        result = _predict_video(tmp_path, threshold, stride)
        result["filename"] = file.filename
        return JSONResponse(result)

    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 — surface unexpected failures as 500
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
    finally:
        file.file.close()
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """A minimal single-file upload page for quick manual testing."""
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Deepfake Detector</title>
  <style>
    :root { color-scheme: dark; }
    body { font-family: system-ui, sans-serif; max-width: 640px; margin: 3rem auto;
           padding: 0 1rem; background: #0f1115; color: #e6e8eb; }
    h1 { font-size: 1.5rem; }
    .card { background: #1a1d24; border: 1px solid #2a2e37; border-radius: 12px;
            padding: 1.5rem; }
    input[type=file] { width: 100%; margin: 1rem 0; }
    button { background: #4f8ef7; color: #fff; border: 0; border-radius: 8px;
             padding: .7rem 1.4rem; font-size: 1rem; cursor: pointer; }
    button:disabled { opacity: .5; cursor: not-allowed; }
    #out { margin-top: 1.5rem; white-space: pre-wrap; font-family: ui-monospace, monospace; }
    .verdict { font-size: 1.6rem; font-weight: 700; }
    .fake { color: #ff6b6b; } .real { color: #51cf66; }
  </style>
</head>
<body>
  <h1>🎭 Deepfake Detector</h1>
  <div class="card">
    <p>Upload a video to check whether it is <b>REAL</b> or <b>FAKE</b>.</p>
    <input id="file" type="file" accept="video/*"/>
    <button id="go">Analyse</button>
    <div id="out"></div>
  </div>
  <script>
    const btn = document.getElementById('go');
    const out = document.getElementById('out');
    btn.onclick = async () => {
      const f = document.getElementById('file').files[0];
      if (!f) { out.textContent = 'Please choose a video file first.'; return; }
      const fd = new FormData(); fd.append('file', f);
      btn.disabled = true; out.textContent = 'Analysing… (this can take a while)';
      try {
        const r = await fetch('/predict', { method: 'POST', body: fd });
        const j = await r.json();
        if (!r.ok) { out.textContent = 'Error: ' + (j.detail || r.statusText); return; }
        const cls = j.is_fake ? 'fake' : 'real';
        out.innerHTML =
          `<div class="verdict ${cls}">${j.is_fake ? '🚨 FAKE' : '✅ REAL'} ` +
          `(${(j.confidence * 100).toFixed(1)}% confidence)</div>` +
          `<pre>${JSON.stringify(j, null, 2)}</pre>`;
      } catch (e) {
        out.textContent = 'Request failed: ' + e;
      } finally { btn.disabled = false; }
    };
  </script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
