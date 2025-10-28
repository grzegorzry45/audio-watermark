import io, os, hashlib, subprocess, tempfile
from typing import Optional
import numpy as np
import soundfile as sf
import librosa
from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# -------- Watermark core: simple DSSS-like in STFT --------

def _text_to_seed(text: str) -> int:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")

def _pn_sequence(length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # +/-1 sequence
    seq = rng.choice([-1.0, 1.0], size=length)
    return seq.astype(np.float32)

def embed_watermark(x: np.ndarray, sr: int, message: str) -> np.ndarray:
    # Normalize & resample to 48k if needed
    target_sr = 48000
    if sr != target_sr:
        x = librosa.resample(x, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    # mono
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    # STFT params
    n_fft = 1024
    hop = 256
    S = librosa.stft(x, n_fft=n_fft, hop_length=hop, window="hann")
    mag, phase = np.abs(S), np.angle(S)

    freqs = np.linspace(0, sr/2, mag.shape[0])
    # Use bins roughly 1000–4000 Hz (robust against lossy encoders)
    band = (freqs >= 1000) & (freqs <= 4000)
    band_idxs = np.where(band)[0]
    if band_idxs.size < 10:
        return x  # nothing to do

    # PN per frame (repeated) seeded by message
    seed = _text_to_seed(message)
    pn = _pn_sequence(mag.shape[1], seed)

    # Adaptive injection level based on local energy
    # alpha determines watermark strength (lower = more transparent)
    base_alpha = 0.015  # conservative; adjust if needed
    # Modulate magnitudes in the selected band
    for t in range(mag.shape[1]):
        frame = mag[:, t]
        band_energy = np.sqrt(np.mean(frame[band_idxs]**2) + 1e-9)
        alpha = base_alpha * (0.5 + 0.5 * min(1.0, band_energy / 0.1))
        frame[band_idxs] *= (1.0 + alpha * pn[t])
        mag[:, t] = frame

    # Reconstruct
    S_mod = mag * np.exp(1j * phase)
    y = librosa.istft(S_mod, hop_length=hop, window="hann", length=len(x))
    # normalize to prevent clipping
    mx = np.max(np.abs(y)) + 1e-9
    if mx > 1.0:
        y = y / mx
    return y.astype(np.float32), sr

def detect_watermark(x: np.ndarray, sr: int, message: str) -> (bool, float):
    target_sr = 48000
    if sr != target_sr:
        x = librosa.resample(x, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    n_fft = 1024
    hop = 256
    S = librosa.stft(x, n_fft=n_fft, hop_length=hop, window="hann")
    mag = np.abs(S)
    freqs = np.linspace(0, sr/2, mag.shape[0])
    band = (freqs >= 1000) & (freqs <= 4000)
    band_idxs = np.where(band)[0]
    if band_idxs.size < 10 or mag.shape[1] < 10:
        return False, 0.0
    # compute per-frame band energy
    band_series = mag[band_idxs, :].mean(axis=0)
    band_series = (band_series - np.mean(band_series)) / (np.std(band_series) + 1e-9)

    seed = _text_to_seed(message)
    pn = _pn_sequence(len(band_series), seed)
    pn = (pn - np.mean(pn)) / (np.std(pn) + 1e-9)

    # correlation score
    score = float(np.dot(band_series, pn) / len(pn))
    # A conservative threshold
    present = score > 0.02
    return present, score

# -------- Helpers for IO --------

def load_audio_from_upload(raw: bytes):
    data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if isinstance(data, np.ndarray):
        if data.ndim > 1:
            data = np.mean(data, axis=1)
    else:
        data = np.array(data, dtype=np.float32)
    return data.astype(np.float32), int(sr)

def wav_bytes(y: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    buf.seek(0)
    return buf.read()

def mp3_bytes(y: np.ndarray, sr: int, bitrate="256k") -> bytes:
    # Encode WAV via ffmpeg to MP3
    wav = wav_bytes(y, sr)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in:
        f_in.write(wav)
        in_path = f_in.name
    out_path = in_path.replace(".wav", ".mp3")
    try:
        cmd = ["ffmpeg", "-y", "-i", in_path, "-vn", "-ar", str(sr), "-b:a", bitrate, out_path]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        try:
            os.remove(in_path)
        except: pass
        try:
            os.remove(out_path)
        except: pass

# -------- Routes --------

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/embed")
async def embed_endpoint(file: UploadFile, message: str = Form(...), output_format: str = Form("wav")):
    raw = await file.read()
    x, sr = load_audio_from_upload(raw)
    y, new_sr = embed_watermark(x, sr, message=message)
    if output_format.lower() == "mp3":
        data = mp3_bytes(y, new_sr, bitrate="256k")
        filename = (file.filename or "audio").rsplit(".",1)[0] + "_wm.mp3"
        media = "audio/mpeg"
    else:
        data = wav_bytes(y, new_sr)
        filename = (file.filename or "audio").rsplit(".",1)[0] + "_wm.wav"
        media = "audio/wav"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(io.BytesIO(data), media_type=media, headers=headers)

@app.post("/detect")
async def detect_endpoint(file: UploadFile, message: str = Form(...)):
    raw = await file.read()
    x, sr = load_audio_from_upload(raw)
    present, score = detect_watermark(x, sr, message=message)
    return JSONResponse({"present": bool(present), "score": float(score)})

# Health check
@app.get("/healthz")
def healthz():
    return {"ok": True}

# For local run
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=port)
