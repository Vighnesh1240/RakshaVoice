import asyncio, io, json, os, sys, time

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import librosa
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from backend.evidence_explainer import explain_evidence
from pydantic import BaseModel

from backend.challenge_response import (
    get_challenge,
    evaluate_challenge,
)

from audio.spoof_stream import score_chunk, score_logit, A as SPOOF_A, SHIFT as SPOOF_SHIFT
from nlp.asr import transcribe
from nlp.intent_stream import score_text
from voice.voice_stream import score_voice, list_enrolled
from backend.chunker import RingBuffer
from fusion.live import LiveFusion

class VerificationAnswerRequest(BaseModel):
    contact: str
    answer: str

SILENCE_RMS = 0.005                      # windows quieter than this are not sent to any model
NEUTRAL_INTENT = {"stream": "intent", "score": 0.0, "labels": {}, "evidence": []}

ALLOWED_EXTENSIONS = (".wav", ".ogg", ".oga", ".opus")
MAX_FILE_BYTES = 25 * 1024 * 1024        # upload limit
MAX_FILE_WINDOWS = 40                    # at most this many 4 s windows are scored per file
WIN, HOP = 64000, 32000                  # 4 s window, 2 s hop at 16 kHz
MAX_PENDING_WINDOWS = 8                   # bounded backlog for sequential live scoring

LOG_DIR = "logs/sessions"                # numbers only: no audio and no text are ever written here
os.makedirs(LOG_DIR, exist_ok=True)
VALID_LABELS = {"normal", "scam", "clone", "unknown"}

FUSION = LiveFusion()
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/contacts")
def contacts():
    return {"contacts": list_enrolled()}


# ---------------------------------------------------------------- file analysis (kept in memory, never saved to disk)
def _analyze_audio(data: bytes, filename: str) -> dict:
    try:
        y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read this audio file. Try a standard WAV file.")
    y = y.mean(axis=1)                                       # mono
    duration = len(y) / sr
    if duration < 0.5:
        raise HTTPException(status_code=400, detail="The audio is too short to analyze (under 0.5 seconds).")
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)

    if len(y) <= WIN:
        windows = [y]
    else:
        starts = list(range(0, len(y) - WIN + 1, HOP))
        if starts[-1] + WIN < len(y):
            starts.append(len(y) - WIN)                      # make sure the end of the file is covered
        windows = [y[s:s + WIN] for s in starts]
    voiced = [w for w in windows if float(np.sqrt(np.mean(w ** 2))) >= SILENCE_RMS]
    if not voiced:
        raise HTTPException(status_code=400, detail="No speech detected in this file.")
    if len(voiced) > MAX_FILE_WINDOWS:                       # keep long files fast: evenly spaced sample
        idx = np.linspace(0, len(voiced) - 1, MAX_FILE_WINDOWS).astype(int)
        voiced = [voiced[i] for i in idx]

    scores = [float(score_chunk(w)) for w in voiced]
    return {
        "filename": filename,
        "duration_seconds": round(duration, 2),
        "spoof_probability_percent": round(float(np.mean(scores)) * 100, 2),
        "sample_rate": int(sr),
        "windows_scored": len(scores),
        "max_spoof_probability_percent": round(float(np.max(scores)) * 100, 2),
    }


@app.post("/analyze-file")
async def analyze_file(file: UploadFile = File(...)):
    name = file.filename or "audio"
    if not name.lower().endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(status_code=400, detail="Please upload a WAV, OGG, OGA, or OPUS audio file.")
    data = await file.read()
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="File too large (limit 25 MB).")
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _analyze_audio, data, name)

@app.get("/verification/challenge/{contact}")
async def verification_challenge(contact: str):
    """
    Retrieve the registered verification question.
    """

    return get_challenge(contact)


@app.post("/verification/evaluate")
async def verification_evaluate(request: VerificationAnswerRequest):
    """
    Evaluate the caller's supplied answer.
    The answer is not returned in the response.
    """

    return evaluate_challenge(
        request.contact,
        request.answer,
    )

# ---------------------------------------------------------------- live pipeline
def _timed(fn, *args):
    t0 = time.time()
    out = fn(*args)
    return out, int((time.time() - t0) * 1000)


def _intent_pipeline(window):
    text, t_asr = _timed(transcribe, window)
    if not text.strip():
        return text, NEUTRAL_INTENT, t_asr, 0
    res, t_int = _timed(score_text, text)
    return text, res, t_asr, t_int


def _log(state, rec):
    rec = {"session": state["session"], "label": state["label"], "step": state["step"], "t": round(time.time(), 2), **rec}
    state["step"] += 1
    with open(f"{LOG_DIR}/{state['session']}.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


async def _process(ws, window, state):
    loop = asyncio.get_running_loop()
    t0 = time.time()
    try:
        if float(np.sqrt(np.mean(window ** 2))) < SILENCE_RMS:
            _log(state, {"silent": True})
            await ws.send_json({"t": time.time(), "silent": True})
            return
        contact = state["contact"]
        jobs = [loop.run_in_executor(None, _timed, score_logit, window),
                loop.run_in_executor(None, _intent_pipeline, window)]
        if contact:
            jobs.append(loop.run_in_executor(None, _timed, score_voice, window, contact))
        res = await asyncio.gather(*jobs)
        (logit, t_spoof), (text, intent, t_asr, t_int) = res[0], res[1]
        p_fake = 1.0 / (1.0 + np.exp(-SPOOF_A * (float(logit) - SPOOF_SHIFT)))
        streams = {"spoof": {"stream": "spoof", "score": round(float(p_fake), 4), "logit": round(float(logit), 2)}, "intent": intent}
        lat = {"spoof": t_spoof, "asr": t_asr, "intent": t_int}
        if contact:
            voice, t_voice = res[2]
            streams["voice"] = voice
            lat["voice"] = t_voice
        
        lab = intent.get("labels") or {}

        t_fusion = time.perf_counter()
        fusion = FUSION.score(
            state["fusion"],
            spoof_logit=float(logit),
            intent_labels=lab,
            voice=streams.get("voice"),
        )
        lat["fusion"] = round((time.perf_counter() - t_fusion) * 1000)

        streams["fusion"] = fusion

        # Phase 6: Evidence Explainer
        streams["explanation"] = explain_evidence(fusion)

        lat["total"] = round((time.time() - t0) * 1000)

        _log(state, {"silent": False, "spoof_logit": streams["spoof"]["logit"], "spoof_p": streams["spoof"]["score"],
                     "urgency": lab.get("urgency"), "authority": lab.get("authority"), "otp_pii": lab.get("otp_pii"), "payment": lab.get("payment"),
                     "voice_active": "voice" in streams, "voice_similarity": streams.get("voice", {}).get("similarity"),
                     "fusion_risk": fusion.get("risk"), "fusion_alert": fusion.get("alert"), "latency_ms": lat["total"]})
        await ws.send_json({"t": time.time(), "window_s": 4, "transcript": text, "streams": streams,
                            "latency_ms": lat, "dropped_windows": state["dropped"]})
    except Exception as e:                                   # never freeze the UI on a model error
        await ws.send_json({"type": "error", "detail": f"{type(e).__name__}: {e}"})
    except asyncio.CancelledError:
        raise


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    buf = RingBuffer()
    state = {"contact": None, "dropped": 0, "label": "unknown", "step": 0,
             "session": time.strftime("%Y%m%d_%H%M%S"), "fusion": FUSION.new_state()}
    window_queue = asyncio.Queue(maxsize=MAX_PENDING_WINDOWS)

    async def process_windows():
        while True:
            window = await window_queue.get()
            try:
                await _process(ws, window, state)
            finally:
                window_queue.task_done()

    worker = asyncio.create_task(process_windows())
    try:
        while True:
            m = await ws.receive()
            if m["type"] == "websocket.disconnect":
                break
            if m.get("bytes") is not None:
                window = buf.push(np.frombuffer(m["bytes"], dtype=np.float32))
                if window is None:
                    continue
                try:
                    window_queue.put_nowait(window)
                except asyncio.QueueFull:
                    state["dropped"] += 1
                    await ws.send_json({
                        "type": "error",
                        "detail": "Live analysis queue overflow; window dropped",
                        "dropped_windows": state["dropped"],
                    })
            elif m.get("text") is not None:
                try:
                    ctrl = json.loads(m["text"])
                except ValueError:
                    continue
                if ctrl.get("type") == "set_label":
                    lab = ctrl.get("label", "unknown")
                    state["label"] = lab if lab in VALID_LABELS else "unknown"
                    await ws.send_json({"type": "label_set", "label": state["label"]})
                if ctrl.get("type") == "select_contact":
                    name = ctrl.get("name") or None
                    if name is None or name in list_enrolled():
                        state["contact"] = name
                        await ws.send_json({"type": "contact_set", "name": name})
                    else:
                        await ws.send_json({"type": "error", "detail": f"'{name}' is not enrolled"})
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        buf.clear()