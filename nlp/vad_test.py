import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np, librosa
from nlp import asr
from faster_whisper import WhisperModel
from jiwer import wer

REFERENCE = "Sir, I am calling from your bank. Please tell me the OTP you just received."

asr._m = WhisperModel(asr.MODEL, device="cpu", compute_type="int8")

rng = np.random.RandomState(0)
tests = {
    "silence":  np.zeros(64000, dtype=np.float32),
    "hiss":     (rng.randn(64000) * 0.01).astype(np.float32),
    "far clip": librosa.load(ROOT / "nlp" / "test_clips" / "en_scam_otp__far.wav", sr=16000)[0],
}
settings = {
    "production (0.3)": {
        "threshold": 0.3,
        "min_silence_duration_ms": 500,
    },
    "default (0.5)": {
        "threshold": 0.5,
        "min_silence_duration_ms": 500,
    },
    "lower (0.2)": {
        "threshold": 0.2,
        "min_silence_duration_ms": 500,
    },
    "filter off": None,
}
for sname, vp in settings.items():
    for tname, y in tests.items():
        kw = dict(vad_filter=False) if vp is None else dict(vad_filter=True, vad_parameters=vp)
        segs, _ = asr._m.transcribe(y, beam_size=1, temperature=0.0, condition_on_previous_text=False, **kw)
        text = " ".join(s.text.strip() for s in segs).strip()
        if tname == "far clip":
            error_rate = wer(REFERENCE.lower(), text.lower())
            print(
                f"{sname:14s} {tname:9s} | "
                f"WER={error_rate:.3f} | {text[:90]!r}"
            )
        else:
            print(f"{sname:14s} {tname:9s} | {text[:90]!r}")