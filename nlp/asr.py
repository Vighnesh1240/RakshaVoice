import os
import re
import sys
from pathlib import Path

import numpy as np

_DLL_DIRECTORIES = []
if os.name == "nt":
    nvidia_dir = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    for package_dir in ("cublas", "cuda_nvrtc", "cudnn"):
        dll_dir = nvidia_dir / package_dir / "bin"
        if dll_dir.is_dir():
            _DLL_DIRECTORIES.append(os.add_dll_directory(str(dll_dir)))
            os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")

from faster_whisper import WhisperModel

MODEL = "Hub84/faster-whisper-hinglish-prime"
try:
    _m = WhisperModel(MODEL, device="cuda", compute_type="int8_float16")
    ASR_DEVICE = "cuda"
except Exception as e:
    print(f"Whisper CUDA initialization failed: {e}")
    _m = WhisperModel(MODEL, device="cpu", compute_type="int8")
    ASR_DEVICE = "cpu"

print(f"RakshaVoice ASR device: {ASR_DEVICE}")

# Chosen with vad_test.py: 0.3 keeps quiet far-field speech, and silence / hiss still give empty text.
# (With the filter fully off, Whisper invented the text 'nan' on hiss.)
VAD = dict(threshold=0.3, min_silence_duration_ms=500)

# Small repair list for key terms Whisper sometimes mishears or spells out. Extend it as you find new cases.
_FIXES = [
    (re.compile(r"\bO\.?\s?T\.?\s?P(?:\.|(?=\b))", re.I), "OTP"),  # "O T P", "o.t.p."
    (re.compile(r"\bO[DB]P(?:\.|(?=\b))", re.I), "OTP"),             # "ODP", "OBP" (heard once on a far-field clip)
    (re.compile(r"\bU\.?\s?P\.?\s?I(?:\.|(?=\b))", re.I), "UPI"),
    (re.compile(r"\bC\.?\s?V\.?\s?V(?:\.|(?=\b))", re.I), "CVV"),
    (re.compile(r"\bK\.?\s?Y\.?\s?C(?:\.|(?=\b))", re.I), "KYC"),
    (re.compile(r"\bC\.?\s?B\.?\s?I(?:\.|(?=\b))", re.I), "CBI"),
]


def _repair(text):
    for pat, rep in _FIXES:
        text = pat.sub(rep, text)
    return text


def _run(wav, language):
    segs, info = _m.transcribe(wav, language=language, beam_size=1, vad_filter=True, vad_parameters=VAD,
                               temperature=0.0, condition_on_previous_text=False)
    return " ".join(s.text.strip() for s in segs).strip(), info.language


def transcribe(wav, language=None):
    """wav: float32 numpy array, 16 kHz mono. Returns the transcript text."""
    wav = np.asarray(wav, dtype=np.float32)
    text, lang = _run(wav, language)
    if language is None and lang not in ("en", "hi"):          # guard against odd language guesses (e.g. Urdu script)
        text, _ = _run(wav, "hi")
    return _repair(text)