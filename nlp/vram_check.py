import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def used():
    return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).decode().split()[0])


base = used(); print("baseline (desktop apps) MiB:", base)
from audio.spoof_stream import score_chunk;  score_chunk(np.zeros(64000, dtype=np.float32)); a = used(); print("after spoof model (wav2vec2):", a - base, "MiB")
from nlp.intent_stream import score_text;    score_text("hello there");                       b = used(); print("after intent model (MuRIL):  ", b - a, "MiB")
from nlp.asr import transcribe;              transcribe(np.zeros(64000, dtype=np.float32));   c = used(); print("after Whisper small:         ", c - b, "MiB")
print("TOTAL in use now:", c, "MiB of 4096")