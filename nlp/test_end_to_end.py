import os
import sys
import time
from pathlib import Path

import librosa

ROOT = Path(__file__).resolve().parents[1]
CLIPS = Path(__file__).resolve().parent / "test_clips"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nlp.asr import transcribe
from nlp.intent_stream import score_text

EXPECT = {"scam_otp": "otp_pii", "scam_police": "authority", "scam_pay": "payment"}
LANG = None                       # run once with None, once with "hi" and keep the better one
score_text("warm up"); transcribe(librosa.tone(440, sr=16000, duration=2))     # exclude start-up cost from timings

ok = n = 0
for f in sorted(CLIPS.glob("*.wav")):
    name = os.path.basename(f)
    y, _ = librosa.load(f, sr=16000, mono=True)
    t0 = time.time(); text = transcribe(y, language=LANG); t1 = time.time()
    r = score_text(text); t2 = time.time()
    exp = next((v for k, v in EXPECT.items() if k in name), None)       # None = normal call, nothing should fire
    got = r["evidence"]
    good = (exp in got) if exp else (len(got) == 0)
    n += 1; ok += good
    print(f"{'OK   ' if good else 'CHECK'} {name:26s} audio {len(y)/16000:4.1f}s asr {t1-t0:4.2f}s nlp {1000*(t2-t1):3.0f}ms "
          f"fired={got} | {text[:80]}")
    if not good: print("      labels:", r["labels"])
print(f"\n{ok}/{n} clips behaved as expected")