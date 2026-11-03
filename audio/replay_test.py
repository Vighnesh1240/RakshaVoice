import glob
import random

import librosa
import numpy as np
import sounddevice as sd
from sklearn.metrics import roc_auc_score

from audio.spoof_stream import score_chunk

SR, N = 16000, 20
random.seed(0)


def load4(path):
    y = librosa.load(path, sr=SR, mono=True)[0]
    y = y[:4 * SR] if len(y) >= 4 * SR else np.pad(y, (0, 4 * SR - len(y)))
    return (y / (np.max(np.abs(y)) + 1e-9) * 0.5).astype(np.float32)


def replay(y):
    pad = np.zeros(SR // 2, dtype=np.float32)
    x = np.concatenate([pad, y, pad]).reshape(-1, 1)
    rec = sd.playrec(x, samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    return rec[:, 0][SR // 2: SR // 2 + 4 * SR]


real_paths = glob.glob("voice/data/LibriSpeech/test-clean/*/*/*.flac")
fake_paths = sorted(glob.glob("audio/data_v2/fake/edge/*.wav"))
if len(real_paths) < N or len(fake_paths) < N:
    raise RuntimeError(
        f"Need at least {N} real and fake clips; found "
        f"{len(real_paths)} real and {len(fake_paths)} fake"
    )

real = [load4(f) for f in random.sample(real_paths, N)]
fake = [load4(f) for f in random.sample(fake_paths, N)]
total = len(real) + len(fake)
print(
    "replaying",
    total,
    "clips through the speaker (about",
    int(total * 5.5),
    "seconds) ..."
)

res = {}
for name, clips in [("real", real), ("fake", fake)]:
    res[name + "_direct"] = [score_chunk(y) for y in clips]
    res[name + "_replay"] = [score_chunk(replay(y)) for y in clips]

for key, scores in res.items():
    scores = np.array(scores)
    print(
        f"{key:12s} n={len(scores)}  mean={scores.mean():.2f}  "
        f"median={np.median(scores):.2f}  "
        f"share above 0.5 = {np.mean(scores > 0.5) * 100:3.0f}%"
    )

for condition in ("direct", "replay"):
    labels = np.array([0] * N + [1] * N)
    scores = np.array(
        res["real_" + condition] + res["fake_" + condition]
    )
    print(
        f"AUC real-vs-fake, {condition:6s}: "
        f"{roc_auc_score(labels, scores):.3f}   "
        "(0.5 = scores carry no information)"
    )
