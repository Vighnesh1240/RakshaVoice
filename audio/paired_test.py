import asyncio, glob, os, random, re, shutil, subprocess, sys, time
import numpy as np, pandas as pd, soundfile as sf, librosa, sounddevice as sd
from sklearn.metrics import roc_auc_score

SR, SEC, PAD = 16000, 4, 0.5
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(REPO_ROOT, "voice", "data", "LibriSpeech", "test-clean")
OUT = os.path.join(REPO_ROOT, "audio", "data_v2", "paired")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from audio.spoof_stream import score_logit

random.seed(1); rng = np.random.RandomState(1)
os.makedirs(OUT, exist_ok=True)
assert shutil.which("ffmpeg"), "ffmpeg is needed (winget install Gyan.FFmpeg)"
if not os.path.isdir(ROOT):
    raise FileNotFoundError(
        f"LibriSpeech test-clean directory not found: {ROOT}. "
        "Place/extract the dataset there before running this test."
    )

def pick_sentences(n_per_speaker=5):
    out = []
    spk = sorted(os.listdir(ROOT))[32:]                                  # the 8 held-out speakers (same split as prep_v2)
    for s in spk:
        items = []
        for t in glob.glob(os.path.join(ROOT, s, "*", "*.trans.txt")):
            with open(t, encoding="utf-8") as transcript:
                lines = transcript
                for line in lines:
                    parts = line.strip().split(" ", 1)
                    if len(parts) != 2:
                        continue
                    uid, text = parts
                    if 8 <= len(text.split()) <= 25:
                        items.append((uid, text.capitalize(), os.path.join(os.path.dirname(t), uid + ".flac")))
        random.shuffle(items); out += [(s,) + it for it in items[:n_per_speaker]]
    return out

def voiced_crop(y):
    y, _ = librosa.effects.trim(y, top_db=30); n = SEC * SR
    if len(y) < int(0.75 * n): return None
    if len(y) < n: y = np.pad(y, (0, n - len(y)))
    s = rng.randint(0, len(y) - n + 1); y = y[s:s + n]
    return (y / (np.max(np.abs(y)) + 1e-9) * 0.5).astype(np.float32)

def mp3_roundtrip(y, name):
    p = f"{OUT}/{name}.wav"; sf.write(p, y, SR)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", p, "-ar", "24000", "-b:a", "48k", p + ".mp3"], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", p + ".mp3", "-ac", "1", "-ar", str(SR), p], check=True)
    os.remove(p + ".mp3"); return sf.read(p, dtype="float32")[0]

def tts_to_wav(mp3_path):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp3_path, "-ac", "1", "-ar", str(SR), mp3_path + ".wav"], check=True)
    y = sf.read(mp3_path + ".wav", dtype="float32")[0]; os.remove(mp3_path + ".wav"); os.remove(mp3_path); return y

async def edge_clip(text, voice, path):
    import edge_tts
    await edge_tts.Communicate(text, voice).save(path)

def replay(y):
    x = np.concatenate([np.zeros(int(PAD * SR)), y, np.zeros(int(PAD * SR))]).astype(np.float32).reshape(-1, 1)
    rec = sd.playrec(x, samplerate=SR, channels=1, dtype="float32"); sd.wait()
    return rec[int(PAD * SR): int(PAD * SR) + len(y), 0]

rows = []
sents = pick_sentences(); print(len(sents), "sentences from held-out speakers")
edge_loop = asyncio.new_event_loop()
asyncio.set_event_loop(edge_loop)
try:
    for k, (spk, uid, text, flac) in enumerate(sents):
        y = sf.read(flac, dtype="float32")[0]
        real = voiced_crop(y)
        if real is None: continue
        rows.append(("real", "librispeech", f"real_{uid}", mp3_roundtrip(real, f"real_{uid}")))
        from gtts import gTTS
        try:
            gTTS(text=text, lang="en", tld="co.in").save(f"{OUT}/g.mp3"); f = voiced_crop(tts_to_wav(f"{OUT}/g.mp3"))
            if f is not None: rows.append(("fake", "gtts", f"gtts_{uid}", mp3_roundtrip(f, f"gtts_{uid}")))
        except Exception as e: print("gTTS skip", uid, type(e).__name__)
        voice = ["en-US-AriaNeural", "en-GB-SoniaNeural"][k % 2]
        try:
            edge_loop.run_until_complete(edge_clip(text, voice, f"{OUT}/e.mp3"))
            f = voiced_crop(tts_to_wav(f"{OUT}/e.mp3"))
            if f is not None: rows.append(("fake", "edge_new_voice", f"edge_{uid}", mp3_roundtrip(f, f"edge_{uid}")))
        except Exception as e: print("edge skip", uid, type(e).__name__)
        time.sleep(0.5)
finally:
    edge_loop.close()
    asyncio.set_event_loop(None)
print(pd.Series([r[1] for r in rows]).value_counts().to_dict(), "clips; replaying them now (about", int(len(rows) * 5.6 / 60), "minutes)")

res = []
for cls, grp, name, y in rows:
    res.append({"cls": cls, "group": grp, "name": name, "direct": score_logit(y), "replay": score_logit(replay(y))})
df = pd.DataFrame(res)
df.to_csv(os.path.join(REPO_ROOT, "audio", "data_v2", "paired_scores.csv"), index=False)

for cond in ("direct", "replay"):
    real = df[df.cls == "real"][cond].values
    print(f"\n=== {cond} (raw logits; negative = human-like) ===")
    print(f"real            n={len(real):3d} mean={real.mean():6.1f}  median={np.median(real):6.1f}  95th pct={np.percentile(real, 95):6.1f}  99th pct={np.percentile(real, 99):6.1f}")
    for g in ("gtts", "edge_new_voice"):
        f = df[df.group == g][cond].values
        if len(f) < 3: continue
        auc = roc_auc_score(np.r_[np.zeros(len(real)), np.ones(len(f))], np.r_[real, f])
        print(f"fake {g:14s} n={len(f):3d} mean={f.mean():6.1f}  median={np.median(f):6.1f}  5th pct={np.percentile(f, 5):6.1f}  AUC vs real={auc:.3f}")
rr = df[df.cls == "real"]["replay"].values; thr = float(np.percentile(rr, 98))
print(f"\nCandidate threshold = 98th percentile of REAL replayed logits = {thr:.1f}")
for g in ("gtts", "edge_new_voice"):
    f = df[df.group == g]["replay"].values
    if len(f): print(f"  at that threshold: {g} detected {np.mean(f >= thr) * 100:3.0f}% | real flagged {np.mean(rr >= thr) * 100:3.0f}%")