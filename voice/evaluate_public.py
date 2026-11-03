import glob, os, json, random, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np, soundfile as sf, librosa
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from voice.voice_stream import embed

DATA_ROOT = REPO_ROOT / "voice" / "data" / "LibriSpeech" / "test-clean"
SR, N_UTT, N_ENROL = 16000, 12, 3
random.seed(0); rng = np.random.RandomState(0)

def load_files():
    out = {}
    for d in sorted(glob.glob(str(DATA_ROOT / "*"))):
        files = sorted(glob.glob(f"{d}/*/*.flac")); random.shuffle(files)
        if len(files) >= N_UTT: out[os.path.basename(d)] = files[:N_UTT]
    return out

def read(f):
    y, sr = sf.read(f, dtype="float32")
    return y if sr == SR else librosa.resample(y, orig_sr=sr, target_sr=SR)

def crop(y, sec):
    n = int(sec * SR)
    if len(y) < n: y = np.pad(y, (0, n - len(y)))
    s = rng.randint(0, len(y) - n + 1)
    return y[s:s + n]

def phone(y, snr_db=20):                                   # narrow-band (8 kHz) + noise, a rough stand-in for a phone call
    y = librosa.resample(librosa.resample(y, orig_sr=SR, target_sr=8000), orig_sr=8000, target_sr=SR)
    return (y + rng.randn(len(y)) * np.std(y) / (10 ** (snr_db / 20))).astype(np.float32)

def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s); fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr)); return float((fpr[i] + fnr[i]) / 2)

files = load_files(); spk = list(files)
audio = {s: [read(f) for f in files[s]] for s in spk}
print(len(spk), "speakers")

def run(sec, phone_test):
    cent = {}
    for s in spk:                                          # enrol from clean 8 s crops
        e = np.stack([embed(crop(y, 8)) for y in audio[s][:N_ENROL]]); c = e.mean(0); cent[s] = c / np.linalg.norm(c)
    scores, labels = [], []
    for s in spk:
        for y in audio[s][N_ENROL:]:
            x = crop(y, sec); x = phone(x) if phone_test else x
            e = embed(x)
            for t in spk:
                scores.append(float(np.dot(e, cent[t]))); labels.append(int(t == s))
    return np.array(labels), np.array(scores)

results, keep = {}, None
for name, sec, ph in [("clean_2s", 2, False), ("clean_4s", 4, False), ("phone_2s", 2, True), ("phone_4s", 4, True)]:
    y, s = run(sec, ph)
    results[name] = {"EER": eer(y, s), "AUC": float(roc_auc_score(y, s)), "target_trials": int(y.sum()), "nontarget_trials": int((1 - y).sum())}
    print(f"{name:9s} EER={results[name]['EER']*100:5.2f}%  AUC={results[name]['AUC']:.4f}")
    if name == "phone_4s": keep = (y, s)

y, s = keep                                                 # calibrate on the most realistic condition
lr = LogisticRegression(class_weight="balanced").fit(s.reshape(-1, 1), y)
a, c = float(lr.coef_[0][0]), float(lr.intercept_[0])
fpr, tpr, thr = roc_curve(y, s); i = np.nanargmin(np.abs((1 - tpr) - fpr))
cal = {"a": a, "c": c, "threshold": float(thr[i])}
os.makedirs("voice/results", exist_ok=True)
json.dump(results, open("voice/results/public_eval.json", "w"), indent=2)
json.dump(cal, open("voice/calibration.json", "w"), indent=2)
print("calibration:", cal)