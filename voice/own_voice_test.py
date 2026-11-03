import glob, json, os, random, sys
import numpy as np, librosa, soundfile as sf
from sklearn.metrics import roc_curve

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice.voice_stream import embed

SR = 16000
rng = np.random.RandomState(0); random.seed(0)
cal = json.load(open("voice/calibration.json")); THR = cal["threshold"]

def windows(y, sec=4, hop=2):
    n = int(sec * SR)
    return [y[i:i + n] for i in range(0, max(1, len(y) - n + 1), int(hop * SR)) if np.sqrt(np.mean(y[i:i + n] ** 2)) > 0.005]

def clip_embs(path):
    y = librosa.load(path, sr=SR, mono=True)[0]
    return [embed(w) for w in windows(y)]

enrol_files = sorted(glob.glob("nlp/test_clips/*_norm_*__clean.wav"))
target = {"clean": sorted(glob.glob("nlp/test_clips/*_scam_*__clean.wav")),
          "far":   sorted(glob.glob("nlp/test_clips/*_scam_*__far.wav"))}
E = [e for f in enrol_files for e in clip_embs(f)]
c = np.mean(E, axis=0); c /= np.linalg.norm(c)
print("enrolment windows:", len(E), "from", len(enrol_files), "clips")

scores, labels = [], []
for cond, files in target.items():
    s = [float(np.dot(e, c)) for f in files for e in clip_embs(f)]
    print(f"{cond:6s} target windows={len(s):3d}  mean similarity={np.mean(s):.3f}  min={np.min(s):.3f}  below threshold={np.mean(np.array(s) < THR)*100:.1f}%")
    scores += s; labels += [1] * len(s)

imp = []
for d in sorted(glob.glob("voice/data/LibriSpeech/test-clean/*")):
    files = glob.glob(f"{d}/*/*.flac")
    for f in random.sample(files, 3):
        y, sr = sf.read(f, dtype="float32"); y = y if sr == SR else librosa.resample(y, orig_sr=sr, target_sr=SR)
        s0 = rng.randint(0, max(1, len(y) - 4 * SR)); imp.append(float(np.dot(embed(y[s0:s0 + 4 * SR]), c)))
print(f"impostor trials={len(imp)}  mean similarity={np.mean(imp):.3f}  max={np.max(imp):.3f}  above threshold={np.mean(np.array(imp) >= THR)*100:.1f}%")
scores += imp; labels += [0] * len(imp)

y, s = np.array(labels), np.array(scores)
fpr, tpr, _ = roc_curve(y, s); i = np.nanargmin(np.abs((1 - tpr) - fpr))
print(f"EER (you vs 40 LibriSpeech speakers) = {(fpr[i] + (1 - tpr[i])) / 2 * 100:.2f}%   threshold used = {THR:.3f}")