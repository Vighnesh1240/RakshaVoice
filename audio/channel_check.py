import glob, os, random
import numpy as np, soundfile as sf, librosa
from audio.spoof_stream import score_chunk

SR = 16000
rng = np.random.RandomState(0); random.seed(0)

def crop4(y):
    n = 4 * SR
    if len(y) < n: y = np.pad(y, (0, n - len(y)))
    s = rng.randint(0, len(y) - n + 1)
    return y[s:s + n]

def noisy(y, snr_db):
    return (y + rng.randn(len(y)) * np.std(y) / (10 ** (snr_db / 20))).astype(np.float32)

def phone(y):                                              # narrow-band (8 kHz) plus light noise
    y = librosa.resample(librosa.resample(y, orig_sr=SR, target_sr=8000), orig_sr=8000, target_sr=SR)
    return noisy(y, 25)

def room(y, rt=0.3):                                       # crude reverb: convolve with a decaying-noise impulse response
    n = int(rt * SR); ir = rng.randn(n) * np.exp(-6.9 * np.arange(n) / n); ir[0] = 1.0
    out = np.convolve(y, ir / np.sqrt(np.sum(ir ** 2)))[:len(y)]
    return (out / (np.max(np.abs(out)) + 1e-9) * np.max(np.abs(y))).astype(np.float32)

def report(name, scores):
    s = np.array(scores)
    print(f"{name:40s} n={len(s):3d}  mean={s.mean():.2f}  median={np.median(s):.2f}  share above 0.5 = {np.mean(s > 0.5) * 100:3.0f}%")

# 1) clean human speech that the model never saw (LibriSpeech test-clean)
libri = []
for d in sorted(glob.glob("voice/data/LibriSpeech/test-clean/*")):
    f = random.choice(glob.glob(f"{d}/*/*.flac")); y, sr = sf.read(f, dtype="float32")
    libri.append(crop4(y if sr == SR else librosa.resample(y, orig_sr=sr, target_sr=SR)))
report("LibriSpeech clean (human)", [score_chunk(y) for y in libri])
report("LibriSpeech + 8 kHz phone-like", [score_chunk(phone(y)) for y in libri])
report("LibriSpeech + noise 15 dB", [score_chunk(noisy(y, 15)) for y in libri])
report("LibriSpeech + room reverb", [score_chunk(room(y)) for y in libri])

# 2) your own recordings (same human, two microphone distances)
def own(pattern):
    out = []
    for f in sorted(glob.glob(pattern)):
        y = librosa.load(f, sr=SR, mono=True)[0]
        for i in range(0, max(1, len(y) - 4 * SR + 1), 2 * SR):
            w = y[i:i + 4 * SR]
            if np.sqrt(np.mean(w ** 2)) > 0.005: out.append(score_chunk(w))
    return out
report("Your voice, near mic (clean clips)", own("nlp/test_clips/*__clean.wav"))
report("Your voice, far (far clips)", own("nlp/test_clips/*__far.wav"))

# 3) optional: put any synthetic / text-to-speech WAV files in audio/test_fake/ to compare
fake = []
for f in sorted(glob.glob("audio/test_fake/*.wav")):
    y = librosa.load(f, sr=SR, mono=True)[0]; fake.append(score_chunk(crop4(y)))
if fake: report("Synthetic speech (audio/test_fake)", fake)

# 4) same person, by language of the line you read (en = English, hg = Hinglish, hi = Hindi)
for lang in ("en", "hg", "hi"):
    report(f"Your voice, lines in '{lang}'", own(f"nlp/test_clips/{lang}_*__clean.wav") + own(f"nlp/test_clips/{lang}_*__far.wav"))