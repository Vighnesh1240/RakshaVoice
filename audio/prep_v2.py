import glob
import os
import random
import re
import shutil
import subprocess

import librosa
import numpy as np
import pandas as pd
import soundfile as sf


SR, SEC = 16000, 4
OUT = "audio/data_v2/direct"
os.makedirs(OUT, exist_ok=True)
random.seed(0)
rng = np.random.RandomState(0)
UTT_PER_SPEAKER = 6
USE_MP3 = shutil.which("ffmpeg") is not None


def voiced_crop(y, sec=SEC):
    y, _ = librosa.effects.trim(y, top_db=30)
    n = int(sec * SR)
    if len(y) < int(0.75 * n):
        return None
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    s = rng.randint(0, len(y) - n + 1)
    y = y[s:s + n]
    return (y / (np.max(np.abs(y)) + 1e-9) * 0.5).astype(np.float32)


def mp3_roundtrip(path):
    tmp = path + ".mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-ar", "24000",
         "-b:a", "48k", tmp],
        check=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-ac", "1",
         "-ar", str(SR), path],
        check=True,
    )
    os.remove(tmp)


def read16k(path):
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    return y if sr == SR else librosa.resample(y, orig_sr=sr, target_sr=SR)


rows = []


def add(cid, y, cls, source, split, lang, who):
    p = f"{OUT}/{cid}.wav"
    sf.write(p, y, SR)
    if USE_MP3:
        mp3_roundtrip(p)
    rows.append(
        {
            "id": cid,
            "path": p,
            "cls": cls,
            "source": source,
            "split": split,
            "lang": lang,
            "who": who,
            "condition": "direct",
        }
    )


# Real: LibriSpeech speakers (28 train / 4 dev / 8 test).
spk = sorted(os.listdir("voice/data/LibriSpeech/test-clean"))
for i, s in enumerate(spk):
    split = "train" if i < 28 else "dev" if i < 32 else "test"
    files = glob.glob(f"voice/data/LibriSpeech/test-clean/{s}/*/*.flac")
    random.shuffle(files)
    k = 0
    for f in files:
        y = voiced_crop(read16k(f))
        if y is not None:
            add(f"libri_{s}_{k:02d}", y, "real", "librispeech", split, "english", s)
            k += 1
        if k >= UTT_PER_SPEAKER:
            break

# Real: own recordings, split by line so each sentence stays in one split.
lines = sorted(
    {
        re.sub(r"__.*", "", os.path.basename(f))
        for f in glob.glob("nlp/test_clips/*__*.wav")
    }
)
for li, line in enumerate(lines):
    split = "train" if li < 12 else "dev" if li < 14 else "test"
    for f in sorted(glob.glob(f"nlp/test_clips/{line}__*.wav")):
        y = voiced_crop(read16k(f), SEC)
        if y is not None:
            tag = os.path.basename(f).split("__")[1].replace(".wav", "")
            add(
                f"own_{line}_{tag}",
                y,
                "real",
                "own",
                split,
                {"en": "english", "hg": "hinglish", "hi": "hindi"}.get(
                    line[:2], "?"
                ),
                "me",
            )

# Fake: edge-tts is split for training; gTTS is held out as an unseen engine.
for engine, folder in (("edge-tts", "edge"), ("gtts", "gtts")):
    files = sorted(glob.glob(f"audio/data_v2/fake/{folder}/*.wav"))
    random.shuffle(files)
    for j, f in enumerate(files):
        y = voiced_crop(read16k(f))
        if y is None:
            continue
        frac = j / max(1, len(files))
        split = (
            "test_unseen_engine"
            if engine == "gtts"
            else "train"
            if frac < 0.7
            else "dev"
            if frac < 0.85
            else "test"
        )
        lang = re.search(r"_(english|hinglish|hindi)_", os.path.basename(f))
        add(
            f"fake_{engine}_{os.path.basename(f)[:-4]}",
            y,
            "fake",
            engine,
            split,
            lang.group(1) if lang else "?",
            os.path.basename(f).split("_")[0],
        )


df = pd.DataFrame(rows)
df.to_csv("audio/data_v2/manifest_direct.csv", index=False)
print(
    "MP3 codec equalisation:",
    "ON" if USE_MP3 else "OFF (ffmpeg not found; direct clips keep a codec "
    "difference between real and fake)",
)
print(df.groupby(["cls", "source", "split"]).size().to_string())
