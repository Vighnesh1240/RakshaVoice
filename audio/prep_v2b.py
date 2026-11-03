import glob, os, random, shutil, subprocess
import numpy as np, pandas as pd, soundfile as sf, librosa

SR, SEC = 16000, 4
OUT = "audio/data_v2/direct"
os.makedirs(OUT, exist_ok=True)
random.seed(3)
rng = np.random.RandomState(3)
USE_MP3 = shutil.which("ffmpeg") is not None
man = pd.read_csv("audio/data_v2/manifest_direct.csv")
have = set(man.id)


def voiced_crop(y, sec=SEC):
    y, _ = librosa.effects.trim(y, top_db=30)
    n = int(sec * SR)
    if len(y) < int(0.75 * n):
        return None
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    s = rng.randint(0, len(y) - n + 1)
    y = y[s : s + n]
    return (y / (np.max(np.abs(y)) + 1e-9) * 0.5).astype(np.float32)


def read16k(path):
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    return y if sr == SR else librosa.resample(y, orig_sr=sr, target_sr=SR)


def mp3_roundtrip(path):
    tmp = path + ".mp3"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            path,
            "-ar",
            "24000",
            "-b:a",
            "48k",
            tmp,
        ],
        check=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-ac", "1", "-ar", str(SR), path],
        check=True,
    )
    os.remove(tmp)


rows = []


def add(cid, y, cls, source, split, lang, who):
    if cid in have:
        return
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


fm = pd.read_csv("audio/data_v2/manifest_fake_multi.csv")
for r in fm.itertuples():
    y = voiced_crop(read16k(r.path))
    if y is not None:
        add(
            f"fake_edgemulti_{os.path.basename(r.path)[:-4]}",
            y,
            "fake",
            "edge-multi",
            r.split,
            "english",
            r.voice,
        )

ROOT = "voice/data/LibriSpeech/test-clean"
spk = sorted(os.listdir(ROOT))
for i, s in enumerate(spk[:32]):
    split = "train" if i < 28 else "dev"
    files = glob.glob(f"{ROOT}/{s}/*/*.flac")
    random.shuffle(files)
    k = 0
    for f in files:
        y = voiced_crop(read16k(f))
        if y is not None:
            add(
                f"libri2_{s}_{k:02d}",
                y,
                "real",
                "librispeech",
                split,
                "english",
                s,
            )
            k += 1
        if k >= 4:
            break

new = pd.DataFrame(rows)
out = pd.concat([man, new], ignore_index=True)
out.to_csv("audio/data_v2/manifest_direct.csv", index=False)
print("MP3 equalisation:", "ON" if USE_MP3 else "OFF", "| added", len(new), "clips")
print(out.groupby(["cls", "source", "split"]).size().to_string())
