import asyncio, glob, os, random, subprocess, time
import numpy as np, pandas as pd, soundfile as sf, librosa

OUT = "audio/data_v2/fake"
SR = 16000
random.seed(0)
VOICES = {"english": ["en-IN-NeerjaNeural", "en-IN-PrabhatNeural"],
          "hinglish": ["en-IN-NeerjaNeural", "en-IN-PrabhatNeural"],      # romanized Hindi read by an Indian-English voice
          "hindi": ["hi-IN-SwaraNeural", "hi-IN-MadhurNeural"]}
N_PER_VOICE = {"english": 25, "hinglish": 25, "hindi": 30}                  # sentences per voice (edge-tts)
N_GTTS = {"english": 25, "hinglish": 25, "hindi": 30}                       # sentences per language (gTTS, kept as the unseen engine)


def pick_texts():
    df = pd.concat([pd.read_csv(f) for f in glob.glob("nlp/data/synthetic/batch*_*.csv")], ignore_index=True)
    df = df.drop_duplicates("text")
    df = df[df.text.str.split().str.len().between(6, 22)]                   # sentences of a natural length
    return {lang: df[df.lang == lang].text.tolist() for lang in VOICES}


def to_wav16k(src, dst):
    try:
        y, sr = sf.read(src, dtype="float32", always_2d=True); y = y.mean(axis=1)
    except Exception:                                                        # soundfile build without MP3 support
        tmp = dst + ".tmp.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-ac", "1", "-ar", str(SR), tmp], check=True)
        y, sr = sf.read(tmp, dtype="float32"); os.remove(tmp)
    if sr != SR: y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    sf.write(dst, y, SR)


async def make_edge(texts, manifest):
    import edge_tts
    os.makedirs(f"{OUT}/edge", exist_ok=True)
    for lang, voices in VOICES.items():
        pool = texts[lang][:]; random.shuffle(pool)
        for vi, voice in enumerate(voices):
            for k, text in enumerate(pool[vi * N_PER_VOICE[lang]:(vi + 1) * N_PER_VOICE[lang]]):
                name = f"{voice}_{lang}_{k:03d}"
                try:
                    await edge_tts.Communicate(text, voice).save(f"{OUT}/edge/{name}.mp3")
                    to_wav16k(f"{OUT}/edge/{name}.mp3", f"{OUT}/edge/{name}.wav"); os.remove(f"{OUT}/edge/{name}.mp3")
                    manifest.append({"path": f"{OUT}/edge/{name}.wav", "engine": "edge-tts", "voice": voice, "lang": lang, "text": text})
                except Exception as e:
                    print("skip", name, type(e).__name__, e)
                await asyncio.sleep(0.2)


def make_gtts(texts, manifest):
    from gtts import gTTS
    os.makedirs(f"{OUT}/gtts", exist_ok=True)
    for lang in VOICES:
        pool = texts[lang][::-1][:N_GTTS[lang]]                              # take from the other end so texts differ from edge-tts
        code, tld = ("hi", "co.in") if lang == "hindi" else ("en", "co.in")
        for k, text in enumerate(pool):
            name = f"gtts_{lang}_{k:03d}"
            try:
                gTTS(text=text, lang=code, tld=tld).save(f"{OUT}/gtts/{name}.mp3")
                to_wav16k(f"{OUT}/gtts/{name}.mp3", f"{OUT}/gtts/{name}.wav"); os.remove(f"{OUT}/gtts/{name}.mp3")
                manifest.append({"path": f"{OUT}/gtts/{name}.wav", "engine": "gtts", "voice": f"{code}-{tld}", "lang": lang, "text": text})
            except Exception as e:
                print("skip", name, type(e).__name__, e)
            time.sleep(0.5)


if __name__ == "__main__":
    texts = pick_texts()
    print({k: len(v) for k, v in texts.items()}, "texts available")
    manifest = []
    asyncio.run(make_edge(texts, manifest))
    make_gtts(texts, manifest)
    pd.DataFrame(manifest).to_csv("audio/data_v2/manifest_fake.csv", index=False)
    print(len(manifest), "fake clips written; manifest saved")