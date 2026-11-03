import asyncio, glob, os, random, subprocess, time
import numpy as np, pandas as pd, soundfile as sf, librosa

ROOT = "voice/data/LibriSpeech/test-clean"
OUT = "audio/data_v2/fake/edge_multi"
HELD_OUT = {"en-US-AriaNeural", "en-GB-SoniaNeural"}
N_VOICES, N_PER_VOICE, SR = 10, 20, 16000
random.seed(2)
os.makedirs(OUT, exist_ok=True)


def sentences(speakers, n):
    items = []
    for s in speakers:
        for t in glob.glob(f"{ROOT}/{s}/*/*.trans.txt"):
            for line in open(t, encoding="utf-8"):
                uid, text = line.strip().split(" ", 1)
                if 8 <= len(text.split()) <= 25:
                    items.append(text.capitalize())
    random.shuffle(items)
    return items[:n]


async def run():
    import edge_tts

    allv = await edge_tts.list_voices()
    en = sorted(
        {
            v["ShortName"]
            for v in allv
            if v["Locale"].startswith("en-")
            and v["ShortName"] not in HELD_OUT
            and "Multilingual" not in v["ShortName"]
        }
    )
    random.shuffle(en)
    voices = en[:N_VOICES]
    print("available English voices:", len(en), "| using:", voices)
    spk = sorted(os.listdir(ROOT))
    train_s, dev_s = spk[:28], spk[28:32]
    manifest = []
    for vi, voice in enumerate(voices):
        split = "train" if vi < N_VOICES - 2 else "dev"
        for k, text in enumerate(
            sentences(train_s if split == "train" else dev_s, N_PER_VOICE)
        ):
            name = f"{voice}_{k:03d}"
            mp3 = f"{OUT}/{name}.mp3"
            wav = f"{OUT}/{name}.wav"
            try:
                await edge_tts.Communicate(text, voice).save(mp3)
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-loglevel",
                        "error",
                        "-i",
                        mp3,
                        "-ac",
                        "1",
                        "-ar",
                        str(SR),
                        wav,
                    ],
                    check=True,
                )
                os.remove(mp3)
                manifest.append(
                    {"path": wav, "voice": voice, "split": split, "text": text}
                )
            except Exception as e:
                print("skip", name, type(e).__name__, e)
            await asyncio.sleep(0.2)
    pd.DataFrame(manifest).to_csv(
        "audio/data_v2/manifest_fake_multi.csv", index=False
    )
    print(len(manifest), "fake clips written")


if __name__ == "__main__":
    asyncio.run(run())
