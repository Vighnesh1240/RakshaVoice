import argparse
import os

import numpy as np
import pandas as pd
import sounddevice as sd
import soundfile as sf


SR, PAD = 16000, 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--tag", default="laptop")
    args = ap.parse_args()

    out = f"audio/data_v2/replayed_{args.tag}"
    os.makedirs(out, exist_ok=True)
    manifest = pd.read_csv("audio/data_v2/manifest_direct.csv")
    done = {os.path.basename(path)[:-4] for path in os.listdir(out)}
    todo = (
        manifest[~manifest.id.isin(done)]
        .sample(frac=1, random_state=0)
        .head(args.limit)
    )
    print(
        len(done),
        "already recorded;",
        len(todo),
        "to do now (about",
        int(len(todo) * 5.6 / 60),
        "minutes)",
    )

    for n, row in enumerate(todo.itertuples(), 1):
        y, _ = sf.read(row.path, dtype="float32")
        x = np.concatenate(
            [np.zeros(int(PAD * SR)), y, np.zeros(int(PAD * SR))]
        ).astype(np.float32).reshape(-1, 1)
        rec = sd.playrec(x, samplerate=SR, channels=1, dtype="float32")
        sd.wait()
        seg = rec[int(PAD * SR): int(PAD * SR) + len(y), 0]
        rms = float(np.sqrt(np.mean(seg ** 2)))
        if rms < 0.003:
            print(
                "  too quiet (rms %.4f): raise the speaker volume or move "
                "the microphone closer" % rms
            )
            continue
        sf.write(f"{out}/{row.id}.wav", seg, SR)
        if n % 25 == 0:
            print(" ", n, "/", len(todo))

    print("done. Total replayed clips:", len(os.listdir(out)))


if __name__ == "__main__":
    main()
