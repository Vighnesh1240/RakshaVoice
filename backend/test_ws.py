import asyncio
import json
import sys

import numpy as np
import soundfile as sf
import websockets


async def main(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)

    if sr != 16000:
        raise ValueError(
            f"Expected 16000 Hz, got {sr} Hz. Convert the file first."
        )

    audio = audio.mean(axis=1)

    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains NaN or infinite values.")

    scores = []
    errors = []

    async with websockets.connect(
        "ws://127.0.0.1:8000/ws",
        max_size=2**20,
    ) as ws:
        await ws.send(json.dumps({"type": "select_contact", "name": "Brother"}))
        print("CONTACT:", await ws.recv(), flush=True)

        async def sender():
            for start in range(0, len(audio), 4000):
                chunk = audio[start:start + 4000]

                if len(chunk) == 0:
                    continue

                await ws.send(
                    np.asarray(chunk, dtype="<f4").tobytes()
                )
                print(f"Sent {len(chunk)} samples", flush=True)
                await asyncio.sleep(len(chunk) / 16000)

        async def receiver():
            while True:
                try:
                    message = await ws.recv()
                    result = json.loads(message)
                    if result.get("type") == "error":
                        errors.append(result)
                        print("SERVER ERROR:", result, flush=True)
                    else:
                        scores.append(result)
                        fusion = result.get("streams", {}).get("fusion", {})
                        print(
                            "SCORE",
                            len(scores),
                            "windows_seen=", fusion.get("windows_seen"),
                            "risk=", fusion.get("risk"),
                            "streak=", fusion.get("consecutive_high_windows"),
                            "alert=", fusion.get("alert"),
                            "dropped=", result.get("dropped_windows"),
                            "transcript=", result.get("transcript"),
                            flush=True,
                        )
                except websockets.ConnectionClosed:
                    return

        recv_task = asyncio.create_task(receiver())
        try:
            await sender()
            print(
                "SENDER FINISHED; collecting pending results for 5 seconds.",
                flush=True,
            )
            await asyncio.sleep(5)
        finally:
            recv_task.cancel()
            await asyncio.gather(recv_task, return_exceptions=True)

        print(
            f"\nSUMMARY: received_scores={len(scores)}, errors={len(errors)}",
            flush=True,
        )
        print(
            "Windows seen:",
            [
                s.get("streams", {}).get("fusion", {}).get("windows_seen")
                for s in scores
            ],
            flush=True,
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python backend/test_ws.py test16k.wav"
        )

    asyncio.run(main(sys.argv[1]))
