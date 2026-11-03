
import csv
import time
from pathlib import Path

from nlp.pipeline import analyze_audio


ROOT = Path(__file__).resolve().parent.parent
AUDIO_FILES = [
    ROOT / "10.wav",
    ROOT / "1959.wav",
    ROOT / "4.wav",
    ROOT / "test16k.wav",
]
OUTPUT = ROOT / "nlp" / "pipeline_evaluation.csv"


def main():
    rows = []

    for audio_path in AUDIO_FILES:
        if not audio_path.is_file():
            print(f"Skipping missing file: {audio_path.name}")
            continue

        print(f"\nEvaluating {audio_path.name}...")
        start = time.perf_counter()

        try:
            result = analyze_audio(audio_path)
            transcription = result["transcription"]
            signals = result["text_signals"]

            row = {
                "filename": audio_path.name,
                "language": transcription["language"],
                "language_probability": transcription["language_probability"],
                "duration_seconds": transcription["duration_seconds"],
                "transcript": transcription["text"],
                "urgency_score": signals.get("urgency", {}).get("score"),
                "urgency_detected": signals.get("urgency", {}).get("detected"),
                "authority_score": signals.get("authority", {}).get("score"),
                "authority_detected": signals.get("authority", {}).get("detected"),
                "otp_pii_score": signals.get("otp_pii", {}).get("score"),
                "otp_pii_detected": signals.get("otp_pii", {}).get("detected"),
                "payment_score": signals.get("payment", {}).get("score"),
                "payment_detected": signals.get("payment", {}).get("detected"),
                "pipeline_latency_ms": round(
                    (time.perf_counter() - start) * 1000, 2
                ),
                "error": "",
            }

            print("Transcript:", transcription["text"])
            print("Language:", transcription["language"])
            print("Signals:", signals)
            print("Latency:", row["pipeline_latency_ms"], "ms")

        except Exception as exc:
            row = {
                "filename": audio_path.name,
                "error": str(exc),
            }
            print("ERROR:", exc)

        rows.append(row)

    if not rows:
        print("No audio files were evaluated.")
        return

    columns = list(dict.fromkeys(key for row in rows for key in row))

    with OUTPUT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nEvaluation saved to: {OUTPUT}")
    print(f"Files evaluated: {len(rows)}")


if __name__ == "__main__":
    main()
