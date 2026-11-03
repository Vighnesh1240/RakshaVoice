
import time
from pathlib import Path

from nlp.asr import transcribe_file
from nlp.intent_stream import score_text


def analyze_audio(audio_path: str | Path) -> dict:
    """Transcribe audio, then classify its transcript."""
    start = time.perf_counter()

    transcription = transcribe_file(audio_path)
    text = transcription["text"]

    if text.strip():
        predictions = score_text(text)
    else:
        predictions = {}

    total_ms = (time.perf_counter() - start) * 1000

    return {
        "transcription": transcription,
        "text_signals": predictions,
        "latency_ms": round(total_ms, 2),
        "warning": (
            "Predictions are experimental model outputs, "
            "not a definitive scam verdict."
        ),
    }


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m nlp.pipeline <audio_path>")
        raise SystemExit(2)

    result = analyze_audio(sys.argv[1])
    print(json.dumps(result, ensure_ascii=False, indent=2))
