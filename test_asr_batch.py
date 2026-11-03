
import csv
import re
import sys
import time
import statistics
from pathlib import Path

import librosa
from jiwer import wer
from faster_whisper import WhisperModel

ROOT = Path(__file__).resolve().parent
AUDIO_DIR = ROOT / "nlp" / "test_clips"
OUTPUT_CSV = ROOT / "asr_batch_results.csv"

VAD = {"threshold": 0.3, "min_silence_duration_ms": 500}

REFERENCES = {
    "en_scam_otp": "Sir, I am calling from your bank. Please tell me the OTP you just received.",
    "en_scam_police": "This is the cyber crime police. Your Aadhaar is linked to a case and you will be arrested today.",
    "en_scam_pay": "Please transfer five thousand rupees to this account right now to clear your penalty.",
    "en_norm_emi": "Hello, this is a reminder that your EMI will be debited on the tenth.",
    "en_norm_warn": "Please never share your OTP or PIN with anyone, even bank staff.",
    "en_norm_deliv": "Your courier is out for delivery, please keep your phone nearby.",

    "hg_scam_otp": "Sir aapka account block ho jayega, abhi apna OTP bataiye.",
    "hg_scam_police": "Main CBI officer bol raha hoon, aapke naam par case darj hua hai.",
    "hg_scam_pay": "Processing fee ke liye abhi paanch hazaar rupaye UPI se bhej dijiye.",
    "hg_norm_emi": "Aapki EMI dus tareekh ko kategi, balance rakhiyega.",
    "hg_norm_warn": "OTP kisi ko mat batana, bank kabhi nahi maangta.",
    "hg_norm_deliv": "Aapka parcel kal subah tak deliver ho jayega.",

    "hi_scam_otp": "आपका खाता बंद हो जाएगा, अभी अपना ओटीपी बताइए।",
    "hi_scam_police": "मैं सीबीआई अधिकारी बोल रहा हूँ, आपके खिलाफ वारंट जारी हुआ है।",
    "hi_scam_pay": "प्रोसेसिंग फीस के लिए अभी पाँच हज़ार रुपये भेज दीजिए।",
    "hi_norm_emi": "आपकी ईएमआई दस तारीख को कटेगी, कृपया बैलेंस रखिएगा।",
    "hi_norm_warn": "अपना ओटीपी किसी को न बताएँ, बैंक कभी नहीं माँगता।",
    "hi_norm_deliv": "आपका पार्सल कल सुबह तक पहुँच जाएगा。",
}

def normalize_for_wer(text):
    # Ignore punctuation and case; retain words and script.
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())

def language_for(stem):
    prefix = stem.split("_", 1)[0]
    return {"en": "en", "hg": "hi", "hi": "hi"}[prefix]

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    files = sorted(AUDIO_DIR.glob("*.wav"))
    if not files:
        raise FileNotFoundError(f"No WAV files found in {AUDIO_DIR}")

    print(f"Found {len(files)} WAV files.", flush=True)
    print("Loading small model on CUDA...", flush=True)
    model = WhisperModel(
        "small",
        device="cuda",
        compute_type="int8_float16",
    )
    print("Model loaded. Running warm-up transcription...", flush=True)

    # Warm up with one existing clip; exclude this from reported results.
    warmup = files[0]
    audio, _ = librosa.load(warmup, sr=16000, mono=True)
    segments, _ = model.transcribe(
        audio, language=language_for(warmup.stem),
        beam_size=1, vad_filter=True, vad_parameters=VAD,
        temperature=0.0, condition_on_previous_text=False,
    )
    list(segments)
    print("Warm-up complete. Processing clips...", flush=True)

    results = []
    for i, path in enumerate(files, start=1):
        # Example: en_scam_otp__far.wav -> en_scam_otp
        scenario = path.stem.rsplit("__", 1)[0]
        condition = path.stem.rsplit("__", 1)[-1]
        reference = REFERENCES.get(scenario)

        if reference is None:
            print(f"SKIP: no reference for {path.name}")
            continue

        audio, _ = librosa.load(path, sr=16000, mono=True)
        lang = language_for(scenario)

        print(f"[{i}/{len(files)}] Processing {path.name}...", flush=True)
        start = time.perf_counter()
        segments, info = model.transcribe(
            audio,
            language=lang,
            beam_size=1,
            vad_filter=True,
            vad_parameters=VAD,
            temperature=0.0,
            condition_on_previous_text=False,
        )
        transcript = " ".join(s.text.strip() for s in segments).strip()
        elapsed_ms = (time.perf_counter() - start) * 1000

        error_rate = wer(
            normalize_for_wer(reference),
            normalize_for_wer(transcript),
        )

        row = {
            "filename": path.name,
            "scenario": scenario,
            "condition": condition,
            "language_requested": lang,
            "language_detected": info.language,
            "latency_ms": round(elapsed_ms, 1),
            "wer": round(error_rate, 4),
            "reference": reference,
            "transcript": transcript,
            "empty_transcript": not bool(transcript.strip()),
        }
        results.append(row)
        print(
            f"[{i}/{len(files)}] {path.name} | "
            f"{elapsed_ms:.0f} ms | WER={error_rate:.3f} | "
            f"{transcript!r}"
        )

    if not results:
        raise RuntimeError("No results generated. Check filenames and references.")

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    latencies = [r["latency_ms"] for r in results]
    wers = [r["wer"] for r in results]

    print("\n===== ASR BENCHMARK SUMMARY =====")
    print(f"Clips evaluated: {len(results)}")
    print(f"Mean latency:    {statistics.mean(latencies):.1f} ms")
    print(f"Median latency:  {statistics.median(latencies):.1f} ms")
    print(f"Maximum latency: {max(latencies):.1f} ms")
    print(f"Mean clip WER:   {statistics.mean(wers):.4f}")
    print(f"Empty outputs:   {sum(r['empty_transcript'] for r in results)}")
    print(f"Results saved:   {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
