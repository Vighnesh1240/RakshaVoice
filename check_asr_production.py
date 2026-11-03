
import csv
import re
import statistics
import time
from collections import Counter
from pathlib import Path

import soundfile as sf

from nlp.asr import transcribe

ROOT = Path(__file__).resolve().parent
AUDIO_DIR = ROOT / "nlp" / "test_clips"
OUTPUT_CSV = ROOT / "asr_production_batch_results.csv"

# Scripted reference transcripts used to create the test clips.
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
    "hg_norm_warn": "OTP kisi ko mat batana, bank kabhi nahi maangta. Nahin, bas karte hain.",
    "hg_norm_deliv": "Aapka parcel kal subah tak deliver ho jayega.",

    
    "hi_scam_otp": "Aapka khaata band ho jaega, abhi apna OTP bataiye.",
    "hi_scam_police": "Main CBI adhikaari bol raha hoon, aapke khilaaf warrant jaari hua hai.",
    "hi_scam_pay": "Processing fees ke liye abhi paanch hazaar rupaye bhej dijiye.",
    "hi_norm_emi": "Aapki EMI das tareekh ko kategi, kripya balance rakhiyega.",
    "hi_norm_warn": "Apna OTP kisi ko na bataen, bank kabhi nahi maangta.",
    "hi_norm_deliv": "Aapka parcel kal subah tak pahunch jaega.",

}


def normalize_text(text):
    """Normalize case and punctuation while preserving script."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def calculate_wer(reference, hypothesis):
    """Compute WER without requiring jiwer."""
    ref = normalize_text(reference).split()
    hyp = normalize_text(hypothesis).split()

    if not ref:
        return 0.0 if not hyp else float("inf")

    # Levenshtein edit distance over word sequences.
    previous = list(range(len(hyp) + 1))

    for i, ref_word in enumerate(ref, start=1):
        current = [i]
        for j, hyp_word in enumerate(hyp, start=1):
            substitution = previous[j - 1] + (ref_word != hyp_word)
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current

    return previous[-1] / len(ref)


def repetition_ratio(text):
    """Share of words belonging to the most frequent token."""
    words = normalize_text(text).split()
    if not words:
        return 0.0

    most_common_count = Counter(words).most_common(1)[0][1]
    return most_common_count / len(words)


def parse_filename(path):
    # Example: hg_scam_otp__far.wav
    if "__" not in path.stem:
        return None

    scenario, condition = path.stem.rsplit("__", 1)
    language = scenario.split("_", 1)[0]

    if language not in {"en", "hg", "hi"}:
        return None

    return scenario, language, condition


def main():
    if not AUDIO_DIR.exists():
        raise FileNotFoundError(f"Audio folder not found: {AUDIO_DIR}")

    files = sorted(AUDIO_DIR.glob("*.wav"))
    if not files:
        raise FileNotFoundError(f"No WAV files found in {AUDIO_DIR}")

    results = []

    print(f"Audio folder: {AUDIO_DIR}")
    print(f"Found {len(files)} WAV files.")
    print("Using production transcribe(audio, language=None).\n")

    for index, path in enumerate(files, start=1):
        parsed = parse_filename(path)

        if parsed is None:
            print(f"Skipping unexpected filename: {path.name}")
            continue

        scenario, language, condition = parsed
        reference = REFERENCES.get(scenario)

        if reference is None:
            print(f"Skipping missing reference: {path.name}")
            continue

        try:
            audio, sample_rate = sf.read(path, dtype="float32")

            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            if sample_rate != 16000:
                raise ValueError(
                    f"Expected 16000 Hz audio; found {sample_rate} Hz"
                )

            start = time.perf_counter()
            transcript = transcribe(audio, language=None)
            latency_ms = (time.perf_counter() - start) * 1000

            transcript = str(transcript).strip()
            wer_score = calculate_wer(reference, transcript)

            # Hinglish references are Roman-script. WER may be misleading
            # if ASR outputs the same words in Devanagari.
            script_mismatch = (
                language == "hg"
                and bool(re.search(r"[\u0900-\u097F]", transcript))
            )

            row = {
                "filename": path.name,
                "scenario": scenario,
                "language": language,
                "condition": condition,
                "latency_ms": round(latency_ms, 1),
                "wer": round(wer_score, 4),
                "reference": reference,
                "transcript": transcript,
                "empty_transcript": not bool(transcript),
                "repetition_ratio": round(repetition_ratio(transcript), 4),
                "hinglish_script_mismatch": script_mismatch,
                "status": "ok",
                "error": "",
            }

            print(
                f"[{index}/{len(files)}] {path.name} | "
                f"{latency_ms:.0f} ms | WER={wer_score:.3f}"
            )

        except Exception as exc:
            row = {
                "filename": path.name,
                "scenario": scenario,
                "language": language,
                "condition": condition,
                "latency_ms": "",
                "wer": "",
                "reference": reference,
                "transcript": "",
                "empty_transcript": True,
                "repetition_ratio": "",
                "hinglish_script_mismatch": "",
                "status": "error",
                "error": repr(exc),
            }
            print(f"[{index}/{len(files)}] ERROR: {path.name}: {exc}")

        results.append(row)

    if not results:
        raise RuntimeError("No clips were evaluated.")

    with OUTPUT_CSV.open(
        "w", newline="", encoding="utf-8-sig"
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    successful = [
        row for row in results
        if row["status"] == "ok"
    ]

    print("\n========== PRODUCTION ASR SUMMARY ==========")
    print(f"Files discovered:  {len(files)}")
    print(f"Clips evaluated:   {len(successful)}")
    print(f"Failed clips:      {len(results) - len(successful)}")

    if successful:
        latencies = [row["latency_ms"] for row in successful]
        wers = [row["wer"] for row in successful]

        sorted_latencies = sorted(latencies)
        p95_index = max(0, int(0.95 * len(sorted_latencies) + 0.999999) - 1)

        print(f"Mean latency:      {statistics.mean(latencies):.1f} ms")
        print(f"Median latency:    {statistics.median(latencies):.1f} ms")
        print(f"P95 latency:       {sorted_latencies[p95_index]:.1f} ms")
        print(f"Maximum latency:   {max(latencies):.1f} ms")
        print(f"Mean clip WER:     {statistics.mean(wers):.4f}")
        print(
            "Empty transcripts:",
            sum(row["empty_transcript"] for row in successful)
        )
        print(
            "High repetition (>50%):",
            sum(row["repetition_ratio"] > 0.5 for row in successful)
        )
        print(
            "Hinglish script mismatches:",
            sum(row["hinglish_script_mismatch"] for row in successful)
        )

        print("\n--- Breakdown by language and condition ---")
        for language in ("en", "hg", "hi"):
            for condition in ("clean", "raw", "far"):
                subset = [
                    row for row in successful
                    if row["language"] == language
                    and row["condition"] == condition
                ]
                if not subset:
                    continue

                print(
                    f"{language}/{condition}: "
                    f"n={len(subset)}, "
                    f"mean_latency={statistics.mean(r['latency_ms'] for r in subset):.1f} ms, "
                    f"mean_clip_WER={statistics.mean(r['wer'] for r in subset):.4f}"
                )

    print(f"\nCSV saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
