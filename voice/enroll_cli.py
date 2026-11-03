import re

import numpy as np
import sounddevice as sd

from voice.voice_stream import enroll, list_enrolled


SR = 16000
RECORDING_SECONDS = 8
PROMPTS = (
    "Read a few sentences naturally in any language.",
    "Continue speaking naturally about your day or your work.",
    "Read anything else for this final recording.",
)
CONSENT_STATEMENT = (
    "I consent to RakshaVoice storing a voice embedding for speaker verification. "
    "The recorded audio will not be stored."
)


def _profile_name(value):
    name = value.strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("The name may contain only letters, numbers, underscores, and hyphens.")
    return name


def _record(prompt):
    input(f"\n{prompt}\nPress Enter to record for {RECORDING_SECONDS} seconds: ")
    try:
        recording = sd.rec(
            int(RECORDING_SECONDS * SR),
            samplerate=SR,
            channels=1,
            dtype="float32",
        )
        sd.wait()
    except Exception as exc:
        raise RuntimeError(f"Could not record audio: {exc}") from exc
    return np.asarray(recording[:, 0], dtype=np.float32)


def main():
    try:
        name = _profile_name(input("Name to enroll: "))
        if name in list_enrolled():
            raise ValueError(f"An enrollment named {name!r} already exists.")

        print(f"\nConsent statement:\n{CONSENT_STATEMENT}")
        if input("Type YES to consent: ").strip().upper() != "YES":
            print("Enrollment cancelled; no audio or embedding was stored.")
            return

        recordings = [_record(prompt) for prompt in PROMPTS]
        enroll(name, recordings, CONSENT_STATEMENT)
        print("Enrolled:", list_enrolled())
    except (KeyboardInterrupt, EOFError):
        print("\nEnrollment cancelled; no enrollment was stored.")
    except (ValueError, RuntimeError) as exc:
        print(f"Enrollment failed: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
