
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DATA_FILE = ROOT / "data" / "test.csv"

LABELS = ["urgency", "authority", "otp_pii", "payment"]
THRESHOLDS = {
    "urgency": 0.5,
    "authority": 0.5,
    "otp_pii": 0.4,
    "payment": 0.5,
}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nlp.intent_stream import score_text


def main():
    df = pd.read_csv(DATA_FILE)
    scores = []

    print(f"Scoring {len(df)} test records...")

    for i, text in enumerate(df["text"].astype(str), start=1):
        result = score_text(text)
        scores.append(result["labels"])

        if i % 50 == 0:
            print(f"Processed {i}/{len(df)}")

    score_df = pd.DataFrame(scores)
    print("\n=== Score distributions ===")

    for label in LABELS:
        actual = df[label].astype(int).to_numpy()
        values = score_df[label].to_numpy(dtype=float)
        threshold = THRESHOLDS[label]

        print(f"\n--- {label} (threshold={threshold}) ---")

        for group_name, mask in [
            ("Actual positive", actual == 1),
            ("Actual negative", actual == 0),
        ]:
            group = values[mask]

            if len(group) == 0:
                continue

            print(
                f"{group_name}: n={len(group)}, "
                f"mean={group.mean():.3f}, "
                f"median={np.median(group):.3f}, "
                f"p90={np.quantile(group, 0.90):.3f}, "
                f"min={group.min():.3f}, "
                f"max={group.max():.3f}"
            )

        missed = (actual == 1) & (values < threshold)
        print(f"Positive examples below threshold: {missed.sum()}")
        print("Examples missed by the model:")

        indices = np.flatnonzero(missed)[:5]
        for idx in indices:
            text = str(df.iloc[idx]["text"]).replace("\n", " ")
            print(
                f"  score={values[idx]:.3f} | "
                f"{text[:240]}"
            )


if __name__ == "__main__":
    main()
