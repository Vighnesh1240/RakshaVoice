
from pathlib import Path
import pandas as pd

DATA_DIR = Path("nlp/data")
INPUT_FILE = DATA_DIR / "auto_labeled.csv"

TRAIN_IDS = [1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 26, 27]
VAL_IDS = [19, 24, 28]
TEST_IDS = [20, 25, 29]

def main():
    df = pd.read_csv(INPUT_FILE)

    splits = {
        "train": TRAIN_IDS,
        "val": VAL_IDS,
        "test": TEST_IDS,
    }

    # Validate scenario assignments.
    all_ids = TRAIN_IDS + VAL_IDS + TEST_IDS

    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Scenario IDs overlap between splits.")

    actual_ids = set(df["scenario_id"].unique())
    if set(all_ids) != actual_ids:
        raise ValueError(
            f"Scenario mismatch. Missing: {actual_ids - set(all_ids)}; "
            f"Unknown: {set(all_ids) - actual_ids}"
        )

    label_cols = ["urgency", "authority", "otp_pii", "payment"]

    for name, scenario_ids in splits.items():
        subset = df[df["scenario_id"].isin(scenario_ids)].copy()
        subset.to_csv(DATA_DIR / f"{name}.csv", index=False)

        print(f"\n{name.upper()}: {len(subset)} records")
        print("Verdict distribution:")
        print(subset["final_verdict"].value_counts().to_string())
        print("Tactic positives:")
        print(subset[label_cols].sum().to_string())

    print("\nSplit files created in nlp/data/")

if __name__ == "__main__":
    main()
