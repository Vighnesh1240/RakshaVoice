import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    accuracy_score,
)

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

DATA_FILE = ROOT / "data" / "reviewed_test_gold.csv"
REPORT_FILE = ROOT / "reviewed_gold_evaluation_report.json"

LABELS = ["urgency", "authority", "otp_pii", "payment"]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nlp.intent_stream import score_text


def main():

    df = pd.read_csv(DATA_FILE)

    required = ["text"] + LABELS
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df[required].isna().any().any():
        raise ValueError("Missing text or reviewed label values.")

    y_true = df[LABELS].astype(int).to_numpy()

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Labels must contain only 0 and 1.")

    predictions = []

    print(f"Evaluating {len(df)} reviewed records...")
    print("Using existing production intent_v4 pipeline.")
    print("This may take a few minutes.")

    for i, text in enumerate(df["text"].astype(str), start=1):

        result = score_text(text)
        evidence = set(result["evidence"])

        predictions.append([
            int(label in evidence) for label in LABELS
        ])

        if i % 25 == 0:
            print(f"Processed {i}/{len(df)}")

    y_pred = np.asarray(predictions, dtype=int)

    report = {
        "model": "intent_v4 production score_text",
        "evaluation_dataset": "reviewed_test_gold.csv",
        "test_records": int(len(df)),
        "labels": LABELS,
        "evaluation_note": (
            "Evaluation against manually reviewed labels. "
            "Predictions use the existing production pipeline, "
            "including configured thresholds and contextual fallback. "
            "The dataset is diagnostic and is not an independent "
            "generalization test."
        ),
        "exact_match_accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "per_label": {},
    }

    for j, label in enumerate(LABELS):

        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                y_true[:, j],
                y_pred[:, j],
                average="binary",
                zero_division=0,
            )
        )

        cm = confusion_matrix(
            y_true[:, j],
            y_pred[:, j],
            labels=[0, 1],
        )

        report["per_label"][label] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(y_true[:, j].sum()),
            "predicted_positive": int(y_pred[:, j].sum()),
            "confusion_matrix": cm.tolist(),
        }

    for average in ("micro", "macro", "weighted"):

        p, r, f, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average=average,
            zero_division=0,
        )

        report[f"{average}_precision"] = float(p)
        report[f"{average}_recall"] = float(r)
        report[f"{average}_f1"] = float(f)

    REPORT_FILE.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n=== Reviewed Gold Evaluation Results ===")

    print(
        f"Exact-match accuracy: "
        f"{report['exact_match_accuracy']:.4f}"
    )

    for average in ("micro", "macro", "weighted"):

        print(
            f"{average.title():8s} "
            f"P={report[f'{average}_precision']:.4f} "
            f"R={report[f'{average}_recall']:.4f} "
            f"F1={report[f'{average}_f1']:.4f}"
        )

    print("\nPER-LABEL RESULTS")

    for label, m in report["per_label"].items():

        print(
            f"\n{label.upper()}"
            f"\nPrecision: {m['precision']:.4f}"
            f"\nRecall: {m['recall']:.4f}"
            f"\nF1: {m['f1']:.4f}"
            f"\nActual positives: {m['support']}"
            f"\nPredicted positives: {m['predicted_positive']}"
            f"\nConfusion matrix: {m['confusion_matrix']}"
        )

    print(f"\nReport saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
