import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support,
    average_precision_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DATA_FILE = ROOT / "data" / "reviewed_test_gold.csv"
OUTPUT_FILE = ROOT / "reviewed_gold_probability_diagnostics.json"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nlp.intent_stream import _probs, LABELS, THR, MAXLEN, _tok

df = pd.read_csv(DATA_FILE)

texts = df["text"].astype(str).tolist()
y_true = df[LABELS].astype(int).to_numpy()

print("Calculating raw model probabilities...")

all_probs = []

for start in range(0, len(texts), 16):
    batch = texts[start:start + 16]
    all_probs.append(_probs(batch))

    print(f"Processed {min(start + 16, len(texts))}/{len(texts)}")

probabilities = np.vstack(all_probs)

# Measure how much of each transcript fits within the model's token limit.
token_lengths = []

for text in texts:
    encoded = _tok(
        text,
        truncation=False,
        add_special_tokens=True
    )
    token_lengths.append(len(encoded["input_ids"]))

token_lengths = np.asarray(token_lengths)

report = {
    "dataset": "reviewed_test_gold.csv",
    "records": len(df),
    "model_max_length": MAXLEN,
    "labels": {},
    "token_length": {
        "mean": float(token_lengths.mean()),
        "median": float(np.median(token_lengths)),
        "maximum": int(token_lengths.max()),
        "records_exceeding_max_length": int(
            (token_lengths > MAXLEN).sum()
        ),
        "percentage_exceeding_max_length": float(
            (token_lengths > MAXLEN).mean() * 100
        ),
    },
}

print("\n" + "=" * 75)
print("RAW MODEL PROBABILITY DIAGNOSTICS")
print("=" * 75)

for j, label in enumerate(LABELS):

    y = y_true[:, j]
    p = probabilities[:, j]

    positive = p[y == 1]
    negative = p[y == 0]

    configured_threshold = float(THR[j])
    raw_predictions = (p >= configured_threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y,
        raw_predictions,
        average="binary",
        zero_division=0
    )

    threshold_results = {}

    for threshold in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:

        pred = (p >= threshold).astype(int)

        pr, rc, f, _ = precision_recall_fscore_support(
            y,
            pred,
            average="binary",
            zero_division=0
        )

        threshold_results[str(threshold)] = {
            "precision": float(pr),
            "recall": float(rc),
            "f1": float(f),
            "predicted_positive": int(pred.sum())
        }

    label_report = {
        "actual_positive": int(y.sum()),
        "actual_negative": int((y == 0).sum()),
        "configured_threshold": configured_threshold,
        "raw_predictions_at_configured_threshold": int(
            raw_predictions.sum()
        ),
        "raw_precision": float(precision),
        "raw_recall": float(recall),
        "raw_f1": float(f1),
        "positive_probability_mean": float(positive.mean()),
        "negative_probability_mean": float(negative.mean()),
        "positive_probability_median": float(np.median(positive)),
        "negative_probability_median": float(np.median(negative)),
        "positive_probability_max": float(positive.max()),
        "negative_probability_max": float(negative.max()),
        "average_precision": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "threshold_analysis": threshold_results,
    }

    report["labels"][label] = label_report

    print(f"\n{label.upper()}")
    print("Actual positives:", int(y.sum()))
    print("Configured threshold:", configured_threshold)
    print("Raw predicted positives:", int(raw_predictions.sum()))
    print(
        f"Positive probability mean: {positive.mean():.4f}"
    )
    print(
        f"Negative probability mean: {negative.mean():.4f}"
    )
    print(
        f"Positive probability median: {np.median(positive):.4f}"
    )
    print(
        f"Negative probability median: {np.median(negative):.4f}"
    )
    print(f"Average precision: {label_report['average_precision']:.4f}")
    print(f"ROC-AUC: {label_report['roc_auc']:.4f}")

    print("\nThreshold comparison:")
    print("Threshold | Precision | Recall | F1 | Predicted+")

    for threshold, values in threshold_results.items():
        print(
            f"{float(threshold):9.2f} | "
            f"{values['precision']:.4f} | "
            f"{values['recall']:.4f} | "
            f"{values['f1']:.4f} | "
            f"{values['predicted_positive']}"
        )

print("\n" + "=" * 75)
print("TOKEN LENGTH DIAGNOSTICS")
print("=" * 75)

print("Model maximum tokens:", MAXLEN)
print("Mean transcript tokens:", round(token_lengths.mean(), 2))
print("Median transcript tokens:", int(np.median(token_lengths)))
print("Maximum transcript tokens:", int(token_lengths.max()))
print(
    "Records exceeding model limit:",
    report["token_length"]["records_exceeding_max_length"]
)
print(
    "Percentage exceeding model limit:",
    round(report["token_length"]["percentage_exceeding_max_length"], 2),
    "%"
)

OUTPUT_FILE.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)

print("\nDiagnostic report saved to:", OUTPUT_FILE)
print("No model weights, thresholds or production files were changed.")
