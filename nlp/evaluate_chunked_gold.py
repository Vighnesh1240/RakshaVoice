import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score
)

from intent_stream import (
    _model, _tok, DEV, LABELS, THR, MAXLEN
)

import torch

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "nlp" / "data" / "reviewed_test_gold.csv"
OUTPUT = ROOT / "nlp" / "chunked_gold_evaluation.json"

CHUNK_SIZE = 64
OVERLAP = 16
BATCH_SIZE = 32


def make_chunks(text):
    encoded = _tok(
        str(text),
        add_special_tokens=False,
        truncation=False
    )["input_ids"]

    if not encoded:
        return [[]]

    step = CHUNK_SIZE - OVERLAP
    chunks = []

    for start in range(0, len(encoded), step):
        chunk = encoded[start:start + CHUNK_SIZE]

        if not chunk:
            continue

        chunks.append(chunk)

        if start + CHUNK_SIZE >= len(encoded):
            break

    return chunks


@torch.no_grad()
def predict_chunks(all_chunks):
    results = []

    for start in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[start:start + BATCH_SIZE]

        enc = _tok.pad(
            {"input_ids": batch},
            padding=True,
            return_tensors="pt"
        ).to(DEV)

        probs = torch.sigmoid(
            _model(
                enc["input_ids"],
                enc["attention_mask"]
            ).float()
        ).cpu().numpy()

        results.extend(probs)

    return np.asarray(results)


def aggregate(scores, method):
    if method == "max":
        return scores.max(axis=0)

    if method == "mean":
        return scores.mean(axis=0)

    if method == "top2_mean":
        top = np.sort(scores, axis=0)[-2:]
        return top.mean(axis=0)

    raise ValueError(method)


def evaluate(y_true, y_prob, thresholds):
    y_pred = (y_prob >= thresholds).astype(int)

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average=None,
            zero_division=0
        )
    )

    output = {}

    for i, label in enumerate(LABELS):
        output[label] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
            "predicted_positive": int(y_pred[:, i].sum()),
            "roc_auc": float(roc_auc_score(
                y_true[:, i], y_prob[:, i]
            )),
            "average_precision": float(average_precision_score(
                y_true[:, i], y_prob[:, i]
            ))
        }

    return output


def main():
    print("Loading reviewed gold dataset...")

    df = pd.read_csv(INPUT)

    y_true = df[LABELS].astype(int).values

    print("Creating overlapping token chunks...")

    all_chunks = []
    conversation_chunks = []

    for text in df["text"]:
        chunks = make_chunks(text)

        indices = list(
            range(len(all_chunks), len(all_chunks) + len(chunks))
        )

        all_chunks.extend(chunks)
        conversation_chunks.append(indices)

    print("Total transcripts:", len(df))
    print("Total chunks:", len(all_chunks))
    print(
        "Average chunks per transcript:",
        round(np.mean([len(x) for x in conversation_chunks]), 2)
    )

    print("Running model inference...")

    chunk_probs = predict_chunks(all_chunks)

    transcript_probs = {}

    for method in ["max", "mean", "top2_mean"]:
        scores = []

        for indices in conversation_chunks:
            scores.append(
                aggregate(chunk_probs[indices], method)
            )

        scores = np.asarray(scores)

        transcript_probs[method] = evaluate(
            y_true,
            scores,
            np.asarray(THR)
        )

    report = {
        "dataset": str(INPUT),
        "transcripts": len(df),
        "total_chunks": len(all_chunks),
        "chunk_size": CHUNK_SIZE,
        "overlap": OVERLAP,
        "configured_thresholds": dict(zip(LABELS, THR)),
        "aggregation_results": transcript_probs
    }

    OUTPUT.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("\n========== CHUNKED GOLD EVALUATION ==========")

    for method, results in transcript_probs.items():
        print(f"\nAGGREGATION: {method.upper()}")

        for label, metrics in results.items():
            print(
                f"{label}: "
                f"P={metrics['precision']:.4f}, "
                f"R={metrics['recall']:.4f}, "
                f"F1={metrics['f1']:.4f}, "
                f"AUC={metrics['roc_auc']:.4f}, "
                f"Predicted+={metrics['predicted_positive']}"
            )

    print("\nReport saved to:", OUTPUT)


if __name__ == "__main__":
    main()