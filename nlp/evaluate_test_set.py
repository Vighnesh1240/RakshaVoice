
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "test.csv"
MODELS_DIR = ROOT / "models"
REPORT_FILE = ROOT / "test_evaluation_report.json"

LABELS = ["urgency", "authority", "otp_pii", "payment"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16


class MultiLabelHead(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_labels=4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, x):
        return self.network(x)


def load_classifier(version):
    folder = MODELS_DIR / version

    config = json.loads(
        (folder / "config.json").read_text(encoding="utf-8")
    )
    checkpoint = torch.load(
        folder / "classifier.pt",
        map_location=DEVICE,
        weights_only=True,
    )

    norm = np.load(folder / "normalization.npz")
    mean = norm["mean"].astype(np.float32)
    std = norm["std"].astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std)

    head = MultiLabelHead(
        input_dim=checkpoint.get("input_dim", 768),
        hidden_dim=checkpoint.get("hidden_dim", 256),
        num_labels=len(LABELS),
    )
    head.load_state_dict(checkpoint["state_dict"])
    head.to(DEVICE).eval()

    return {
        "config": config,
        "head": head,
        "mean": mean,
        "std": std,
    }


def main():
    df = pd.read_csv(DATA_FILE)

    required = ["text"] + LABELS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df[required].isna().any().any():
        raise ValueError(
            "Missing text or label values found. "
            "Inspect the test CSV before evaluation."
        )

    y_true = df[LABELS].astype(int).to_numpy()
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Labels must contain only 0 and 1.")

    models = {
        "text_v1": load_classifier("text_v1"),
        "text_v2": load_classifier("text_v2"),
    }

    encoders = {
        m["config"]["encoder"] for m in models.values()
    }
    if len(encoders) != 1:
        raise ValueError(f"Models use different encoders: {encoders}")

    encoder_name = next(iter(encoders))
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    encoder = AutoModel.from_pretrained(encoder_name).to(DEVICE)
    encoder.eval()

    texts = df["text"].astype(str).tolist()
    embeddings = []

    print(f"Device: {DEVICE}")
    print(f"Test records: {len(df)}")
    print(f"Encoder: {encoder_name}")
    print("Extracting test embeddings...")

    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start:start + BATCH_SIZE]
            max_length = models["text_v1"]["config"].get(
                "max_length", 192
            )
            tokens = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tokens = {k: v.to(DEVICE) for k, v in tokens.items()}
            vectors = encoder(**tokens).last_hidden_state[:, 0, :]
            embeddings.append(vectors.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    del encoder

    report = {
        "test_records": len(df),
        "device": str(DEVICE),
        "labels": LABELS,
        "models": {},
    }

    for name, model in models.items():
        mean = model["mean"]
        std = model["std"]
        normalized = (embeddings - mean) / std

        with torch.no_grad():
            logits = model["head"](
                torch.tensor(
                    normalized, dtype=torch.float32, device=DEVICE
                )
            )
            probabilities = torch.sigmoid(logits).cpu().numpy()

        thresholds = np.asarray(model["config"]["thresholds"])
        y_pred = (probabilities >= thresholds).astype(int)

        exact_match = float(np.all(y_pred == y_true, axis=1).mean())
        per_label = {}

        for j, label in enumerate(LABELS):
            precision, recall, f1, support = (
                precision_recall_fscore_support(
                    y_true[:, j],
                    y_pred[:, j],
                    average="binary",
                    zero_division=0,
                )
            )
            per_label[label] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(y_true[:, j].sum()),
                "confusion_matrix": confusion_matrix(
                    y_true[:, j], y_pred[:, j], labels=[0, 1]
                ).tolist(),
                "threshold": float(thresholds[j]),
            }

        micro = precision_recall_fscore_support(
            y_true, y_pred, average="micro", zero_division=0
        )
        macro = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )

        report["models"][name] = {
            "exact_match_accuracy": exact_match,
            "micro_precision": float(micro[0]),
            "micro_recall": float(micro[1]),
            "micro_f1": float(micro[2]),
            "macro_f1": float(macro[2]),
            "per_label": per_label,
        }

        print(f"\n=== {name} ===")
        print(f"Exact-match accuracy: {exact_match:.4f}")
        print(f"Micro F1:             {micro[2]:.4f}")
        print(f"Macro F1:             {macro[2]:.4f}")

        for label, metrics in per_label.items():
            print(
                f"{label:10s} "
                f"P={metrics['precision']:.3f} "
                f"R={metrics['recall']:.3f} "
                f"F1={metrics['f1']:.3f} "
                f"support={metrics['support']}"
            )

    REPORT_FILE.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(f"\nReport saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
