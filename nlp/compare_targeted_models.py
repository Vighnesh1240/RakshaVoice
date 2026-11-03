
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
CASES_FILE = ROOT / "targeted_sanity_cases.json"
REPORT_FILE = ROOT / "targeted_model_comparison.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = ["urgency", "authority", "otp_pii", "payment"]


# ==================================================
# CLASSIFIER ARCHITECTURE
# Must match the training architecture.
# ==================================================

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


# ==================================================
# LOAD A MODEL
# ==================================================

def load_model(version):
    folder = MODELS_DIR / version

    config_path = folder / "config.json"
    checkpoint_path = folder / "classifier.pt"
    normalization_path = folder / "normalization.npz"

    for path in [config_path, checkpoint_path, normalization_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    normalization = np.load(normalization_path)

    mean = normalization["mean"].astype(np.float32)
    std = normalization["std"].astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True,
    )

    classifier = MultiLabelHead(
        input_dim=checkpoint.get("input_dim", 768),
        hidden_dim=checkpoint.get("hidden_dim", 256),
        num_labels=len(LABELS),
    )

    classifier.load_state_dict(checkpoint["state_dict"])
    classifier.to(DEVICE)
    classifier.eval()

    encoder_name = config["encoder"]

    print(f"\nLoading {version}: {folder}")
    print(f"Encoder: {encoder_name}")

    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    encoder = AutoModel.from_pretrained(encoder_name)
    encoder.to(DEVICE)
    encoder.eval()

    return {
        "version": version,
        "tokenizer": tokenizer,
        "encoder": encoder,
        "classifier": classifier,
        "mean": torch.tensor(mean, dtype=torch.float32, device=DEVICE),
        "std": torch.tensor(std, dtype=torch.float32, device=DEVICE),
        "thresholds": np.asarray(config["thresholds"], dtype=np.float32),
        "max_length": config.get("max_length", 192),
    }


# ==================================================
# PREDICT ONE TEXT
# ==================================================

@torch.no_grad()
def predict_text(model, text):
    inputs = model["tokenizer"](
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=model["max_length"],
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    outputs = model["encoder"](**inputs)

    # CLS embedding, matching the training pipeline.
    embedding = outputs.last_hidden_state[:, 0, :]

    embedding = (embedding - model["mean"]) / model["std"]

    logits = model["classifier"](embedding)
    scores = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    predictions = (
        scores >= model["thresholds"]
    ).astype(int)

    return scores, predictions


# ==================================================
# LOAD TEST CASES
#
# Supports your existing format:
# {
#   "id": "D01",
#   "lang": "English",
#   "text": "...",
#   "expected": ["urgency", "otp_pii"]
# }
#
# Also supports a JSON list of cases.
# ==================================================

def load_cases():
    if not CASES_FILE.exists():
        raise FileNotFoundError(f"Test file not found: {CASES_FILE}")

    with open(CASES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support either a JSON array or a single JSON object.
    if isinstance(data, dict):
        if "cases" in data:
            cases = data["cases"]
        else:
            cases = [data]
    elif isinstance(data, list):
        cases = data
    else:
        raise ValueError("The JSON must contain a list or an object of test cases.")

    if not cases:
        raise ValueError("No test cases were found.")

    for i, case in enumerate(cases, start=1):
        if "text" not in case:
            raise ValueError(f"Case {i} is missing 'text'.")

        if "expected" in case:
            expected = case["expected"]
            if not isinstance(expected, list):
                raise ValueError(
                    f"Case {i}: 'expected' must be a list of label names."
                )

            invalid = set(expected) - set(LABELS)
            if invalid:
                raise ValueError(
                    f"Case {i} has unknown expected labels: {sorted(invalid)}"
                )

        elif "labels" in case:
            # Backward compatibility with dictionary-based labels.
            if not isinstance(case["labels"], dict):
                raise ValueError(f"Case {i}: 'labels' must be a dictionary.")
        else:
            raise ValueError(
                f"Case {i} needs either 'expected' or 'labels'."
            )

    return cases


def get_expected_vector(case):
    if "expected" in case:
        expected_names = set(case["expected"])
        return np.array(
            [int(label in expected_names) for label in LABELS],
            dtype=int,
        )

    return np.array(
        [int(case["labels"].get(label, 0)) for label in LABELS],
        dtype=int,
    )


# ==================================================
# EVALUATE MODEL
# ==================================================

def evaluate_model(model, cases):
    results = []
    exact_matches = 0
    total_label_correct = 0

    for index, case in enumerate(cases, start=1):
        scores, predictions = predict_text(model, case["text"])
        expected = get_expected_vector(case)

        exact_match = bool(np.array_equal(predictions, expected))

        exact_matches += int(exact_match)
        total_label_correct += int(np.sum(predictions == expected))

        results.append({
            "id": case.get("id", str(index)),
            "lang": case.get("lang", "Unknown"),
            "text": case["text"],
            "expected": expected.tolist(),
            "scores": scores.tolist(),
            "predictions": predictions.tolist(),
            "exact_match": exact_match,
        })

    total_labels = len(cases) * len(LABELS)

    summary = {
        "version": model["version"],
        "total_cases": len(cases),
        "exact_matches": exact_matches,
        "exact_match_accuracy": exact_matches / len(cases),
        "label_accuracy": total_label_correct / total_labels,
    }

    return summary, results


# ==================================================
# PRINT RESULTS
# ==================================================

def print_results(summary, results):
    print("\n" + "=" * 76)
    print(f"MODEL: {summary['version'].upper()}")
    print("=" * 76)

    print(f"Cases: {summary['total_cases']}")
    print(f"Exact matches: {summary['exact_matches']}/{summary['total_cases']}")
    print(f"Exact-match accuracy: {summary['exact_match_accuracy']:.1%}")
    print(f"Individual-label accuracy: {summary['label_accuracy']:.1%}")

    for result in results:
        expected_names = [
            label for label, value in zip(LABELS, result["expected"]) if value
        ]
        predicted_names = [
            label for label, value in zip(LABELS, result["predictions"]) if value
        ]

        scores = {
            label: round(float(score), 3)
            for label, score in zip(LABELS, result["scores"])
        }

        print("\n" + "-" * 76)
        print(f"{result['id']} | {result['lang']}")
        print(f"Text: {result['text']}")
        print(f"Expected:  {expected_names or ['none']}")
        print(f"Predicted: {predicted_names or ['none']}")
        print(f"Scores:    {scores}")
        print("Result:    " + ("PASS" if result["exact_match"] else "FAIL"))


# ==================================================
# MAIN
# ==================================================

def main():
    print(f"Project NLP directory: {ROOT}")
    print(f"Device: {DEVICE}")

    cases = load_cases()
    print(f"Loaded {len(cases)} test cases.")

    models = {
        "v1": load_model("text_v1"),
        "v2": load_model("text_v2"),
    }

    summaries = {}
    all_results = {}

    for version, model in models.items():
        summary, results = evaluate_model(model, cases)
        summaries[version] = summary
        all_results[version] = results
        print_results(summary, results)

    print("\n" + "=" * 76)
    print("FINAL COMPARISON")
    print("=" * 76)

    for version in ["v1", "v2"]:
        s = summaries[version]
        print(
            f"{version.upper()}: {s['exact_matches']}/{s['total_cases']} "
            f"exact matches | exact-match accuracy "
            f"{s['exact_match_accuracy']:.1%} | "
            f"label accuracy {s['label_accuracy']:.1%}"
        )

    report = {
        "device": str(DEVICE),
        "case_count": len(cases),
        "summaries": summaries,
        "results": all_results,
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nReport saved to: {REPORT_FILE}")
    print(
        "Note: This hand-authored test set is diagnostic; "
        "it is not a statistically reliable benchmark."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise
