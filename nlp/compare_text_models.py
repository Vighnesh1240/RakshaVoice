import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
CASES = ROOT / "sanity_test_cases.json"
LABELS = ["urgency", "authority", "otp_pii", "payment"]
DEVICE = torch.device("cpu")

class MultiLabelHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(256, 4),
        )

    def forward(self, x):
        return self.network(x)

def load_model(folder):
    folder = ROOT / "models" / folder
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(
        folder / "classifier.pt",
        map_location="cpu",
        weights_only=True,
    )
    head = MultiLabelHead()
    head.load_state_dict(checkpoint["state_dict"])
    head.eval()

    norm = np.load(folder / "normalization.npz")
    return head, norm["mean"], norm["std"], config

cases = json.loads(CASES.read_text(encoding="utf-8"))

tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
encoder = AutoModel.from_pretrained("google/muril-base-cased").to(DEVICE)
encoder.eval()

models = {
    "v1": load_model("text_v1"),
    "v2": load_model("text_v2"),
    "v3": load_model("text_v3"),
}

results = {name: [] for name in models}

for i, case in enumerate(cases, start=1):
    text = case["text"]
    expected = set(case["expected"])

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=192,
    )

    with torch.no_grad():
        embedding = encoder(**encoded).last_hidden_state[:, 0, :].numpy()

    print(f"\n{i}. {text}")
    print("Expected:", sorted(expected))

    for name, (head, mean, std, config) in models.items():
        normalized = (embedding - mean) / np.where(std < 1e-6, 1.0, std)

        with torch.no_grad():
            probabilities = torch.sigmoid(
                head(torch.tensor(normalized, dtype=torch.float32))
            ).numpy()[0]

        thresholds = config["thresholds"]
        predicted = {
            LABELS[j] for j in range(len(LABELS))
            if probabilities[j] >= thresholds[j]
        }

        exact = predicted == expected
        results[name].append(exact)

        scores = ", ".join(
            f"{LABELS[j]}={probabilities[j]:.2f}"
            for j in range(len(LABELS))
        )
        print(
            f"  {name}: predicted={sorted(predicted)} | "
            f"exact_match={exact} | {scores}"
        )

print("\n=== SUMMARY ===")
for name, matches in results.items():
    print(f"{name}: {sum(matches)}/{len(matches)} exact matches")

print("\nNote: These are 10 hand-written sanity cases, not a representative test set.")
