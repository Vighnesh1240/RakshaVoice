
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

MODEL_DIR = Path(__file__).parent / "models" / "text_v1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiLabelHead(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.network(x)


def main():
    with open(MODEL_DIR / "config.json", encoding="utf-8") as f:
        config = json.load(f)

    checkpoint = torch.load(
        MODEL_DIR / "classifier.pt",
        map_location="cpu",
        weights_only=True,
    )

    labels = config["labels"]
    thresholds = np.asarray(config["thresholds"], dtype=np.float32)

    head = MultiLabelHead(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        output_dim=len(checkpoint["labels"]),
    )
    head.load_state_dict(checkpoint["state_dict"])
    head.to(DEVICE).eval()

    with np.load(MODEL_DIR / "normalization.npz") as data:
        mean = torch.tensor(data["mean"], dtype=torch.float32, device=DEVICE)
        std = torch.tensor(data["std"], dtype=torch.float32, device=DEVICE)

    print(f"Loading MuRIL on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(config["encoder"])
    encoder = AutoModel.from_pretrained(config["encoder"]).to(DEVICE).eval()

    def score_text(text):
        tokens = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=config["max_length"],
            padding=True,
        )
        tokens = {k: v.to(DEVICE) for k, v in tokens.items()}

        with torch.inference_mode():
            output = encoder(**tokens)
            embedding = output.last_hidden_state[:, 0, :]
            embedding = (embedding - mean) / std.clamp_min(1e-6)
            probabilities = torch.sigmoid(head(embedding))[0].cpu().numpy()

        return {
            label: {
                "score": round(float(probability), 4),
                "detected": bool(probability >= threshold),
            }
            for label, probability, threshold in zip(
                labels, probabilities, thresholds
            )
        }

    


    test_file = Path(__file__).parent / "sanity_test_cases.json"
    with open(test_file, encoding="utf-8") as f:
        cases = json.load(f)

    for i, case in enumerate(cases, start=1):
        text = case["text"]
        expected = set(case["expected"])
        predictions = score_text(text)

        detected = {
            label for label, result in predictions.items()
            if result["detected"]
        }

        print(f"\n--- Test {i} ---")
        print("Text:", text)
        print("Expected:", sorted(expected) or ["none"])
        print("Detected:", sorted(detected) or ["none"])

        for label, result in predictions.items():
            print(
                f"  {label}: score={result['score']:.4f}, "
                f"detected={result['detected']}"
            )

        print("Exact match:", detected == expected)



if __name__ == "__main__":
    main()
