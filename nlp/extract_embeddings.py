
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "experimental" / "model_ready"
OUT_DIR = ROOT / "data" / "experimental" / "embeddings"

MODEL_NAME = "google/muril-base-cased"
MAX_LENGTH = 128
BATCH_SIZE = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = ["urgency", "authority", "otp_pii", "payment"]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Loading encoder: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    for split in ["train", "val", "test"]:
        csv_path = DATA_DIR / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing dataset: {csv_path}")

        df = pd.read_csv(csv_path)
        if df["text"].isna().any():
            raise ValueError(f"{split}: missing text values")

        embeddings = []

        print(f"\nExtracting {split}: {len(df)} records")

        for start in range(0, len(df), BATCH_SIZE):
            texts = df["text"].iloc[start:start + BATCH_SIZE].astype(str).tolist()

            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

            with torch.inference_mode():
                outputs = model(**inputs)
                # CLS-token representation: one 768-D vector per text.
                batch_embeddings = outputs.last_hidden_state[:, 0, :]

            embeddings.append(batch_embeddings.cpu().numpy().astype(np.float32))

            completed = min(start + BATCH_SIZE, len(df))
            if completed % 100 == 0 or completed == len(df):
                print(f"  Processed {completed}/{len(df)}")

        X = np.concatenate(embeddings, axis=0)
        y = df[LABELS].to_numpy(dtype=np.float32)

        if not np.isfinite(X).all():
            raise ValueError(f"{split}: non-finite embeddings found")

        np.savez_compressed(
            OUT_DIR / f"{split}_embeddings.npz",
            X=X,
            y=y,
            ids=df["id"].to_numpy(),
            labels=np.array(LABELS),
        )

        print(f"Saved: {OUT_DIR / f'{split}_embeddings.npz'}")
        print(f"Embedding shape: {X.shape}")
        print(f"Label shape: {y.shape}")

    print("\nEmbedding extraction completed.")


if __name__ == "__main__":
    main()
