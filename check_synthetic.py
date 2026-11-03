
import pandas as pd

original = pd.read_csv("nlp/data/auto_labeled.csv")
synthetic = pd.read_csv(
    "nlp/data/synthetic/synthetic_combined_reviewed.csv"
)

print("Synthetic IDs overlapping original IDs:",
      synthetic["id"].astype(str).isin(original["id"].astype(str)).sum())

print("\nSynthetic source distribution:")
print(synthetic["source"].value_counts(dropna=False))

print("\nOriginal source distribution:")
print(original["source"].value_counts(dropna=False))

print("\nSynthetic verdict distribution:")
print(synthetic["final_verdict"].value_counts(dropna=False))
