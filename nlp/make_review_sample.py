
import pandas as pd

INPUT_FILE = r"D:\RakshaVoice\nlp\data\label_review.csv"
OUTPUT_FILE = r"D:\RakshaVoice\nlp\data\label_review_sample.csv"

df = pd.read_csv(INPUT_FILE)
sample = df.sample(n=100, random_state=42)
sample.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print(f"Review sample created: {len(sample)} records")
print(f"Saved to: {OUTPUT_FILE}")
