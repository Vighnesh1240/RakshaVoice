
import json
import csv

INPUT_FILE = r"D:\RakshaVoice\nlp\data\icfd_sample.jsonl"
OUTPUT_FILE = r"D:\RakshaVoice\nlp\data\label_review.csv"

COLUMNS = [
    "id",
    "text",
    "lang",
    "urgency",
    "authority",
    "otp_pii",
    "payment",
    "source",
    "final_verdict",
]

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=COLUMNS)
    writer.writeheader()

    for record in records:
        writer.writerow({
            "id": record["id"],
            "text": record["text"],
            "lang": "",
            "urgency": "",
            "authority": "",
            "otp_pii": "",
            "payment": "",
            "source": record["source"],
            "final_verdict": record["final_verdict"],
        })

print(f"Created label-review CSV with {len(records)} records.")
print(f"Saved to: {OUTPUT_FILE}")
print("Category labels are intentionally blank for review.")
