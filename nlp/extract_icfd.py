
import json
import zstandard

INPUT_FILE = r"C:\Users\asus\Downloads\train-00000-of-00010.jsonl.zst"
OUTPUT_FILE = r"D:\RakshaVoice\nlp\data\icfd_sample.jsonl"

count = 0

with zstandard.open(INPUT_FILE, "rt", encoding="utf-8") as f_in:
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue

            record = json.loads(line)

            # Keep agent speech only; preserve metadata for later labelling.
            agent_text = " ".join(
                turn.get("text", "").strip()
                for turn in record.get("transcript", [])
                if turn.get("speaker", "").lower() == "agent"
                and turn.get("text", "").strip()
            )

            if not agent_text:
                continue

            row = {
                "id": record.get("session_id", f"icfd_{count}"),
                "text": agent_text,
                "final_verdict": record.get("final_verdict", ""),
                "scenario_id": record.get("scenario", {}).get("scenario_id"),
                "scenario_type": record.get("scenario", {}).get("case_type", ""),
                "source": "icfd_train"
            }

            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

print(f"Extracted {count} records.")
print(f"Saved to: {OUTPUT_FILE}")
