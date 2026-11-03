import pandas as pd
L = ["urgency", "authority", "otp_pii", "payment"]
D = "nlp/data/chunks"
orig = pd.read_csv(f"{D}/gold_labeled.csv", encoding="cp1252").set_index("gid")
edit = pd.read_csv(f"{D}/gold_review_edited.csv", encoding="utf-8-sig").set_index("gid")
assert edit[L].isin([0, 1]).all().all(), "every label cell in the edited sheet must be 0 or 1"

changed = (orig.loc[edit.index, L].values != edit[L].values).any(axis=1)
rep = pd.DataFrame({"why": edit.why.values, "changed": changed})
print("rows you changed, by reason:")
print(rep.groupby("why").changed.agg(["sum", "count"]).rename(columns={"sum": "changed", "count": "reviewed"}))
r = rep[rep.why == "random_check"]
print(f"\nGOLD ERROR RATE estimate: {r.changed.mean():.1%}  ({int(r.changed.sum())} of {len(r)})")

new = orig.copy()
new.loc[edit.index, L] = edit[L].values
new[L] = new[L].astype(int)
new["labeling_status"] = "human-reviewed: all positives, all model disagreements, 40 random checks"
new.reset_index().to_csv(f"{D}/gold_verified.csv", index=False)
print("\nlabel positives before:", orig[L].sum().to_dict())
print("label positives after :", new[L].sum().to_dict())