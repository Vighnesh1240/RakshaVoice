# nlp/build_chunk_dataset.py   (run from rakshavoice/)
import re
import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

SRC, OUT = "nlp/data/auto_labeled.csv", "nlp/data/chunks"
L, WIN = ["urgency", "authority", "otp_pii", "payment"], 2

src = open("nlp/auto_label.py", encoding="utf-8").read().split("def main():")[0]
src = re.sub(r'(INPUT_FILE|OUTPUT_FILE)\s*=.*', '', src)
ns = {}; exec(src, ns)                       # reuse your regex rules without running main()
R, cany, cauth, cotp = ns["RULES"], ns["contains_any"], ns["contains_authority"], ns["contains_otp_pii"]
def regex_labels(t):
    return [int(cany(t, R["urgency"])), int(cauth(t)), int(cotp(t)), int(cany(t, R["payment"]))]

def chunks(text):
    s = [x.strip() for x in re.split(r'(?<=[.!?\u0964])\s+', str(text).strip()) if x.strip()]
    out = [" ".join(s[i:i+WIN]) for i in range(0, len(s), WIN)]
    return [c for c in out if len(c.split()) >= 4]

df = pd.read_csv(SRC)
rows = [[r["id"], k, c, r["final_verdict"]] + regex_labels(c)
        for _, r in df.iterrows() for k, c in enumerate(chunks(r["text"]))]
ch = pd.DataFrame(rows, columns=["conv_id", "pos", "text", "verdict"] + L)
ch = ch.drop_duplicates(["conv_id", "text"]).reset_index(drop=True)

tr_i, rest_i = next(GroupShuffleSplit(1, test_size=0.30, random_state=0).split(ch, groups=ch.conv_id))
train, rest = ch.iloc[tr_i], ch.iloc[rest_i]
va_i, te_i = next(GroupShuffleSplit(1, test_size=0.50, random_state=0).split(rest, groups=rest.conv_id))
val, test_pool = rest.iloc[va_i], rest.iloc[te_i]
os.makedirs(OUT, exist_ok=True)
train.to_csv(f"{OUT}/train_regex.csv", index=False); val.to_csv(f"{OUT}/val_regex.csv", index=False)

def pick(pool, l, n):
    p = pool[pool[l] == 1]; return p.sample(min(n, len(p)), random_state=0)

def sample_n(pool, n):
    return pool.sample(min(n, len(pool)), random_state=0)

negs = test_pool[test_pool[L].sum(axis=1) == 0]
gold = pd.concat([pick(test_pool, l, 45) for l in L] +
                 [sample_n(negs[negs["verdict"] == "NO"], 90),
                  sample_n(negs[negs["verdict"] == "YES"], 60)],
                 ignore_index=True)
gold = gold.drop_duplicates("text").sample(frac=1, random_state=1).reset_index(drop=True)
for l in L: gold[l + "_regex"] = gold[l]; gold[l] = ""      # you fill 0/1
gold.insert(0, "gid", range(1, len(gold) + 1))
gold.to_csv(f"{OUT}/gold_to_label.csv", index=False)
print(len(train), len(val), len(test_pool), "gold rows:", len(gold))