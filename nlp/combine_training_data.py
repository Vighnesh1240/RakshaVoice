# nlp/combine_training_data.py   (run from rakshavoice/)   -- version 2
import re, os, pandas as pd

L = ["urgency", "authority", "otp_pii", "payment"]
CH = "nlp/data/chunks"
OUT = "nlp/data/final"; os.makedirs(OUT, exist_ok=True)
SYN = ["batch1_short.csv", "batch2_short.csv", "batch3_authority.csv", "batch4_negatives.csv", "batch5_otp_variants.csv"]
SYN_REPEAT = 4
NEG_KEEP = 5000

CUE = re.compile(r"\b(?:provide|give|tell|share|confirm|need|require|enter|read out|send|batao|bataiye|batayiye|dena|dijiye|bata dijiye|verify)\b|बताइए|दीजिए|बता दीजिए", re.I)
SECRET = re.compile(r"\b(?:otp|one[- ]time|upi pin|pin|cvv|password|passcode|card number|card details|aadhaar|pan number|pan card|credentials)\b|ओटीपी|पिन|पासवर्ड|आधार|सीवीवी", re.I)
UPI    = re.compile(r"\bupi pin\b", re.I)
WARN   = re.compile(r"\b(?:do not|don't|never|should not|shouldn't|mat\b|nahi dena)\b.{0,40}\b(?:share|give|tell|enter|provide|batana|bata)\b", re.I)
MONEY  = re.compile(r"\b(?:pay|transfer|send|deposit)\b.{0,50}\b(?:money|amount|rupees|rs\.?|fee|fees|fine|penalty|charge|charges)\b|\benter the amount\b|\bsend (?:me )?(?:the )?(?:money|amount)\b|₹|भुगतान|ट्रांसफर|paise|rupaye", re.I)
LEGAL  = re.compile(r"legal action|legal consequences|legal proceedings|\barrest|\bwarrant\b|\bpolice\b|\bcourt\b|\bFIR\b|lawsuit|prosecut|cyber ?crime|\bCBI\b", re.I)
SELF_PROMISE = re.compile(r"\b(?:we|i)(?:'ll| will| shall| can)\b[^.?!]{0,60}\b(?:immediately|right away|turant)\b|\b(?:will be|is being|are being)\b[^.?!]{0,40}\b(?:processed|credited|refunded|unlocked|blocked|activated|resolved)\b[^.?!]{0,20}\b(?:immediately|turant)\b|\b(?:immediately|turant)\b[^.?!]{0,30}\bunlock kar den", re.I)
THREAT = re.compile(r"if you (?:do not|don't|dont) [^.?!]{0,60}(?:locked|closed|suspended|blocked|frozen|legal|consequences|penalt)|will remain locked|not a discussion|limited[- ]time", re.I)

def lang_of(t):
    return "hindi" if re.search(r"[\u0900-\u097F]", str(t)) else "latin"

def clean(d):
    d = d.copy()
    old_pay = d.payment.where(d.verdict != "NO", 0)
    d.loc[d.verdict == "NO", L] = 0
    yes = d.verdict == "YES"
    has_secret = d.text.str.contains(SECRET)

    keep_otp   = (d.otp_pii == 1) & has_secret
    upi_req    = yes & d.text.str.contains(UPI) & ~d.text.str.contains(WARN)
    pay_secret = (old_pay == 1) & has_secret & ~d.text.str.contains(WARN)
    req_secret = yes & d.text.str.contains(SECRET) & d.text.str.contains(CUE) & ~d.text.str.contains(WARN)
    d["otp_pii"] = (keep_otp | upi_req | pay_secret | req_secret).astype(int)

    u = d.urgency.copy()
    u[(u == 1) & d.text.str.contains(SELF_PROMISE)] = 0
    u[yes & d.text.str.contains(THREAT) & (u == 0)] = 1
    d["urgency"] = u

    d["authority"] = (yes & d.text.str.contains(LEGAL)).astype(int)
    d["payment"] = ((old_pay == 1) & d.text.str.contains(MONEY)).astype(int)

    d["lang"] = d.text.map(lang_of)
    d["source"] = "icfd_chunk"
    return d[["text", "lang", "source"] + L]

tr = clean(pd.read_csv(f"{CH}/train_regex.csv"))
va = clean(pd.read_csv(f"{CH}/val_regex.csv"))
print("ICFD train positives after cleaning:", tr[L].sum().to_dict())

is_pos = tr[L].sum(axis=1) > 0
pos = tr[is_pos]
neg = tr[~is_pos].sample(min(NEG_KEEP, int((~is_pos).sum())), random_state=0)

syn = []
for f in SYN:
    p = f"nlp/data/synthetic/{f}"
    if os.path.exists(p): syn.append(pd.read_csv(p))
    else: print("WARNING: missing", p)
syn = pd.concat(syn, ignore_index=True)
assert syn[L].isin([0, 1]).all().all(), "synthetic labels must be 0/1"
syn["source"] = "synthetic"
print("synthetic rows:", len(syn), syn[L].sum().to_dict())

var_path = "nlp/data/synthetic/variants.csv"
var = pd.read_csv(var_path) if os.path.exists(var_path) else syn.iloc[0:0]
var["source"] = "synthetic_variant"
train = pd.concat([pos, neg] + [syn] * SYN_REPEAT + [var], ignore_index=True).sample(frac=1, random_state=0)
train.to_csv(f"{OUT}/train_combined.csv", index=False)
va.to_csv(f"{OUT}/val_clean.csv", index=False)
print("train_combined:", len(train), train[L].sum().to_dict())
print("val_clean:", len(va), va[L].sum().to_dict())