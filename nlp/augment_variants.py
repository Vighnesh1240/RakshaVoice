import re, random, glob, pandas as pd
random.seed(0)
L = ["urgency", "authority", "otp_pii", "payment"]
VAR = {
 r"\bjayega\b": ["jaega", "jaayega", "jayegaa"],   r"\bjayegi\b": ["jaegi", "jaayegi"],
 r"\bbataiye\b": ["bataie", "batayiye", "bataiyae", "bataiye"],  r"\bbatao\b": ["bata do", "bataao"],
 r"\bdijiye\b": ["dijie", "dijiyega", "deejiye"], r"\bkijiye\b": ["kijie", "keejiye", "kijiyega"],
 r"\bbhejiye\b": ["bhejie", "bhejiyega"],
 r"\brupaye\b": ["rupe", "rupay", "rupiye", "rs"], r"\bpaanch\b": ["panch", "paach", "paanch"],
 r"\bhazaar\b": ["hazar", "hajar", "hajaar"],  r"\baapka\b": ["apka", "aapka", "ap ka"], r"\baapki\b": ["apki", "aapki"],
 r"\baapke\b": ["apke", "aapke"],  r"\bhoon\b": ["hun", "hu", "hoon"], r"\bmein\b": ["me", "mai", "mein"],
 r"\bwarna\b": ["varna", "warna", "nahi to"], r"\bjaldi\b": ["jldi", "jaldi", "jaldee"],
 r"\bkhaata\b": ["khata", "khaata"], r"\babhi\b": ["abhi", "abhee", "abi"],
}
def variant(t, p=0.7):
    out = t
    for pat, alts in VAR.items():
        if re.search(pat, out, re.I) and random.random() < p:
            out = re.sub(pat, random.choice(alts), out, flags=re.I)
    return out
def asr_style(t):
    t = re.sub(r"[.,?!]", "", t)
    return random.choice([t.lower(), t.lower(), t])

files = [f for f in glob.glob("nlp/data/synthetic/batch*_*.csv") if "reviewed" not in f]
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df = df[df.lang == "hinglish"]
rows = []
for _, r in df.iterrows():
    for k in range(2):
        v = asr_style(variant(r.text))
        if v != r.text: rows.append({**r.to_dict(), "text": v, "source": "synthetic_variant"})
out = pd.DataFrame(rows).drop_duplicates("text")
out = out[~out.text.isin(df.text)]
out.to_csv("nlp/data/synthetic/variants.csv", index=False)
print("hinglish source rows:", len(df), "-> variants:", len(out), out[L].sum().to_dict())