import glob, json, os
import numpy as np, pandas as pd

LOG_DIR, NOTES = "logs/sessions", "logs/sessions_notes.csv"


def clean_notes(write=True):
    n = pd.read_csv(NOTES, encoding="utf-8-sig", dtype=str)
    n.columns = [c.strip().lower() for c in n.columns]
    n = n.drop(columns=[c for c in n.columns if c == "path" and n[c].str.contains(r"\\|/", regex=True, na=False).any()], errors="ignore")
    for c in n.columns:
        n[c] = n[c].str.strip()
    n["file"] = n["file"].str.replace(r"\s+", "", regex=True)               # removes stray tab characters
    n["person"] = n["person"].str.lower(); n["label"] = n["label"].str.lower()
    if "capture" not in n.columns or n["capture"].isna().any():
        t = pd.to_datetime(n.file.str[:15], format="%Y%m%d_%H%M%S")
        n["capture"] = "direct"
        key = np.where(n.person.str.startswith("tts"), "tts", n.person)
        for k in np.unique(key):
            idx = n.index[key == k]
            if k == "tts":
                n.loc[idx, "capture"] = "phone"                                  # TTS voices were played from the phone speaker
            else:
                order = idx[np.argsort(t[idx].values)]
                gaps = np.diff(t[order].values).astype("timedelta64[s]").astype(int)
                if len(gaps): n.loc[order[int(np.argmax(gaps)) + 1:], "capture"] = "phone"   # sessions after the longest pause = phone call (GUESS, please check)
        print("capture column was missing or incomplete: filled with a GUESS. Open logs/sessions_notes.csv and correct it if needed.")
    if write:
        n.to_csv(NOTES, index=False)
    return n


def load():
    notes = clean_notes()
    rows = []
    for f in sorted(glob.glob(f"{LOG_DIR}/*.jsonl")):
        for line in open(f):
            r = json.loads(line); r["file"] = os.path.basename(f); rows.append(r)
    df = pd.DataFrame(rows).rename(columns={"label": "label_in_log"})
    missing = sorted(set(notes.file) - set(df.file))
    if missing: print("notes rows without a log file:", missing)
    df = df.merge(notes[["file", "person", "label", "capture"]], on="file", how="inner")
    df["silent"] = df["silent"].fillna(False).astype(bool)
    return df


if __name__ == "__main__":
    df = load()
    print("sessions:", df.file.nunique(), "| windows:", len(df), "| label in the page differed from the notes in",
          int((df.drop_duplicates("file").label_in_log != df.drop_duplicates("file").label).sum()), "sessions")
    act = df[~df.silent]
    keys = ["person", "label", "capture"]
    out = act.groupby(keys).agg(sessions=("file", "nunique"), windows=("step", "size"), spoof_mean=("spoof_logit", "mean"), spoof_min=("spoof_logit", "min"),
                                spoof_max=("spoof_logit", "max"), urgency=("urgency", "mean"), authority=("authority", "mean"),
                                otp_pii=("otp_pii", "mean"), payment=("payment", "mean"), voice_on=("voice_active", "mean")).round(2)
    out["silent_share"] = df.groupby(keys).silent.mean().round(2)
    print(out.to_string())
    print("\nsessions per label:", df.drop_duplicates("file").label.value_counts().to_dict())