import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

FEATS = ["spoof", "urgency", "authority", "otp_pii", "payment", "voice_mismatch", "voice_active"]
D, T_MAX, STEP_S = len(FEATS), 24, 2.0           # features, max windows per session, seconds between windows


def group_of(person):
    p = str(person).lower()
    return "me" if p == "me" else "brother" if p == "brother" else "tts" if p.startswith("tts") else "media"


def real_sessions(df):
    """df from fusion.load_sessions.load(). Voice features are masked (set to 0): none of the real sessions was a 'caller claims to be X' scenario."""
    out = []
    for f, g in df[~df.silent].groupby("file"):
        g = g.sort_values("step").head(T_MAX)
        x = np.zeros((T_MAX, D), dtype=np.float32); n = len(g)
        x[:n, 0] = np.clip(g.spoof_logit.values / 10.0, -4, 4)
        for j, c in enumerate(["urgency", "authority", "otp_pii", "payment"], start=1):
            x[:n, j] = g[c].fillna(0).values
        m = np.zeros(T_MAX, dtype=np.float32); m[:n] = 1
        label = g.label.iloc[0]
        if label == "normal":
            y_value = 0
        elif label == "scam":
            y_value = 1
        else:
            raise ValueError(f"Unknown ground-truth label: {label}")
        y = y_value
        onset = int(np.argmax(g[["urgency", "authority", "otp_pii", "payment"]].fillna(0).max(axis=1).values >= 0.3)) if y else T_MAX
        ys = np.zeros(T_MAX, dtype=np.float32)
        if y: ys[onset:n] = 1
        out.append({"file": f, "x": x, "m": m, "y": y, "ys": ys, "group": group_of(g.person.iloc[0]), "n": n})
    return out


KINDS = ["normal_human", "normal_synth", "normal_media", "scam_human", "scam_clone", "scam_voice", "normal_relative"]
PROBS = np.array([0.28, 0.12, 0.12, 0.20, 0.17, 0.06, 0.05]); PROBS = PROBS / PROBS.sum()
K_PERSIST = 2                                      # an alert needs this many consecutive windows (about 4 s) above the threshold


def persist(a, k=K_PERSIST):
    """True only where the last k windows were all above the threshold."""
    a = np.asarray(a, dtype=bool); out = a.copy()
    for j in range(1, k):
        shifted = np.zeros_like(a)
        if len(a) > j: shifted[j:] = a[:-j]
        out &= shifted
    return out


def simulate(n, seed=0):
    rng = np.random.RandomState(seed)
    X = np.zeros((n, T_MAX, D), dtype=np.float32); M = np.zeros((n, T_MAX), dtype=np.float32)
    Y = np.zeros(n, dtype=np.float32); YS = np.zeros((n, T_MAX), dtype=np.float32)
    for i in range(n):
        kind = rng.choice(KINDS, p=PROBS); T = rng.randint(10, T_MAX + 1)
        scam = kind.startswith("scam"); synth = kind in ("scam_clone", "normal_synth"); media = kind == "normal_media"
        off = rng.uniform(1, 6) if synth else rng.uniform(-8, 8) if media else rng.uniform(-10, 3)   # speaker/channel offset of the spoof logit
        spoof = off + rng.normal(0, 3, T)
        intent = rng.beta(0.8, 10, (T, 4)) if media else rng.beta(0.6, 14, (T, 4))                   # noise floor of the intent model (real calls are not clean)
        rate = 0.06 if media else 0.03
        for t in range(T):                                                  # false spikes in any call (misheard words, odd phrases)
            if rng.rand() < rate: intent[t, rng.randint(4)] = rng.uniform(0.2, 0.85)
        onset = T
        if scam:
            onset = rng.randint(0, max(1, T // 2)); weak = rng.rand() < 0.15      # 15 % of scam calls: speech-to-text garbled, signals stay weak
            for e in range(rng.randint(1, 4)):
                s = onset if e == 0 else rng.randint(onset, T)
                ln = rng.randint(2, 6); labs = rng.choice(4, size=rng.randint(1, 3), replace=False)
                for l in labs:
                    v = rng.beta(3, 3, size=len(intent[s:s + ln, l])); intent[s:s + ln, l] = np.maximum(intent[s:s + ln, l], v * (0.5 if weak else 1.0))
        vm = np.zeros(T); va = np.zeros(T)
        if kind in ("scam_voice", "normal_relative"):
            sim = (rng.normal(0.1, 0.1) if kind == "scam_voice" else rng.normal(0.7, 0.1)) + rng.normal(0, 0.05, T)
            vm = 1 - np.clip(sim, 0, 1); va = np.ones(T)
        X[i, :T, 0] = np.clip(spoof / 10.0, -4, 4); X[i, :T, 1:5] = intent; X[i, :T, 5] = vm; X[i, :T, 6] = va
        M[i, :T] = 1; Y[i] = float(scam)
        if scam: YS[i, onset:T] = 1
    return X, M, Y, YS


def rule_scores(x, m, kind):
    """No-training baselines (with the same persistence rule). Returns the per-window alert array for one session."""
    n = int(m.sum()); z = x[:n, 0] * 10; it = x[:n, 1:5].max(axis=1)
    if kind == "spoof_only": a = z > 0
    elif kind == "intent_only": a = it >= 0.5
    else: a = (z > 0) | (it >= 0.5)
    return persist(a)


def session_metrics(sessions, alert_fn):
    scam = [s for s in sessions if s["y"] == 1]; norm = [s for s in sessions if s["y"] == 0]
    det = [a.any() for a in (alert_fn(s) for s in scam)]; fa = [a.any() for a in (alert_fn(s) for s in norm)]
    tta = [int(np.argmax(a)) * STEP_S for a in (alert_fn(s) for s in scam) if a.any()]
    return {"n_scam": len(scam), "detected_%": round(100 * float(np.mean(det)), 0) if det else None, "n_normal": len(norm),
            "false_alarm_%": round(100 * float(np.mean(fa)), 0) if fa else None, "median_alert_s": float(np.median(tta)) if tta else None}