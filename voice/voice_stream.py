import json, datetime, os
from pathlib import Path
import numpy as np, torch
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

ROOT = Path(__file__).parent
ENROLLED = ROOT / "enrolled"
requested_device = os.environ.get("RV_VOICE_DEVICE", "").strip().lower()
if requested_device not in {"", "cpu", "cuda"}:
    raise ValueError("RV_VOICE_DEVICE must be 'cpu' or 'cuda'.")
DEV = requested_device or ("cuda" if torch.cuda.is_available() else "cpu")

_clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                      savedir=str(ROOT / "pretrained_ecapa"),
                                      run_opts={"device": "cuda:0" if DEV == "cuda" else DEV},
                                      local_strategy=LocalStrategy.COPY)


def embed(wav):
    """wav: float32 numpy, 16 kHz mono -> L2-normalised 192-d embedding (numpy)."""
    x = torch.from_numpy(np.asarray(wav, dtype=np.float32)).unsqueeze(0).to(DEV)
    with torch.no_grad():
        e = _clf.encode_batch(x).squeeze()
    return torch.nn.functional.normalize(e, dim=0).cpu().numpy()


def similarity(a, b):
    return float(np.dot(a, b))                                   # both L2-normalised, so this is cosine similarity


def enroll(name, wavs, consent_text):
    """Store ONLY the averaged embedding (no audio). consent_text records what the enrolled person agreed to."""
    if not consent_text.strip():
        raise ValueError("Enrolment needs the enrolled person's consent text.")
    embs = np.stack([embed(w) for w in wavs])
    c = embs.mean(0); c /= np.linalg.norm(c)
    ENROLLED.mkdir(exist_ok=True)
    np.save(ENROLLED / f"{name}.npy", c)
    json.dump({"name": name, "n_clips": len(wavs), "consent": consent_text,
               "date": datetime.date.today().isoformat()}, open(ENROLLED / f"{name}.json", "w"), indent=2)


def delete_enrolled(name):
    for ext in (".npy", ".json"):
        (ENROLLED / f"{name}{ext}").unlink(missing_ok=True)


def list_enrolled():
    return sorted(p.stem for p in ENROLLED.glob("*.npy")) if ENROLLED.exists() else []


def _calibration():
    p = ROOT / "calibration.json"
    return json.load(open(p)) if p.exists() else {"a": 12.0, "c": -6.0, "threshold": 0.5}   # placeholder until Step 3 runs


def score_voice(wav, name):
    """Shared stream format. score = probability the caller is NOT the enrolled person (0 = same, 1 = different)."""
    cal = _calibration()
    sim = similarity(embed(wav), np.load(ENROLLED / f"{name}.npy"))
    p_same = 1.0 / (1.0 + np.exp(-(cal["a"] * sim + cal["c"])))
    return {"stream": "voice", "score": round(float(1.0 - p_same), 4), "similarity": round(sim, 4),
            "enrolled": name, "evidence": ["voice_mismatch"] if sim < cal["threshold"] else []}