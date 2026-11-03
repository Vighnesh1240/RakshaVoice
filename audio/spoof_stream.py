import json, os
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

SR, LEN = 16000, 64000
MODELS = Path(__file__).parent / "models"
dev = os.environ.get("RV_SPOOF_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", dev)

fe = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
w2v = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(dev).eval()


class SpoofHead(nn.Module):
    def __init__(self, L=13, D=768):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(L))
        self.net = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 1))

    def forward(self, x):
        a = torch.softmax(self.w, 0)[None, :, None]
        return self.net((x * a).sum(1)).squeeze(-1)


head = SpoofHead().to(dev)
head.load_state_dict(torch.load(MODELS / "spoof_head.pt", map_location=dev))
head.eval()

_cal_path = MODELS / "calibration_v2.json"
CAL = json.load(open(_cal_path)) if _cal_path.exists() else {"a": 0.5}
A = float(CAL.get("a", 0.5))              # slope of the calibration (Platt scaling on the logit)
SHIFT = float(CAL.get("shift", 0.0))      # the logit that should count as "50 % synthetic"; set it after choosing the threshold


@torch.no_grad()
def score_logit(wav):
    """Raw head output. Large negative = human-like, large positive = synthetic-like."""
    wav = np.asarray(wav, dtype=np.float32)
    if len(wav) < LEN:
        wav = np.pad(wav, (0, LEN - len(wav)))
    x = fe([wav[:LEN]], sampling_rate=SR, return_tensors="pt").input_values.to(dev)
    hs = w2v(x, output_hidden_states=True).hidden_states
    f = torch.stack([h.mean(1) for h in hs], 1).float()
    return float(head(f).item())


def score_chunk(wav):
    """Calibrated probability (0-1) that the audio is synthetic. Same name and range as before, so the backend is unchanged."""
    z = score_logit(wav)
    return float(1.0 / (1.0 + np.exp(-A * (z - SHIFT))))