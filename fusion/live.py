"""Runtime loader and rolling-window inference for RakshaVoice fusion GRU."""
from collections import deque
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn

FEATURES = ["spoof", "urgency", "authority", "otp_pii", "payment", "voice_mismatch", "voice_active"]
T_MAX = 24
THRESHOLD = 0.7
STEP_SECONDS = 2.0


class FusionGRU(nn.Module):
    """Must stay architecture-compatible with train_fusion.py."""
    def __init__(self, d=7, h=32):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d, h), nn.GELU())
        self.gru = nn.GRU(h, h, batch_first=True)
        self.att_w = nn.Linear(h, h)
        self.att_v = nn.Linear(h, 1, bias=False)
        self.out = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Dropout(0.1), nn.Linear(h, 1))

    def forward(self, x, m):
        H, _ = self.gru(self.inp(x))
        s = self.att_v(torch.tanh(self.att_w(H))).squeeze(-1)
        T = x.shape[1]
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        allow = causal[None] & m.bool()[:, None, :]
        a = torch.softmax(s[:, None, :].expand(-1, T, -1).masked_fill(~allow, -1e9), dim=-1)
        return self.out(torch.cat([H, a @ H], -1)).squeeze(-1)


class LiveFusion:
    def __init__(self, model_path=None, config_path=None):
        root = Path(__file__).resolve().parent
        model_path = Path(model_path) if model_path else root / "models" / "fusion_gru.pt"
        config_path = Path(config_path) if config_path else root / "models" / "fusion_config.json"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.available = False
        self.error = None
        self.features = FEATURES
        self.t_max = T_MAX
        self.threshold = THRESHOLD
        self.step_seconds = STEP_SECONDS
        try:
            with config_path.open() as f:
                cfg = json.load(f)
            self.features = cfg.get("features", FEATURES)
            self.t_max = int(cfg.get("t_max", T_MAX))
            self.threshold = float(cfg.get("threshold", THRESHOLD))
            self.step_seconds = float(cfg.get("step_seconds", STEP_SECONDS))
            if self.features != FEATURES:
                raise ValueError(f"Fusion feature order mismatch: {self.features}")
            self.model = FusionGRU(d=len(self.features))
            try:
                state = torch.load(model_path, map_location=self.device, weights_only=True)
            except TypeError:  # compatibility with older PyTorch releases
                state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.model.to(self.device).eval()
            self.available = True
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def new_state(self):
        return {"windows": deque(maxlen=self.t_max), "high_streak": 0, "alert": False}

    @torch.no_grad()
    def score(self, state, *, spoof_logit, intent_labels, voice=None):
        if not self.available:
            return {"available": False, "error": self.error}
        labels = intent_labels or {}
        voice_active = 1.0 if voice is not None else 0.0
        voice_mismatch = float(voice.get("score", 0.0)) if voice else 0.0
        row = np.asarray([
            np.clip(float(spoof_logit) / 10.0, -4.0, 4.0),
            float(labels.get("urgency") or 0.0),
            float(labels.get("authority") or 0.0),
            float(labels.get("otp_pii") or 0.0),
            float(labels.get("payment") or 0.0),
            np.clip(voice_mismatch, 0.0, 1.0),
            voice_active,
        ], dtype=np.float32)
        print(
            "[FUSION INPUT]",
            dict(zip(self.features, np.round(row, 4).tolist())),
            flush=True,
        )
        state["windows"].append(row)
        seq = np.zeros((self.t_max, len(self.features)), dtype=np.float32)
        n = len(state["windows"])
        seq[:n] = np.stack(state["windows"])
        mask = np.zeros(self.t_max, dtype=np.float32)
        mask[:n] = 1.0
        x = torch.from_numpy(seq).unsqueeze(0).to(self.device)
        m = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        risk = float(torch.sigmoid(self.model(x, m)[0, n - 1]).item())
        state["high_streak"] = state["high_streak"] + 1 if risk >= self.threshold else 0
        state["alert"] = state["high_streak"] >= 2
        active = []
        if row[0] >= 0.5:
            active.append("spoof")
        for name, idx in (("urgency", 1), ("authority", 2), ("otp_pii", 3), ("payment", 4)):
            if row[idx] >= 0.5:
                active.append(name)
        if voice_active and voice_mismatch >= 0.5:
            active.append("voice_mismatch")
        return {
            "available": True,
            "risk": round(risk, 4),
            "threshold": self.threshold,
            "alert": bool(state["alert"]),
            "consecutive_high_windows": int(state["high_streak"]),
            "windows_seen": n,
            "evidence": active,
            "feature_order": self.features,
        }
