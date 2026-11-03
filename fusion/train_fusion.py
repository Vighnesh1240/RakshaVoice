import json, os, sys
import numpy as np, torch, torch.nn as nn
from fusion.load_sessions import load
from fusion.sim import FEATS, D, T_MAX, STEP_S, real_sessions, simulate, rule_scores, session_metrics, persist

torch.manual_seed(0); np.random.seed(0)
os.makedirs("fusion/models", exist_ok=True)
N_SIM, EPOCHS, REAL_REPEAT, THR = 6000, 14, 25, 0.7


class FusionGRU(nn.Module):
    """Per-window risk from the whole call so far: GRU state + causal attention over earlier windows."""
    def __init__(self, d=D, h=32):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d, h), nn.GELU())
        self.gru = nn.GRU(h, h, batch_first=True)
        self.att_w = nn.Linear(h, h); self.att_v = nn.Linear(h, 1, bias=False)
        self.out = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Dropout(0.1), nn.Linear(h, 1))

    def forward(self, x, m):
        H, _ = self.gru(self.inp(x))                                           # [B,T,h]
        s = self.att_v(torch.tanh(self.att_w(H))).squeeze(-1)                  # [B,T]
        T = x.shape[1]
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
        allow = causal[None] & m.bool()[:, None, :]
        a = torch.softmax(s[:, None, :].expand(-1, T, -1).masked_fill(~allow, -1e9), dim=-1)
        return self.out(torch.cat([H, a @ H], -1)).squeeze(-1)                 # logits [B,T]


class StaticMLP(nn.Module):
    """Baseline: looks at one window at a time, no memory of the call."""
    def __init__(self, d=D, h=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))
    def forward(self, x, m): return self.net(x).squeeze(-1)


def fit(model, X, M, YS, epochs=EPOCHS, bs=64):
    X, M, YS = map(torch.tensor, (X, M, YS)); opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(X))
        for i in range(0, len(X), bs):
            b = perm[i:i + bs]; logit = model(X[b], M[b])
            loss = (nn.functional.binary_cross_entropy_with_logits(logit, YS[b], reduction="none") * M[b]).sum() / M[b].sum()
            opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def alerts(model, s):
    model.eval(); r = torch.sigmoid(model(torch.tensor(s["x"])[None], torch.tensor(s["m"])[None]))[0].numpy()
    return persist(r[:int(s["m"].sum())] >= THR)


def stack(sessions, repeat=1):
    return (np.stack([s["x"] for s in sessions] * repeat), np.stack([s["m"] for s in sessions] * repeat), np.stack([s["ys"] for s in sessions] * repeat))


if __name__ == "__main__":
    real = real_sessions(load()); groups = ["me", "brother", "tts", "media"]
    print(len(real), "real sessions:", {g: sum(s["group"] == g for s in real) for g in groups})
    Xs, Ms, Ys, YSs = simulate(N_SIM, 0)
    rows = []
    def report(method, g, sess, fn):
        rows.append({"method": method, "test_group": g, **session_metrics(sess, fn)})

    for g in groups:
        test = [s for s in real if s["group"] == g]
        if not test: continue
        for kind in ("spoof_only", "intent_only", "either"):
            report(kind, g, test, lambda s, k=kind: rule_scores(s["x"], s["m"], k))
    print("training on simulated sessions only ...")
    for name, cls in (("MLP_sim", StaticMLP), ("GRU_sim", FusionGRU)):
        torch.manual_seed(0); mdl = fit(cls(), Xs, Ms, YSs)
        for g in groups:
            test = [s for s in real if s["group"] == g]
            if test: report(name, g, test, lambda s, m=mdl: alerts(m, s))
    print("training on simulated + real sessions (leave one group out) ...")
    for name, cls in (("MLP_sim+real", StaticMLP), ("GRU_sim+real", FusionGRU)):
        for g in groups:
            test = [s for s in real if s["group"] == g]; train = [s for s in real if s["group"] != g]
            if not test or not train: continue
            Xr, Mr, Yr = stack(train, REAL_REPEAT)
            torch.manual_seed(0); mdl = fit(cls(), np.concatenate([Xs, Xr]), np.concatenate([Ms, Mr]), np.concatenate([YSs, Yr]))
            report(name, g, test, lambda s, m=mdl: alerts(m, s))
    import pandas as pd
    res = pd.DataFrame(rows); pd.set_option("display.width", 200)
    print(res.to_string(index=False)); res.to_csv("fusion/results_fusion.csv", index=False)
    # final model for the live app: simulated + all real sessions
    Xr, Mr, Yr = stack(real, REAL_REPEAT); torch.manual_seed(0)
    final = fit(FusionGRU(), np.concatenate([Xs, Xr]), np.concatenate([Ms, Mr]), np.concatenate([YSs, Yr]))
    torch.save(final.state_dict(), "fusion/models/fusion_gru.pt")
    json.dump({"features": FEATS, "t_max": T_MAX, "threshold": THR, "step_seconds": STEP_S}, open("fusion/models/fusion_config.json", "w"), indent=1)
    print("saved fusion/models/fusion_gru.pt")
    print("\nDIAGNOSTIC: real NORMAL sessions the final model scored (max over windows)")
    final.eval()
    for s in real:
        if s["y"] == 0:
            n = int(s["m"].sum())
            with torch.no_grad(): r = torch.sigmoid(final(torch.tensor(s["x"])[None], torch.tensor(s["m"])[None]))[0].numpy()[:n]
            print(f"{s['group']:8s} {s['file'][:15]}  max_intent={s['x'][:n,1:5].max():.2f}  max_spoof_z={s['x'][:n,0].max()*10:5.1f}  max_risk={r.max():.2f}  windows_over_threshold={int((r >= THR).sum())}")