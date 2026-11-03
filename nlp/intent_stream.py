import json
from pathlib import Path
import torch, torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
import re
_MONEY_CUE = re.compile(r"rupee|rupaye|rupe\b|rupiye|\brs\b|₹|रुपये|रुपए|रुपया|paise|paisa|पैसे|पैसा|\bfees?\b|फीस|शुल्क|\bfine\b|jurmana|जुर्माना|penalty|deposit|\bjama\b|जमा|transfer|ट्रांसफर|\bpay\b|payment|paying|भुगतान|\bbhar\b|bharni|bharna|bhariye|भरना|भर दी|भर दो|भर लो|\bsend money\b|\bupi\b|यूपीआई|\bqr\b|क्यूआर|charges?\b|\bamount\b|राशि|challan|चालान|\bbail\b|जमानत|\brefund\b|रिफंड|\btax\b|savings|safe account|सुरक्षित खाते|surakshit khate|bachat|बचत|\bmoney\b|thousand|hundred|lakh|\bhaz(?:aa)?r\b|\bhajar\b|हज़ार|हजार|लाख|\bsau\b", re.I)

# Conservative fallback for common Hinglish ASR errors.
# Requires both account-threat context and a request to disclose a code.
_OTP_THREAT_CUE = re.compile(
    r"\b(?:account|bank account|khaata|khata|account)\b"
    r".{0,45}\b(?:block|band|blocked|close|freeze|bandh)\b"
    r".{0,100}\b(?:otp|o\s*t\s*p|one time password|"
    r"hoti(?:\s+bhi)?|bata(?:iye|ie|o)?|batana|"
    r"share|tell|provide)\b",
    re.I,
)

ROOT = Path(__file__).parent / "models" / "intent_v4"
cfg = json.load(open(ROOT / "config.json"))
LABELS, THR, MAXLEN = cfg["labels"], cfg["thresholds"], cfg["max_length"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEV == "cuda" else torch.float32


class Net(nn.Module):                       # same architecture as the Kaggle training notebook
    def __init__(self, encoder):
        super().__init__()
        self.enc = encoder
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(encoder.config.hidden_size, len(LABELS))

    def forward(self, ids, mask):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1)                       # mean pooling
        return self.head(self.drop(pooled.to(self.head.weight.dtype)))


_tok = AutoTokenizer.from_pretrained(cfg["encoder"])
_model = Net(AutoModel.from_config(AutoConfig.from_pretrained(cfg["encoder"])))   # architecture only
_sd = torch.load(ROOT / "model_fp16.pt", map_location="cpu")
_model.load_state_dict({k: v.float() for k, v in _sd.items()})
_model = _model.to(DTYPE).to(DEV).eval() 
if DEV == "cuda": torch.cuda.empty_cache()


@torch.no_grad()
def _probs(texts):
    enc = _tok(texts, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to(DEV)
    return torch.sigmoid(_model(enc["input_ids"], enc["attention_mask"]).float()).cpu().numpy()



def score_text(text):
    """Returns one stream message in the shared format."""
    p = _probs([text])[0].copy()

    if "payment" in LABELS and not _MONEY_CUE.search(text):
        p[LABELS.index("payment")] = min(
            float(p[LABELS.index("payment")]), 0.10
        )

    # Targeted recovery for likely OTP requests after an account threat.
    if (
        "otp_pii" in LABELS
        and _OTP_THREAT_CUE.search(text)
    ):
        p[LABELS.index("otp_pii")] = max(
            float(p[LABELS.index("otp_pii")]), THR[LABELS.index("otp_pii")]
        )

    labels = {
        label: round(float(value), 4)
        for label, value in zip(LABELS, p)
    }
    fired = [
        label
        for label, value, threshold in zip(LABELS, p, THR)
        if value >= threshold
    ]

    return {
        "stream": "intent",
        "score": round(float(p.max()), 4),
        "labels": labels,
        "evidence": fired,
    }


score_text_stream = score_text
