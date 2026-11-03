
import sys
from pathlib import Path

import numpy as np
import torch

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusion.live import LiveFusion

fusion = LiveFusion()
if not fusion.available:
    raise RuntimeError(f"Fusion model unavailable: {fusion.error}")

def test_sequence(name, rows):
    state = fusion.new_state()
    print(f"\n{name}")
    for i, row in enumerate(rows, 1):
        result = fusion.score(
            state,
            spoof_logit=-9.0,
            intent_labels={
                "urgency": row[0],
                "authority": 0.02,
                "otp_pii": row[1],
                "payment": 0.02,
            },
            voice=None,
        )
        print(
            f"window={i}, urgency={row[0]:.3f}, "
            f"otp={row[1]:.3f}, risk={result['risk']:.4f}, "
            f"alert={result['alert']}"
        )

# Compare one strong OTP window with two consecutive strong OTP windows.
test_sequence("Single strong OTP window", [(0.07, 0.9814)])
test_sequence("Two strong OTP windows", [(0.07, 0.9814), (0.07, 0.9814)])

# Approximate the observed Hindi sequence.
test_sequence("Observed Hindi OTP pattern", [(0.6169, 0.15), (0.0674, 0.9814)])
