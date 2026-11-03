import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusion.live import LiveFusion


fusion = LiveFusion()
if not fusion.available:
    raise RuntimeError(f"Fusion unavailable: {fusion.error}")


def run_case(name, rows):
    state = fusion.new_state()
    print(f"\n{name}")
    for i, (urgency, otp) in enumerate(rows, 1):
        result = fusion.score(
            state,
            spoof_logit=-9.0,
            intent_labels={
                "urgency": urgency,
                "authority": 0.02,
                "otp_pii": otp,
                "payment": 0.02,
            },
            voice=None,
        )
        print(
            f"window={i}, risk={result['risk']:.4f}, "
            f"streak={result['consecutive_high_windows']}, "
            f"alert={result['alert']}"
        )


run_case(
    "Observed Hindi OTP pattern",
    [
        (0.6169, 0.15),
        (0.0674, 0.9814),
        (0.0214, 0.0109),
    ],
)
run_case(
    "Two consecutive strong OTP windows",
    [
        (0.0674, 0.9814),
        (0.0674, 0.9814),
    ],
)
