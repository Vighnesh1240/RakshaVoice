import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from auto_label import contains_any, RULES
from auto_label import contains_otp_pii
TESTS = [
    # Direct requests
    ("Please read out your full account number.", 1),
    ("Send me your OTP to complete verification.", 1),
    ("Please confirm your date of birth.", 1),

    # Protective guidance
    ("Never share your OTP with anyone.", 0),
    ("You don't need to share any documents with me.", 0),
    ("Please do not disclose your Aadhaar number.", 0),

    # Official-channel guidance
    ("Provide your name and date of birth on our official website.", 0),

    # Context-sensitive verification
    ("Can you confirm your date of birth to block your lost card?", 1),
    ("I'm not supposed to confirm your account number over the phone.", 0),
]

failures = 0

for text, expected in TESTS:
    actual = int(contains_otp_pii(text))
    status = "PASS" if actual == expected else "FAIL"

    print(f"{status}: expected={expected}, actual={actual}")
    print(f"  {text}")

    failures += actual != expected

print(f"\nResults: {len(TESTS) - failures}/{len(TESTS)} passed")