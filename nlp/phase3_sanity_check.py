
from nlp.intent_stream import score_text

TEST_CASES = [
    {
        "name": "Ordinary conversation",
        "text": "Good morning. I will attend the meeting at ten.",
    },
    {
        "name": "Educational narration",
        "text": "The earth was without form and void.",
    },
    {
        "name": "Request for class notes",
        "text": "Please send me the class notes when you have time.",
    },
    {
        "name": "Urgent OTP request",
        "text": "Your bank account will be blocked today. Share your OTP immediately.",
    },
    {
        "name": "Bank impersonation",
        "text": "I am calling from the bank. Tell me your OTP to verify your account.",
    },
    {
        "name": "Threat and payment demand",
        "text": "Transfer the money now or your account will be frozen.",
    },
    {
        "name": "Prize payment demand",
        "text": "Pay the processing fee immediately to receive your prize.",
    },
]

def main():
    for case in TEST_CASES:
        print(f"\n--- {case['name']} ---")
        print("Text:", case["text"])

        try:
            predictions = score_text(case["text"])
            for label, result in predictions.items():
                print(
                    f"{label}: score={result['score']:.4f}, "
                    f"detected={result['detected']}"
                )
        except Exception as exc:
            print("ERROR:", exc)

if __name__ == "__main__":
    main()
