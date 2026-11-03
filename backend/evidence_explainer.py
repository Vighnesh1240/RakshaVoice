
EVIDENCE_LABELS = {
    "spoof": "Possible synthetic or manipulated voice signal",
    "urgency": "Urgent or pressuring language",
    "authority": "Possible authority impersonation",
    "otp_pii": "Request for OTP or personal information",
    "payment": "Payment-related request",
    "voice_mismatch": "Voice does not sufficiently match the enrolled contact",
}


def explain_evidence(fusion: dict) -> dict:
    """
    Generate a user-friendly explanation from fusion evidence.
    Does not access or store audio or transcripts.
    """

    if not fusion or not fusion.get("available"):
        return {
            "available": False,
            "title": "Explanation unavailable",
            "summary": "The fusion model is currently unavailable.",
            "recommended_action": "Use caution and verify suspicious requests independently.",
            "signals": [],
        }

    evidence = fusion.get("evidence") or []
    signals = [
        EVIDENCE_LABELS[item]
        for item in evidence
        if item in EVIDENCE_LABELS
    ]

    if fusion.get("alert"):
        title = "Elevated call risk detected"

        if signals:
            summary = (
                "The recent call windows showed the following suspicious "
                "indicators: " + "; ".join(signals) + "."
            )
        else:
            summary = (
                "The temporal fusion model detected elevated risk across "
                "recent call windows."
            )

        action = (
            "Do not share OTPs, PINs, passwords or financial information. "
            "Verify the caller independently using an official contact number."
        )

    elif signals:
        title = "Potentially suspicious signals"

        summary = (
            "Some indicators were detected in the latest call window: "
            + "; ".join(signals) + "."
        )

        action = (
            "Remain cautious. Do not act on urgent payment or information "
            "requests without independent verification."
        )

    else:
        title = "No strong scam indicators detected"

        summary = (
            "The model has not detected thresholded scam evidence in "
            "the latest call window."
        )

        action = (
            "Continue using normal caution. A low model risk score "
            "does not guarantee that a call is safe."
        )

    return {
        "available": True,
        "title": title,
        "summary": summary,
        "recommended_action": action,
        "signals": signals,
    }
