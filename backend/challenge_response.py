
from verification.storage import (
    get_question,
    verify_answer,
    list_verified_contacts,
)


def get_challenge(contact: str) -> dict:
    """
    Retrieve the registered verification question.
    Never retrieve or expose the stored answer.
    """

    if not contact:
        return {
            "available": False,
            "status": "unavailable",
            "question": None,
            "message": "No contact selected.",
        }

    if contact not in list_verified_contacts():
        return {
            "available": False,
            "status": "not_enrolled",
            "question": None,
            "message": "This contact has no verification question registered.",
        }

    question = get_question(contact)

    if not question:
        return {
            "available": False,
            "status": "unavailable",
            "question": None,
            "message": "Verification question could not be retrieved.",
        }

    return {
        "available": True,
        "status": "pending",
        "question": question,
        "message": "Ask the caller the displayed verification question.",
    }


def evaluate_challenge(contact: str, supplied_answer: str) -> dict:
    """
    Verify the caller's response locally.
    The supplied answer is never returned or logged.
    """

    if not contact or not supplied_answer or not supplied_answer.strip():
        return {
            "status": "inconclusive",
            "verified": False,
            "message": "No usable answer was provided.",
        }

    if contact not in list_verified_contacts():
        return {
            "status": "inconclusive",
            "verified": False,
            "message": "Contact verification is not enrolled.",
        }

    is_correct = verify_answer(contact, supplied_answer)

    if is_correct:
        return {
            "status": "verified",
            "verified": True,
            "message": "The supplied answer matches the registered verification answer.",
        }

    return {
        "status": "failed",
        "verified": False,
        "message": "The supplied answer does not match the registered answer.",
    }


def mark_inconclusive() -> dict:
    return {
        "status": "inconclusive",
        "verified": False,
        "message": "Caller identity could not be verified.",
    }
