
import base64
import hashlib
import hmac
import json
import os
import re
import tempfile
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken


ROOT = Path(__file__).parent
DATA_FILE = ROOT / "verification_data.enc"

PBKDF2_ITERATIONS = 600_000


def _get_fernet():
    key = os.environ.get("RV_VERIFICATION_KEY")

    if not key:
        raise RuntimeError(
            "RV_VERIFICATION_KEY is not set. "
            "Configure the encryption key before using verification storage."
        )

    try:
        return Fernet(key.encode("utf-8"))
    except Exception as exc:
        raise RuntimeError("Invalid verification encryption key.") from exc


def _validate_name(name):
    if not isinstance(name, str) or not re.fullmatch(
        r"[A-Za-z0-9_-]+", name
    ):
        raise ValueError("Invalid contact name.")

    return name


def _load():
    if not DATA_FILE.exists():
        return {}

    try:
        encrypted = DATA_FILE.read_bytes()
        decrypted = _get_fernet().decrypt(encrypted)
        data = json.loads(decrypted.decode("utf-8"))

        if not isinstance(data, dict):
            raise ValueError("Invalid verification data format.")

        return data

    except InvalidToken as exc:
        raise RuntimeError(
            "Could not decrypt verification data. "
            "Check that the original encryption key is available."
        ) from exc


def _save(data):
    ROOT.mkdir(parents=True, exist_ok=True)

    plaintext = json.dumps(
        data, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")

    encrypted = _get_fernet().encrypt(plaintext)

    fd, temporary_path = tempfile.mkstemp(
        prefix="verification_", suffix=".tmp", dir=ROOT
    )

    try:
        with os.fdopen(fd, "wb") as f:
            f.write(encrypted)

        os.replace(temporary_path, DATA_FILE)

    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _normalize_answer(answer):
    return " ".join(answer.strip().casefold().split())


def _hash_answer(answer, salt):
    normalized = _normalize_answer(answer)

    return hashlib.pbkdf2_hmac(
        "sha256",
        normalized.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
        dklen=32,
    )


def add_verification(contact, question, answer):
    contact = _validate_name(contact)

    if not isinstance(question, str) or not question.strip():
        raise ValueError("A verification question is required.")

    if not isinstance(answer, str) or len(answer.strip()) < 4:
        raise ValueError("The answer must contain at least 4 characters.")

    question = question.strip()
    salt = os.urandom(16)
    answer_hash = _hash_answer(answer, salt)

    data = _load()

    if contact in data:
        raise ValueError(
            f"Verification information already exists for {contact}."
        )

    data[contact] = {
        "question": question,
        "salt": base64.b64encode(salt).decode("ascii"),
        "answer_hash": base64.b64encode(answer_hash).decode("ascii"),
        "iterations": PBKDF2_ITERATIONS,
    }

    _save(data)


def get_question(contact):
    contact = _validate_name(contact)
    record = _load().get(contact)

    if not record:
        return None

    return record["question"]


def list_verified_contacts():
    return sorted(_load().keys())


def verify_answer(contact, supplied_answer):
    """
    Compare an answer locally.
    The expected answer is never returned.
    """

    contact = _validate_name(contact)
    record = _load().get(contact)

    if not record:
        return False

    try:
        salt = base64.b64decode(record["salt"])
        expected = base64.b64decode(record["answer_hash"])

        supplied = _hash_answer(supplied_answer, salt)

        return hmac.compare_digest(expected, supplied)

    except (ValueError, TypeError):
        return False


def delete_verification(contact):
    contact = _validate_name(contact)
    data = _load()

    if contact not in data:
        return False

    del data[contact]
    _save(data)

    return True
