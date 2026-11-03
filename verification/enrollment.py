
from getpass import getpass

from voice.voice_stream import list_enrolled
from verification.storage import (
    add_verification,
    list_verified_contacts,
)


def main():
    contacts = list_enrolled()

    if not contacts:
        print("No voice-enrolled contacts found.")
        print("Enroll a test contact using your existing enroll_cli.py first.")
        return

    print("\nVoice-enrolled contacts:")
    for index, name in enumerate(contacts, start=1):
        status = (
            "Already registered"
            if name in list_verified_contacts()
            else "Not registered"
        )
        print(f"{index}. {name} — {status}")

    try:
        selected = int(input("\nSelect contact number: ").strip())
        if selected < 1 or selected > len(contacts):
            raise ValueError

        contact = contacts[selected - 1]

    except (ValueError, IndexError):
        print("Invalid contact selection.")
        return

    if contact in list_verified_contacts():
        print("Verification information already exists for this contact.")
        print("Delete the existing record before replacing it.")
        return

    print("\nVerification enrollment consent:")
    print(
        "I consent to storing an encrypted private verification question "
        "and a protected answer for caller verification."
    )

    consent = input("Type YES to confirm consent: ").strip().upper()

    if consent != "YES":
        print("Verification enrollment cancelled.")
        return

    question = input("\nEnter a private verification question: ").strip()

    answer = getpass("Enter the expected answer (hidden): ")
    confirmation = getpass("Re-enter the expected answer: ")

    if answer != confirmation:
        print("Answers do not match. Enrollment cancelled.")
        return

    try:
        add_verification(contact, question, answer)
        print(f"\nVerification enrollment completed for {contact}.")
        print("The question is encrypted and the answer is hashed.")

    except (ValueError, RuntimeError) as exc:
        print(f"Enrollment failed: {exc}")


if __name__ == "__main__":
    main()
