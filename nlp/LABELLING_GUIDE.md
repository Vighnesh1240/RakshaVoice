# RakshaVoice — NLP Labelling Guide

## 1. Purpose

This guide defines the four multi-label categories used to identify scam-related intent in Hindi, English, and Hinglish transcript chunks.

Each transcript is labelled independently for all four categories.

## 2. Label Definitions

### 2.1 Urgency (`urgency`)

**Definition:** The caller creates pressure to act immediately by threatening a consequence, imposing a deadline, or claiming that urgent action is required.

**Positive examples (label = 1):**
1. "Your account will be blocked today if you do not verify immediately."
2. "Aaj hi payment karo, warna aapka account band ho jayega."

**Negative example (label = 0):**
- "Your account statement is available whenever you need it."

### 2.2 Authority (`authority`)

**Definition:** The caller claims or implies official authority or impersonates an institution or an authorized person to influence the recipient's actions.

**Positive examples (label = 1):**
1. "This is an officer from the cybercrime department. You must follow my instructions."
2. "Main bank ke fraud department se bol raha hoon; verification ke liye meri baat maaniye."

**Negative example (label = 0):**
- "Your bank branch will remain open until 4 PM."

Merely mentioning a bank, police department, or other institution is not sufficient for a positive label.

### 2.3 OTP or Personal Information Request (`otp_pii`)

**Definition:** The caller requests an OTP, password, PIN, CVV, account credentials, identity details, or other sensitive personal or financial information.

**Positive examples (label = 1):**
1. "Please tell me the OTP you just received on your phone."
2. "Apna Aadhaar number aur bank ka PIN mujhe bata dijiye."

**Negative example (label = 0):**
- "Never share your OTP or PIN with anyone."

### 2.4 Payment Pressure (`payment`)

**Definition:** The caller demands or pressures the recipient to transfer money, make a payment, pay a fee, or send funds under suspicious or coercive circumstances.

**Positive examples (label = 1):**
1. "Transfer ₹10,000 immediately to this account to avoid legal action."
2. "Verification complete karne ke liye abhi is UPI ID par paise bhejo."

**Negative example (label = 0):**
- "Your electricity bill has been paid successfully."

## 3. Annotation Rules

1. Every row is multi-label and must contain all four binary labels: `urgency`, `authority`, `otp_pii`, and `payment`.
2. A normal call has all four labels set to `0`.
3. Label what the caller actually says, not what the annotator suspects or assumes.
4. "Your account will be blocked today" indicates urgency.
5. Merely mentioning a bank does not indicate authority impersonation.
6. A genuine call that asks for no secrets and no suspicious payment is normal, even if it mentions a bank or OTP.
7. A warning such as "Never share your OTP" is not an OTP request.
8. Multiple labels may be `1` for the same transcript when the caller's actual words support multiple categories.
9. Do not infer a label solely from the caller's identity, the topic of conversation, or an assumed hidden intention.
10. If the transcript does not provide sufficient evidence for a category, label that category `0`.

## 4. CSV Format

The dataset must contain these columns in this exact order:

`id,text,lang,urgency,authority,otp_pii,payment,source`

| Column | Description |
|---|---|
| `id` | Unique identifier for the row |
| `text` | Transcript chunk |
| `lang` | `hindi`, `english`, or `hinglish` |
| `urgency` | Binary label: `0` or `1` |
| `authority` | Binary label: `0` or `1` |
| `otp_pii` | Binary label: `0` or `1` |
| `payment` | Binary label: `0` or `1` |
| `source` | `real` or `synthetic` |

## 5. Example CSV Rows

```csv
id,text,lang,urgency,authority,otp_pii,payment,source
1,Your account will be blocked today,english,1,0,0,0,real
2,Please tell me your OTP,english,0,0,1,0,real
3,Your bank branch opens at 10 AM,english,0,0,0,0,real
4,Aaj hi paise bhejo warna account band hoga,hinglish,1,0,0,1,synthetic
5,Never share your OTP with anyone,english,0,0,0,0,real
