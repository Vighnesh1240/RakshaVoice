
import json
import csv
import re

INPUT_FILE = r"D:\RakshaVoice\nlp\data\icfd_sample.jsonl"
OUTPUT_FILE = r"D:\RakshaVoice\nlp\data\auto_labeled.csv"




def contains_any(text, patterns):
    text = str(text)

    # Check each sentence independently.
    sentences = re.split(r'(?<=[.!?])\s+', text)

    warning_patterns = [
        # English safety warnings
        r"\bdo not share\b",
        r"\bdon't share\b",
        r"\bnever share\b",
        r"\bshould not share\b",
        r"\bshouldn't share\b",
        r"\bdo not disclose\b",
        r"\bnever disclose\b",
        r"\bwill never ask for your\b",
        r"\bwe never ask for\b",
        r"\bwe do not ask for\b",
        r"\bdo not enter your (?:upi )?pin\b",

        # Hinglish safety warnings
        r"\bshare mat karna\b",
        r"\bmat share karo\b",
        r"\bkabhi mat batana\b",

        # Hindi: never tell anyone your OTP/PIN
        r"\u0915\u092d\u0940.*\u0913\u091f\u0940\u092a\u0940.*\u0928\u0939\u0940\u0902.*\u092c\u0924\u093e",
        r"\u0915\u093f\u0938\u0940.*\u0913\u091f\u0940\u092a\u0940.*\u0928\u0939\u0940\u0902.*\u092c\u0924\u093e",
        r"\u0913\u091f\u0940\u092a\u0940.*\u0915\u093f\u0938\u0940 \u0915\u094b.*\u0928\u0939\u0940\u0902.*\u092c\u0924\u093e",
    ]

    for sentence in sentences:
        # Ignore sentences that explicitly warn against sharing.
        if any(
            re.search(p, sentence, re.IGNORECASE)
            for p in warning_patterns
        ):
            continue

        # Check the category patterns in other sentences.
        if any(
            re.search(p, sentence, re.IGNORECASE)
            for p in patterns
        ):
            return True

    return False
def contains_authority(text):
    text = str(text)

    # Existing authority rules
    if contains_any(text, RULES["authority"]):
        return True

    # Stronger evidence: claimed authority combined with
    # a suspicious request or coercive threat.
    authority_claims = [
        r"\b(?:i am|i'm|this is) (?:a |an )?"
        r"(?:bank officer|officer\s+[a-z][a-z'-]+|"
        r"police officer|government official|inspector)\b",
    ]

    suspicious_actions = [
        r"\b(?:password|pin|otp|username|credentials|"
        r"mother's maiden name|date of birth|place of birth)\b",
        r"\b(?:transfer money|send money|pay immediately)\b",
        r"\b(?:arrest|legal action|face legal consequences|"
        r"blocked permanently|suspended permanently)\b",
    ]

    has_authority_claim = any(
        re.search(pattern, text, re.IGNORECASE)
        for pattern in authority_claims
    )

    has_suspicious_action = any(
        re.search(pattern, text, re.IGNORECASE)
        for pattern in suspicious_actions
    )

    return has_authority_claim and has_suspicious_action
def contains_otp_pii(text):
    text = str(text)

    # Split into sentences so protective language in one
    # sentence does not suppress a request in another.
    sentences = re.split(r'(?<=[.!?])\s+', text)

    protective_patterns = [
        r"\b(?:i am|i'm|we are|we're) not supposed to\b"
        r".{0,80}\b(?:confirm|request|collect|share|provide|disclose)\b",

        r"\b(?:do not|don't|never|should not|shouldn't)\b"
        r".{0,50}\b(?:share|give|provide|tell|reveal|disclose)\b",
    ]

    official_channel = (
        r"\b(?:on|through|via)\s+(?:our|the|an?)?\s*"
        r"(?:official\s+)?(?:website|mobile app|bank app)\b"
    )

    for sentence in sentences:
        # Ignore explicit statements against collecting information.
        if any(
            re.search(p, sentence, re.IGNORECASE)
            for p in protective_patterns
        ):
            continue

        # Ignore instructions to enter details through official channels.
        if re.search(official_channel, sentence, re.IGNORECASE):
            continue

        if contains_any(sentence, RULES["otp_pii"]):
            return True

    return False


RULES = {
    "urgency": [
        r"\bimmediately\b", r"\bright now\b", r"\burgent\b.{0,100}\b(?:act|action|respond|reply|pay|transfer|provide|verify|share|click|call back|confirm) immediately\b",
        r"\bwithin \d+ (?:minutes?|hours?)\b",
        r"\baccount (?:will be )?blocked\b",
        r"\bexpire(?:s|d)? today\b", r"\bfinal warning\b",
        r"\bjaldi\b", r"\babhi\b", r"\bturant\b",
        r"\bforan\b", r"\babhi ke abhi\b",
        # Hinglish urgency
        r"\b(?:das|do|teen|char|chaar|paanch)\s+minute(?:s)?\s+mein\b",
        r"\b(?:abhi|turant|jaldi)\b.{0,80}\b(?:bhejo|karo|verify|band|block)\b",
        r"\b(?:warna|nahi to|nahin to)\b.{0,80}\b(?:account|khata|band|block)\b",

        # Hindi urgency
        r"(?:दस|दो|तीन|चार|पाँच)\s+मिनट में",
        r"(?:अभी|तुरंत|जल्दी).{0,60}(?:भेजें|करें|सत्यापित|बंद)",
        r"(?:नहीं तो|वरना).{0,60}(?:खाता|खाते|बंद|ब्लॉक)",
        # English deadline and consequence patterns
        r"\bwithin \w+ minutes?\b",
        r"\bwithin \w+ hours?\b",
        r"\bunless you\b.{0,100}\b(?:verify|provide|share|confirm|respond|comply)\b",
        r"\b(?:today|immediately)\b.{0,80}\b(?:verify|provide|pay|transfer|confirm)\b",

        # Hinglish urgency
        r"\b(?:aaj hi|abhi|turant)\b.{0,60}\b(?:payment|paise|paisa|details|otp|verify|bhejo|karo)\b",
        r"\b(?:warna|nahi to|nahin to)\b.{0,80}\b(?:account|card|arrest|police|legal action|block|band)\b",
        r"\b(?:aaj hi|abhi)\b.{0,50}\b(?:payment karo|paise bhejo|details do)\b",

        # Hindi urgency
        r"(?:आज ही|अभी|तुरंत).{0,60}(?:भुगतान|पैसे|विवरण|ओटीपी|सत्यापित|भेजें|करें)",
        r"(?:वरना|नहीं तो).{0,80}(?:खाता|कार्ड|गिरफ्तार|पुलिस|कानूनी कार्रवाई|बंद)",
        r"(?:दस|दो|तीन|चार|पाँच|पंद्रह)\s+मिनट में",
        r"\u0906\u091c \u0939\u0940",
        r"\u0924\u0941\u0930\u0902\u0924",
        r"\u0928\u0939\u0940\u0902 \u0924\u094b .* \u092c\u0902\u0926 \u0939\u094b \u091c\u093e\u090f\u0917\u093e",
        r"\u0935\u0930\u0928\u093e .* \u0916\u093e\u0924\u093e .* \u092c\u0902\u0926",

    ],
    
    
    "authority": [
        # Explicit official identity claims
        r"\bi am (?:an? )?(?:officer|inspector|police officer|government official)\b",
        r"\bthis is (?:officer|inspector|police officer|government official)\b",
        r"\b(?:i am|this is) officer\s+[a-z][a-z'-]+\b",
        r"\b(?:calling|speaking) from the (?:cybercrime|cyber crime|police|law enforcement) department\b",
        r"\b(?:i am|this is) (?:calling )?from (?:the )?(?:cybercrime|cyber crime|police|law enforcement) department\b",
        r"\b(?:i am|i'm|this is) (?:a |an )?bank officer\b.{0,150}\b(?:transfer|pay|send money|face legal action|arrest|comply)\b",
        # Explicit legal or institutional authority
        r"\bby order of the court\b",
        r"\bunder government orders\b",
        r"\bthis is a mandatory (?:government|police|legal) instruction\b",
        r"\b(?:police|court|judge|government officer|tax officer|law enforcement)\b.{0,80}\b(?:investigation|arrest|warrant|legal action)\b",

        # Legal or enforcement threats used to pressure compliance
        r"\b(?:face|facing|threatened with|may face|will face) legal action\b.{0,100}\b(?:if you don't|if you do not|unless you|failure to|refuse|comply|cooperate)\b",

        r"\b(?:face|facing) arrest\b",
        r"\b(?:legal action|arrest|warrant)\b.{0,60}\b(?:if you don't|if you do not|unless you|failure to)\b",
        r"\b(?:account|card|service) (?:will be )?(?:suspended|blocked|frozen)\b.{0,80}\b(?:legal action|law|court|authority)\b",
        
        # Coercive account threats used to force compliance

        r"\baccount will be blocked if you don't\b.{0,100}\b(?:password|pin|otp|username|credentials|transfer|pay)\b",

        r"\baccount will be suspended if you don't\b",
      
        r"\baccount will be closed permanently\b",
        r"\b(?:serious|severe) legal consequences\b",
        r"\bif you don't (?:provide|verify|update|comply|cooperate)\b.{0,100}\b(?:blocked|suspended|locked|legal consequences|legal action)\b",
        r"\b(?:last warning|final warning)\b.{0,100}\b(?:provide|verify|update|comply|password|pin|details|information)\b",
        
        # Institutional authority or coercion used to pressure compliance
        r"\b(?:i am|i have|we have) (?:the )?authority to\b.{0,100}\b(?:you|your|pay|provide|comply|cooperate|report|record)\b",
        r"\b(?:avoid|prevent|unless you avoid) legal action\b",
        # Hinglish authority
        r"\bmain (?:cybercrime|cyber crime|police) department se bol raha hoon\b",
        r"\bmain (?:bank|police) (?:ke )?(?:officer|department) se bol raha hoon\b",

        # Hindi authority
        r"(?:मैं|हम).{0,50}(?:साइबर अपराध|साइबर क्राइम|पुलिस).{0,50}(?:विभाग|अधिकारी).{0,30}बोल रहा हूँ",
        r"(?:मैं|हम).{0,50}(?:बैंक|पुलिस).{0,30}अधिकारी.{0,30}बोल रहा हूँ",
        r"\bthis is an? officer from the (?:cybercrime|cyber crime) department\b",
        
        # Additional English authority claims and threats
        r"\b(?:i am|i'm) (?:a )?police officer\b",
        r"\b(?:police officer|officer).{0,100}\b(?:arrest|instructions|comply)\b",
        r"\b(?:tax officer|government officer)\b.{0,100}\b(?:transfer|pay|penalty)\b",

        # Hinglish authority claims
        r"\bmain (?:police|bank) (?:officer|se|department)\b",
        r"\bmain police officer hoon\b",
        r"\bmeri baat nahi maani to\b",

        # Hindi authority claims and threats
        r"(?:मैं|हम).{0,40}(?:पुलिस अधिकारी|बैंक अधिकारी)",
        r"(?:निर्देश नहीं माने|बात नहीं मानी).{0,40}(?:गिरफ्तार|पुलिस)",

        # Additional English authority claims and threats
        r"\b(?:i am|i'm) (?:a )?police officer\b",
        r"\b(?:police officer|officer).{0,100}\b(?:arrest|instructions|comply)\b",
        r"\b(?:tax officer|government officer)\b.{0,100}\b(?:transfer|pay|penalty)\b",

        # Hinglish authority claims
        r"\bmain (?:police|bank) (?:officer|se|department)\b",
        r"\bmain police officer hoon\b",
        r"\bmeri baat nahi maani to\b",

        # Hindi authority claims and threats
        r"(?:मैं|हम).{0,40}(?:पुलिस अधिकारी|बैंक अधिकारी)",
        r"(?:निर्देश नहीं माने|बात नहीं मानी).{0,40}(?:गिरफ्तार|पुलिस)",
        r"\u092e\u0948\u0902.*\u092a\u0941\u0932\u093f\u0938.*\u0938\u0947 \u092c\u094b\u0932",
        r"\u092e\u0948\u0902 .* \u092c\u0948\u0902\u0915 .* \u0938\u0947 \u092c\u094b\u0932",
        r"\u092e\u0948\u0902.*\u092a\u0941\u0932\u093f\u0938.*\u0938\u0947 \u092c\u094b\u0932",
        r"\u0938\u093e\u0907\u092c\u0930 \u0905\u092a\u0930\u093e\u0927",
        r"\u092c\u0948\u0902\u0915 \u0915\u0947 \u0938\u0941\u0930\u0915\u094d\u0937\u093e \u0935\u093f\u092d\u093e\u0917"

    ],


    
    "otp_pii": [
        # Requests for authentication secrets
        r"\b(?:tell|give|share|provide|read out|send me)\b.{0,40}\b(?:otp|one.time password|password|passcode|cvv|upi pin)\b",
        r"\b(?:otp|one.time password|password|passcode|cvv|upi pin)\b.{0,40}\b(?:tell|give|share|provide|read out|send me)\b",
        r"\benter your (?:upi )?pin\b",
        r"\bshare your otp\b",
        r"\bprovide your password\b",
      

        # Requests for security answers or sensitive identifiers
        
        # Additional personal information used for identity verification
        r"\b(?:tell|give|share|provide|confirm|verify)\b.{0,50}\b(?:place of birth|birthplace|first school|name of your first school)\b",
        r"\b(?:tell|give|share|provide|confirm|verify)\b.{0,50}\b(?:current occupation|annual income|yearly income)\b",
        r"\b(?:place of birth|birthplace|first school|current occupation|annual income|yearly income)\b.{0,50}\b(?:tell|give|share|provide|confirm|verify)\b",
        r"\b(?:what was|what is|can you tell me|could you tell me)\b.{0,50}\b(?:your )?(?:place of birth|birthplace|first school|mother'?s maiden name)\b",

        r"\b(?:tell|give|share|provide|read out)\b.{0,40}\b(?:mother'?s maiden name|security answer|security question answer)\b(?![^.!?]{0,80}\b(?:not|never|don't|do not|shouldn't|should not)\b)",
        r"\b(?:tell|give|share|provide|read out|confirm|verify)\b.{0,60}\b(?:your )?(?:aadhaar number|pan number|full card number|full account number)\b(?![^.!?]{0,60}\b(?:not|never|don't|do not|shouldn't|should not)\b)",
        
        # Ignore explicit warnings against sharing sensitive information
        r"\b(?:do not|don't|never|should not|shouldn't|never share)\b.{0,50}\b(?:share|give|provide|tell|reveal|disclose)\b.{0,40}\b(?:account number|card number|card details|password|pin|otp|date of birth|dob|aadhaar|pan)\b",

        
        # Additional requests for sensitive identifiers and card details
        r"\b(?:tell|give|share|provide|confirm|verify)\b.{0,40}\b(?:date of birth|dob|debit card number|card expiry date|card expiration date|account number)\b(?![^.!?]{0,60}\b(?:not|never|don't|do not|shouldn't|should not)\b)",
        r"\b(?:date of birth|dob|debit card number|card expiry date|card expiration date|account number)\b.{0,40}\b(?:tell|give|share|provide|confirm|verify)\b",
        # Hinglish credential requests
        r"\b(?:apna|aapka) (?:otp|upi pin|pin|password).{0,50}\b(?:bata|bataye|bataiye|do|dijiye|share karo)\b",
        r"\b(?:otp|upi pin|pin|password).{0,50}\b(?:mujhe bata|bata do|de do)\b",

        # Hindi credential requests
        r"(?:अपना|आपका).{0,30}(?:ओटीपी|यूपीआई पिन|पिन|पासवर्ड).{0,40}(?:बताएं|बताइए|बता दीजिए|बताओ|दे दें)",
        r"(?:ओटीपी|यूपीआई पिन|पिन|पासवर्ड).{0,40}(?:मुझे बताएं|मुझे बताइए|बता दीजिए)",
        
        # Additional English OTP requests
        r"\basks? for the otp\b",
        r"\basks? for your otp\b",
        r"\b(?:otp|one.time password)\b.{0,80}\b(?:prevent|avoid|stop)\b.{0,40}\b(?:closure|closing|blocking|suspension)\b",

        # Mixed English-Hinglish credential requests
        r"\botp\b.{0,40}\b(?:abhi|mujhe|bhejo|batao|bata do)\b",
        r"\b(?:enter|share|send)\b.{0,40}\b(?:your )?(?:upi pin|otp|password)\b",

        # Hindi and Hinglish security credentials
        r"(?:ओटीपी|यूपीआई पिन|पासवर्ड).{0,60}(?:मुझे|बताएं|बताइए|भेजें|दीजिए)",
        r"\u0905\u092a\u0928\u093e.*\u0913\u091f\u0940\u092a\u0940",
        r"\u0913\u091f\u0940\u092a\u0940.*\u092c\u0924\u093e\u0907\u090f",
        r"\u0905\u092a\u0928\u093e.*\u092a\u093f\u0928.*\u092c\u0924\u093e\u0907\u090f",
        r"\u092a\u093e\u0938\u0935\u0930\u094d\u0921.*\u092c\u0924\u093e\u0907\u090f"
    ],

    
    
    
    "payment": [
        # Explicit payment demands tied to threats or coercion
        r"\bpay (?:now|immediately|today|within \d+ hours?)\b",
        r"\byou must pay\b",
        r"\byou have to pay\b",
        r"\bpay (?:a )?penalty\b",
        r"\bpay or (?:your|the) account\b",
        r"\bpay or face (?:legal action|consequences)\b",
        r"\bpay to avoid (?:legal action|arrest|suspension|blocking)\b",
        r"\bpay (?:the )?(?:fine|penalty|fee|amount) or\b",

        # Threatening or forced collection
        r"\bcollect the payment\b",
        r"\bsend someone to collect\b",
        r"\bsettle (?:this|the amount|your dues) within \d+ hours?\b",
        r"\btransfer (?:the )?money (?:now|immediately|today)\b",
        r"\bsend (?:the )?money (?:now|immediately|today)\b",
        r"\bdeposit (?:the )?money (?:now|immediately|today)\b",

        # Suspicious payment instructions
        r"\b(?:pay|transfer|send) (?:the )?(?:money|amount) to (?:my|this) (?:account|number|upi)\b",
        r"\bpay (?:through|via) (?:this|my) upi\b",
        r"\benter (?:the )?amount\b.{0,60}\b(?:refund|unauthorized payment)\b",
        r"\bupi pin\b.{0,60}\b(?:refund|authorize|receive money)\b",
        r"\bpaytm app\b.{0,60}\benter the amount\b",
        r"\bphonepe\b.{0,60}\benter the amount\b",
        r"\bgoogle pay\b.{0,60}\benter the amount\b",
        
        r"\b(?:take|pursue) legal action\b.{0,150}\b(?:pay|payment|settle|money|amount)\b",
        # Hinglish payment pressure
        r"\b(?:abhi|turant|jaldi)\b.{0,40}\b(?:paise|paisa|payment|money)\b.{0,30}\b(?:bhejo|bhej do|transfer karo)\b",
        r"\b(?:paise|paisa)\b.{0,40}\b(?:bhejo|bhej do)\b.{0,60}\b(?:warna|nahi to|nahin to)\b",

        # Hindi payment pressure
        r"(?:अभी|तुरंत|जल्दी).{0,40}(?:पैसे|रुपये|राशि).{0,40}(?:भेजें|भेजिए|जमा करें|ट्रांसफर करें)",
        r"(?:पैसे|रुपये).{0,40}(?:भेजें|भेजिए).{0,60}(?:नहीं तो|वरना)",

        # Additional English coercive payment demands
        r"\bpay (?:this |the )?(?:fine|penalty|fee) (?:today )?to avoid legal action\b",
        r"\bpay the fine to my personal upi id\b",
        r"\bprocessing fee\b.{0,80}\b(?:claim|receive|release)\b.{0,40}\b(?:prize|reward|winnings)\b",
        r"\b(?:send|transfer) (?:the )?money\b.{0,100}\b(?:close|dismiss|withdraw) (?:the )?(?:police )?case\b",

        # Hinglish payment demands
        r"\bpenalty\b.{0,50}\b(?:aaj hi|abhi)\b.{0,40}\bpayment karo\b",
        r"\bprize claim karne ke liye\b.{0,80}\bprocessing fee\b",
        r"\b(?:police case|case) close karne ke liye\b.{0,80}\b(?:paise|paisa|money)\b",

        # Hindi payment demands
        r"(?:मामला बंद करने|केस बंद करने).{0,80}(?:पैसे|रुपये).{0,60}(?:भेजें|भेजिए|मेरे खाते में)",
        r"(?:पुरस्कार|इनाम).{0,80}(?:प्रोसेसिंग फीस|प्रसंस्करण शुल्क).{0,60}(?:जमा करें|भुगतान करें)",
        r"\u092a\u0948\u0938\u0947.*\u092d\u0947\u091c\u094b",
        r"\u092a\u0948\u0938\u0947.*\u091f\u094d\u0930\u093e\u0902\u0938\u092b\u0930",
        r"\u0924\u0941\u0930\u0902\u0924.*\u092d\u0941\u0917\u0924\u093e\u0928",
        r"\u092b\u0940\u0938.*\u091c\u092e\u093e \u0915\u0930\u0947\u0902"
    ],



}

COLUMNS = [
    "id", "text", "lang", "urgency", "authority",
    "otp_pii", "payment", "source", "final_verdict",
    "scenario_id", "scenario_type"
]


def detect_language(text):
    text = str(text).lower()

    # Devanagari script indicates Hindi or Hindi-dominant text.
    if re.search(r"[\u0900-\u097F]", text):
        return "Hindi"

    hinglish_words = {
        "hai", "hain", "nahi", "nahin", "haan", "achha",
        "thik", "theek", "ji", "aap", "kya", "kripya",
        "bataye", "bataiye", "paisa", "paise", "turant",
        "abhi", "mat", "karna", "karo", "hoga", "ho",
        "main", "mera", "meri", "aapka", "apna", "mujhe",
        "bata", "bolo", "bhejo", "bhej", "warna", "band",
        "se", "raha", "rahi", "kar", "karna", "liye",
        "paise", "account", "paisa", "dedo", "dijiye"
    }

    words = set(re.findall(r"\b[a-z]+\b", text))
    matches = words & hinglish_words

    if len(matches) >= 2:
        return "Hinglish"

    if words:
        return "English"

    return "Unknown"



def main():
    with open(INPUT_FILE, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()

        for record in records:
            text = record.get("text", "")
            labels = {
                category: int(
                contains_authority(text)
                if category == "authority"
                else contains_otp_pii(text)
                if category == "otp_pii"
                else contains_any(text, patterns)
                )
                for category, patterns in RULES.items()
            }    

            writer.writerow({
                "id": record.get("id", ""),
                "text": text,
                "lang": detect_language(text),
                **labels,
                "source": record.get("source", "icfd_train"),
                "final_verdict": record.get("final_verdict", ""),
                "scenario_id": record.get("scenario_id", ""),
                "scenario_type": record.get("scenario_type", ""),
            })

    print(f"Processed {len(records)} records.")
    print(f"Saved preliminary labels to: {OUTPUT_FILE}")
    print("IMPORTANT: These are heuristic labels, not verified ground truth.")


if __name__ == "__main__":
    main()


