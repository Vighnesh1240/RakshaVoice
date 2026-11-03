import os, sounddevice as sd, soundfile as sf
SR, SEC = 16000, 8
OUT = "nlp/test_clips"; os.makedirs(OUT, exist_ok=True)
LINES = [
 ("en_scam_otp",    "Sir, I am calling from your bank. Please tell me the OTP you just received."),
 ("en_scam_police", "This is the cyber crime police. Your Aadhaar is linked to a case and you will be arrested today."),
 ("en_scam_pay",    "Please transfer five thousand rupees to this account right now to clear your penalty."),
 ("en_norm_emi",    "Hello, this is a reminder that your EMI will be debited on the tenth."),
 ("en_norm_warn",   "Please never share your OTP or PIN with anyone, even bank staff."),
 ("en_norm_deliv",  "Your courier is out for delivery, please keep your phone nearby."),
 ("hg_scam_otp",    "Sir aapka account block ho jayega, abhi apna OTP bataiye."),
 ("hg_scam_police", "Main CBI officer bol raha hoon, aapke naam par case darj hua hai."),
 ("hg_scam_pay",    "Processing fee ke liye abhi paanch hazaar rupaye UPI se bhej dijiye."),
 ("hg_norm_emi",    "Aapki EMI dus tareekh ko kategi, balance rakhiyega."),
 ("hg_norm_warn",   "OTP kisi ko mat batana, bank kabhi nahi maangta."),
 ("hg_norm_deliv",  "Aapka parcel kal subah tak deliver ho jayega."),
 ("hi_scam_otp",    "आपका खाता बंद हो जाएगा, अभी अपना ओटीपी बताइए।"),
 ("hi_scam_police", "मैं सीबीआई अधिकारी बोल रहा हूँ, आपके खिलाफ वारंट जारी हुआ है।"),
 ("hi_scam_pay",    "प्रोसेसिंग फीस के लिए अभी पाँच हज़ार रुपये भेज दीजिए।"),
 ("hi_norm_emi",    "आपकी ईएमआई दस तारीख को कटेगी, कृपया बैलेंस रखिएगा।"),
 ("hi_norm_warn",   "अपना ओटीपी किसी को न बताएँ, बैंक कभी नहीं माँगता।"),
 ("hi_norm_deliv",  "आपका पार्सल कल सुबह तक पहुँच जाएगा।"),
]
TAG = input("Tag for this session (e.g. clean or far): ").strip() or "clean"
for name, text in LINES:
    input(f"\n[{name}]  Press Enter, then read aloud within {SEC}s:\n   {text}\n")
    y = sd.rec(int(SEC * SR), samplerate=SR, channels=1, dtype="float32"); sd.wait()
    sf.write(f"{OUT}/{name}__{TAG}.wav", y, SR)
print("saved to", OUT)