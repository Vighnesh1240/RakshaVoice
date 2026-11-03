# RakshaVoice — Phase 3: Speech Recognition and Intent Analysis

## 1. Objective
Integrate Automatic Speech Recognition (ASR) with the existing text intent classifier to extract conversation-level risk signals from audio.

## 2. Implementation
- **ASR:** Faster-Whisper
- **Text classification:** Existing MuRIL-based multi-label classifier (`nlp/models/text_v1`)
- **Pipeline:** Audio → Transcription → Intent scoring
- **Intent categories:** Urgency, authority, OTP/PII, and payment
- **Pipeline module:** `nlp/pipeline.py`
- **ASR module:** `nlp/asr.py`
- **Reusable classifier:** `nlp/intent_stream.py`
- **Evaluation script:** `nlp/evaluate_pipeline.py`
- **Evaluation results:** `nlp/pipeline_evaluation.csv`

## 3. Evaluation
Four WAV recordings were processed successfully.

| Audio | Latency (ms) | OTP/PII detected | Payment detected |
|---|---:|---|---|
| `10.wav` | 2801.23 | Yes | No |
| `1959.wav` | 948.99 | No | No |
| `4.wav` | 469.55 | Yes | Yes |
| `test16k.wav` | 646.70 | Yes | No |

The first run was slower than subsequent runs, potentially due to model initialization. These measurements are preliminary and should not be treated as definitive steady-state latency.

## 4. Observations
- ASR produced coherent English transcripts for three recordings.
- `4.wav` produced only the transcript “Um,” so its intent predictions are not dependable.
- `10.wav`, containing biblical-style narration, received a high OTP/PII score.
- `test16k.wav`, containing a request for college fees, was also flagged for OTP/PII.
- These observations indicate possible false positives and unreliable intent predictions.

## 5. Limitations
- Only four recordings were evaluated.
- Verified ground-truth labels are unavailable for this evaluation.
- Accuracy, precision, recall, and F1-score cannot be established from these results.
- The classifier is experimental and must not be treated as a definitive scam detector.
- More representative, labelled recordings are needed for a proper evaluation.

## 6. Conclusion
The ASR-to-NLP integration is operational, and preliminary evaluation has been completed. However, classifier reliability remains a limitation. The current model and thresholds have been preserved; further validation is required before relying on these predictions in a live risk assessment system.

**Phase 3 status:** Integration and preliminary testing completed; reliability validation and documentation are being finalized.
