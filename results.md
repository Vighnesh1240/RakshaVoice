# RakshaVoice - Phase 2 Results

## 1. Live Audio Streaming

- **Live detection:** Working
- **Gauge update interval:** Approximately 2 seconds
- **Observed inference latency:** 281.6 ms
- **Stop button:** Verified; microphone stops and status changes to Disconnected.

## 2. Speaker-Playback Test

- **Test type:** Voice recording played through phone speaker near laptop microphone
- **Observed spoof score:** 0.01%
- **Observation:** The model returned a very low predicted spoof probability for this input. This single test does not establish general classification accuracy.

## 3. WebSocket Test

- **Test script:** `backend/test_ws.py`
- **Input file:** `test16k.wav`
- **Windows scored:** 4
- **Window 1:** 99.9995% spoof probability; 293.06 ms latency
- **Window 2:** 99.9361% spoof probability; 306.55 ms latency
- **Window 3:** 99.9990% spoof probability; 289.90 ms latency
- **Window 4:** 99.9945% spoof probability; 290.33 ms latency
- **Average spoof probability:** Approximately 99.9823%
- **Average inference latency:** Approximately 294.96 ms
- **Ground-truth label:** Not yet verified
- **Bonafide ASVspoof clip score:** Not yet recorded
- **Spoof ASVspoof clip score:** Not yet recorded

**Observation:** All four windows received high predicted spoof probabilities. The input file's ground-truth label must be established before judging whether these predictions are correct.

## 4. Audio Storage Check

- **WebSocket processing:** Audio is processed in memory by the `/ws` route.
- **Source-code search:** No matches found for the searched common audio-writing patterns in `backend` and `frontend/src`.
- **Conclusion:** No evidence of microphone audio being saved by the inspected WebSocket route or searched source patterns.

## 5. Remaining Checks

- [ ] Record the bonafide ASVspoof test score.
- [ ] Record the spoof ASVspoof test score.
- [ ] Record a live microphone voice score.
- [ ] Perform a phone-call speakerphone test, if available.
- [ ] Verify `.gitignore` and review `git status`.
- [ ] Commit the Phase 2 code and results.

## 6. Overall Observation

The live audio pipeline is functioning, the gauge updates at approximately the expected interval, and the observed inference latency is below the approximate 2000 ms target. Further testing on labeled genuine and spoofed audio is needed to assess classification performance.

## Phase 3: Intent stream (fine-tuned MuRIL, chunk level)
- Training: ICFD chunks (strict relabel) + 246 synthetic rows (x4), 6,421 rows; 8 epochs, best epoch by val mean-AP
- Verified gold (249 rows; 110 reviewed by hand, 4 changed): otp_pii P 1.00 R 0.97 (thr 0.40), PR-AUC 1.00;
  urgency P 0.82 R 0.64 F1 0.72 (thr 0.50), PR-AUC 0.84
- Probe set (60 fresh rows, never used in training): authority F1 0.91, payment F1 0.67, urgency F1 0.63 (P 0.52)
- Sanity cases: 0.925 (original frozen-embedding model: 0.783)
- Limitations: gold labels AI-assisted and partly human-checked; gold from the same source as training (optimistic);
  authority/payment have too few real-data positives; urgency over-fires on scam demands; unseen regulators (SEBI) missed

  ## Phase 3: Intent stream (fine-tuned MuRIL on chunks, Hub84 Hinglish Whisper)
- Training data: ICFD chunks (strict relabel) + synthetic batches 1-5 (282 rows x4) + spelling variants (197)
- Verified gold (249 rows; 110 reviewed, 4 changed): otp_pii P 0.91 R 1.00 F1 0.95 (CI 0.89-1.0); urgency P 0.77 R 0.71 F1 0.74 (CI 0.62-0.84)
- Probe set (60 fresh synthetic rows, thresholds fixed in advance): urgency F1 0.84, authority F1 0.85, payment F1 0.63 (0.92 with money-word rule, rule built after seeing the probe)
- Sanity cases: 0.925 (original frozen-embedding model 0.783)
- ASR comparison on 36 own clips: small default 26/36, small+prompt 30/36, small no prompt 21/36, Hinglish-tuned 31/36
- End to end after retrain: 35/36 clips (normal-call false alarms: 0), latency ASR 1.1-1.3 s per 8 s clip, intent model 10-30 ms
- VRAM: 3,432 MiB of 4,096 with spoof + intent + Whisper loaded (desktop baseline 545 MiB)
- Limitations: gold from the same source as training and AI-assisted (partly human-checked); authority/payment evaluated on a small synthetic probe;
  payment depends on a rule gate; the 36 clips were used to choose ASR and design fixes, so 35/36 is a development-set number