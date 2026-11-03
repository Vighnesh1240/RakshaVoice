# RakshaVoice — Dataset Quality Control Log

## Dataset
- Source: ICFD-31k training shard sample
- Input records: 2,100
- Labeling method: Preliminary regex-based heuristic labeling
- Status: Not yet manually validated as ground truth

## Authority Label Review
- Initial authority-positive records: 5
- After removing the broad fraud-department rule: 3
- After removing the authorized-representative rule: 2
- Remaining positive IDs: 1132, 1860
- Review status: Provisional; further investigation required

## Known Quality Issues
- ID 1860 requests screen sharing and login credentials despite its scenario being marked Clear Normal.
- Scenario-level classifications do not guarantee correct individual category labels.
- Language detection is heuristic and requires validation.
- Automated labels must not be treated as verified ground truth.

## Next Actions
- Review label quality for all four categories.
- Validate a sample of positive and negative labels manually.
- Check duplicate and near-duplicate records.
- Establish leakage-safe train/dev/test splits.
- Preserve source provenance and document any corrections.


## Authority Rule Review — Final Pass

- Dataset size: 2,100 records.
- Authority positives after targeted rule updates: 11.
- Reviewed all 11 authority-positive records.
- No obvious false positives identified in the displayed excerpts.
- Authority includes explicit official identity claims and legal/enforcement pressure used to compel compliance.
- Ordinary bank and department references are not sufficient by themselves.
- Authority remains a sparse, heuristic label; additional verified examples are required before model training.
- Decision: freeze the current rules for this labeling pass. Do not continue adding regex patterns without new evidence.
- Other current positive counts: urgency 380, otp_pii 305, payment 343.
- All labels remain preliminary and are not verified ground truth.