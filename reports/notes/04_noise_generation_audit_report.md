# Controlled Noise Generation and Audit Report

- Created at: `2026-06-18T04:50:14`
- Stage: `04_noise_generation_audit`
- Source split: `d:\project-ml-engineering\nlp-phobert-student-feedback\data\processed\test.csv`
- Noise is generated only from the clean test split.
- Original sentiment and topic labels are preserved.

## Noise types

- `no_accent`
- `domain_abbreviation`
- `teencode_colloquial`
- `typo`
- `elongation`
- `mixed_noise`


## Metadata definitions

- `changed_token_ratio`: number of original whitespace tokens covered by changed spans divided by the number of original whitespace tokens.
- `severity`: `none`, `low`, `medium`, or `high` based on changed-token ratio.
- `changed_spans`: JSON string with original token span, original text, replacement text, and rule name.
- `num_subwords_clean` and `num_subwords_noisy`: counted with the configured PhoBERT tokenizer when available.

## Generation summary

| noise_type | num_rows | num_changed_rows | changed_row_percentage | mean_changed_token_ratio | median_changed_token_ratio | max_changed_token_ratio | mean_subword_delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| no_accent | 3166 | 3162 | 99.87 | 0.7583 | 0.7778 | 0.9615 | 3.5158 |
| domain_abbreviation | 3166 | 1530 | 48.33 | 0.0902 | 0.0 | 0.8 | -0.085 |
| teencode_colloquial | 3166 | 1541 | 48.67 | 0.0484 | 0.0 | 0.4 | 0.1775 |
| typo | 3166 | 3074 | 97.09 | 0.0956 | 0.0833 | 0.5 | 0.5332 |
| elongation | 3166 | 3164 | 99.94 | 0.1019 | 0.0909 | 0.5 | 2.1295 |
| mixed_noise | 3166 | 3135 | 99.02 | 0.2089 | 0.1818 | 0.8 | 1.3714 |


## Severity distribution

| noise_type | severity | count | percentage |
| --- | --- | --- | --- |
| domain_abbreviation | high | 23 | 0.73 |
| domain_abbreviation | low | 1001 | 31.62 |
| domain_abbreviation | medium | 504 | 15.92 |
| domain_abbreviation | none | 1638 | 51.74 |
| elongation | high | 4 | 0.13 |
| elongation | low | 3020 | 95.39 |
| elongation | medium | 140 | 4.42 |
| elongation | none | 2 | 0.06 |
| mixed_noise | high | 227 | 7.17 |
| mixed_noise | low | 1859 | 58.72 |
| mixed_noise | medium | 1049 | 33.13 |
| mixed_noise | none | 31 | 0.98 |
| no_accent | high | 3130 | 98.86 |
| no_accent | medium | 32 | 1.01 |
| no_accent | none | 4 | 0.13 |
| teencode_colloquial | low | 1469 | 46.4 |
| teencode_colloquial | medium | 69 | 2.18 |
| teencode_colloquial | none | 1628 | 51.42 |
| typo | high | 2 | 0.06 |
| typo | low | 2969 | 93.78 |
| typo | medium | 103 | 3.25 |
| typo | none | 92 | 2.91 |


## Audit policy

- Audit sample per noise type: `25`
- The audit file contains empty columns for manual review: `preserve_meaning`, `label_still_valid`, and `audit_note`.
- Samples that clearly change meaning or invalidate the original label should be removed or used only for qualitative analysis.

## Limitations

- The noise rules are deterministic/reproducible but still heuristic.
- Some noise types may not affect every sentence because the required phrase or token pattern may be absent.
- Manual audit is required before treating all generated noisy samples as semantically valid.
- Model evaluation is not performed in Stage 4.
