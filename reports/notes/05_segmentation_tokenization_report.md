# Segmentation and PhoBERT Tokenization Analysis Report

- Created at: `2026-06-18T05:02:36`
- Stage: `05_segmentation_tokenization_analysis`
- Clean test split: `d:\project-ml-engineering\nlp-phobert-student-feedback\data\processed\test.csv`
- Noisy data directory: `d:\project-ml-engineering\nlp-phobert-student-feedback\data\noisy`
- Segmentation method: `whitespace`
- Tokenizer source: `vinai/phobert-base`
- Model evaluation is not performed in Stage 5.

## Analysis scope

- Compare `original_text` and `noisy_text` from each Stage 4 noisy test set.
- Measure segmentation/word-count change.
- Measure PhoBERT subword tokenization change.
- Identify examples with large subword inflation.

## Metric definitions

- `segment_delta`: `num_segments_noisy - num_segments_clean`.
- `subword_delta`: `num_subwords_noisy - num_subwords_clean`.
- `subword_inflation_ratio`: `num_subwords_noisy / num_subwords_clean`.
- `segmentation_change_rate`: percentage of rows whose segment sequence changed.
- `subword_change_rate`: percentage of rows whose PhoBERT subword sequence changed.

## Summary by noise type

| noise_type | num_rows | num_changed_rows | mean_changed_token_ratio | mean_segment_delta | segmentation_change_rate | mean_subword_delta | mean_subword_inflation_ratio | subword_change_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| domain_abbreviation | 3166 | 1530 | 0.0902 | -0.6415 | 48.2628 | -0.085 | 0.9945 | 48.2628 |
| elongation | 3166 | 3164 | 0.1019 | 0.0 | 99.9368 | 2.1295 | 1.2166 | 99.9368 |
| mixed_noise | 3166 | 3135 | 0.2089 | -0.4899 | 99.0208 | 1.3714 | 1.1363 | 99.0208 |
| no_accent | 3166 | 3162 | 0.7583 | 0.0 | 99.8737 | 3.5158 | 1.2367 | 99.8737 |
| teencode_colloquial | 3166 | 1541 | 0.0484 | 0.0 | 48.5786 | 0.1775 | 1.0135 | 48.5786 |
| typo | 3166 | 3074 | 0.0956 | 0.0 | 97.0941 | 0.5332 | 1.0516 | 97.0941 |


## Changed-only summary

| noise_type | num_changed_subset_rows | mean_changed_token_ratio | mean_segment_delta | segmentation_change_rate | mean_subword_delta | mean_subword_inflation_ratio | subword_change_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| domain_abbreviation | 1530 | 0.1867 | -1.3275 | 99.8693 | -0.1758 | 0.9887 | 99.8693 |
| elongation | 3164 | 0.1019 | 0.0 | 100.0 | 2.1308 | 1.2167 | 100.0 |
| mixed_noise | 3135 | 0.2109 | -0.4947 | 100.0 | 1.385 | 1.1377 | 100.0 |
| no_accent | 3162 | 0.7592 | 0.0 | 100.0 | 3.5202 | 1.237 | 100.0 |
| teencode_colloquial | 1541 | 0.0994 | 0.0 | 99.8053 | 0.3647 | 1.0277 | 99.8053 |
| typo | 3074 | 0.0984 | 0.0 | 100.0 | 0.5491 | 1.0531 | 100.0 |


## Interpretation notes

- `no_accent` is a systematic transformation. A high tokenization change rate should be interpreted as broad diacritic removal, not necessarily semantic corruption.
- `domain_abbreviation` and `teencode_colloquial` may contain many unchanged rows because only sentences matching the configured rules are transformed.
- Changed-only statistics are useful for isolating the effect of rows that were actually modified.
- This stage does not conclude which model is more robust; that is handled in Stage 6.

## Limitations

- The default segmentation method is whitespace unless an external Vietnamese segmenter is configured.
- PhoBERT tokenizer analysis is performed directly on the clean/noisy text produced by Stage 4.
- Manual semantic validation of all noisy samples is not performed in this project scope.
