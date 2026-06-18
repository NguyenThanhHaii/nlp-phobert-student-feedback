# Error Analysis Report

- Created at: `2026-06-18T14:37:42`
- Stage: `07_error_analysis`
- This stage analyzes predictions generated in Stage 6.
- No model is trained or evaluated again in Stage 7.

## Inputs

- Baseline predictions: `d:\project-ml-engineering\nlp-phobert-student-feedback\reports\tables\06_model_predictions_all.csv`
- PhoBERT predictions: `d:\project-ml-engineering\nlp-phobert-student-feedback\reports\tables\06b_phobert_predictions_light.csv`
- Tokenization analysis: `d:\project-ml-engineering\nlp-phobert-student-feedback\reports\tables\05_segmentation_tokenization_analysis.csv`

## Error-rate summary

| task | model_type | model_name | noise_type | evaluation_scope | num_samples | num_errors | error_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sentiment | phobert | phobert_base | clean | full | 3166 | 214 | 0.0676 |
| sentiment | phobert | phobert_base | domain_abbreviation | full | 3166 | 228 | 0.072 |
| sentiment | phobert | phobert_base | elongation | full | 3166 | 259 | 0.0818 |
| sentiment | phobert | phobert_base | mixed_noise | full | 3166 | 343 | 0.1083 |
| sentiment | phobert | phobert_base | no_accent | full | 3166 | 2030 | 0.6412 |
| sentiment | phobert | phobert_base | teencode_colloquial | full | 3166 | 259 | 0.0818 |
| sentiment | phobert | phobert_base | typo | full | 3166 | 316 | 0.0998 |
| sentiment | baseline | tfidf_char_svm | clean | full | 3166 | 399 | 0.126 |
| sentiment | baseline | tfidf_char_svm | domain_abbreviation | full | 3166 | 396 | 0.1251 |
| sentiment | baseline | tfidf_char_svm | elongation | full | 3166 | 388 | 0.1226 |
| sentiment | baseline | tfidf_char_svm | mixed_noise | full | 3166 | 514 | 0.1623 |
| sentiment | baseline | tfidf_char_svm | no_accent | full | 3166 | 1704 | 0.5382 |
| sentiment | baseline | tfidf_char_svm | teencode_colloquial | full | 3166 | 505 | 0.1595 |
| sentiment | baseline | tfidf_char_svm | typo | full | 3166 | 480 | 0.1516 |
| sentiment | baseline | tfidf_word_svm | clean | full | 3166 | 342 | 0.108 |
| sentiment | baseline | tfidf_word_svm | domain_abbreviation | full | 3166 | 339 | 0.1071 |
| sentiment | baseline | tfidf_word_svm | elongation | full | 3166 | 401 | 0.1267 |
| sentiment | baseline | tfidf_word_svm | mixed_noise | full | 3166 | 490 | 0.1548 |
| sentiment | baseline | tfidf_word_svm | no_accent | full | 3166 | 1878 | 0.5932 |
| sentiment | baseline | tfidf_word_svm | teencode_colloquial | full | 3166 | 433 | 0.1368 |
| sentiment | baseline | tfidf_word_svm | typo | full | 3166 | 399 | 0.126 |
| topic | phobert | phobert_base | clean | full | 3166 | 319 | 0.1008 |
| topic | phobert | phobert_base | domain_abbreviation | full | 3166 | 470 | 0.1485 |
| topic | phobert | phobert_base | elongation | full | 3166 | 360 | 0.1137 |
| topic | phobert | phobert_base | mixed_noise | full | 3166 | 556 | 0.1756 |
| topic | phobert | phobert_base | no_accent | full | 3166 | 1672 | 0.5281 |
| topic | phobert | phobert_base | teencode_colloquial | full | 3166 | 328 | 0.1036 |
| topic | phobert | phobert_base | typo | full | 3166 | 416 | 0.1314 |
| topic | baseline | tfidf_char_svm | clean | full | 3166 | 530 | 0.1674 |
| topic | baseline | tfidf_char_svm | domain_abbreviation | full | 3166 | 664 | 0.2097 |
| topic | baseline | tfidf_char_svm | elongation | full | 3166 | 550 | 0.1737 |
| topic | baseline | tfidf_char_svm | mixed_noise | full | 3166 | 739 | 0.2334 |
| topic | baseline | tfidf_char_svm | no_accent | full | 3166 | 1845 | 0.5828 |
| topic | baseline | tfidf_char_svm | teencode_colloquial | full | 3166 | 528 | 0.1668 |
| topic | baseline | tfidf_char_svm | typo | full | 3166 | 683 | 0.2157 |
| topic | baseline | tfidf_word_svm | clean | full | 3166 | 448 | 0.1415 |
| topic | baseline | tfidf_word_svm | domain_abbreviation | full | 3166 | 549 | 0.1734 |
| topic | baseline | tfidf_word_svm | elongation | full | 3166 | 508 | 0.1605 |
| topic | baseline | tfidf_word_svm | mixed_noise | full | 3166 | 635 | 0.2006 |
| topic | baseline | tfidf_word_svm | no_accent | full | 3166 | 2122 | 0.6702 |
| topic | baseline | tfidf_word_svm | teencode_colloquial | full | 3166 | 443 | 0.1399 |
| topic | baseline | tfidf_word_svm | typo | full | 3166 | 541 | 0.1709 |


## Top confusion pairs

| task | model_type | model_name | noise_type | evaluation_scope | true_label | pred_label | num_errors |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sentiment | phobert | phobert_base | clean | full | positive | negative | 50 |
| sentiment | phobert | phobert_base | clean | full | neutral | positive | 47 |
| sentiment | phobert | phobert_base | clean | full | negative | positive | 35 |
| sentiment | phobert | phobert_base | clean | full | negative | neutral | 30 |
| sentiment | phobert | phobert_base | clean | full | neutral | negative | 28 |
| sentiment | phobert | phobert_base | clean | full | positive | neutral | 24 |
| sentiment | phobert | phobert_base | domain_abbreviation | full | positive | negative | 54 |
| sentiment | phobert | phobert_base | domain_abbreviation | full | neutral | positive | 44 |
| sentiment | phobert | phobert_base | domain_abbreviation | full | negative | positive | 37 |
| sentiment | phobert | phobert_base | domain_abbreviation | full | negative | neutral | 34 |
| sentiment | phobert | phobert_base | domain_abbreviation | full | positive | neutral | 32 |
| sentiment | phobert | phobert_base | domain_abbreviation | full | neutral | negative | 27 |
| sentiment | phobert | phobert_base | elongation | full | positive | negative | 56 |
| sentiment | phobert | phobert_base | elongation | full | neutral | positive | 45 |
| sentiment | phobert | phobert_base | elongation | full | negative | positive | 43 |
| sentiment | phobert | phobert_base | elongation | full | negative | neutral | 42 |
| sentiment | phobert | phobert_base | elongation | full | positive | neutral | 37 |
| sentiment | phobert | phobert_base | elongation | full | neutral | negative | 36 |
| sentiment | phobert | phobert_base | mixed_noise | full | positive | negative | 73 |
| sentiment | phobert | phobert_base | mixed_noise | full | positive | neutral | 70 |
| sentiment | phobert | phobert_base | mixed_noise | full | negative | neutral | 66 |
| sentiment | phobert | phobert_base | mixed_noise | full | negative | positive | 61 |
| sentiment | phobert | phobert_base | mixed_noise | full | neutral | positive | 41 |
| sentiment | phobert | phobert_base | mixed_noise | full | neutral | negative | 32 |
| sentiment | phobert | phobert_base | no_accent | full | negative | neutral | 795 |
| sentiment | phobert | phobert_base | no_accent | full | positive | neutral | 665 |
| sentiment | phobert | phobert_base | no_accent | full | negative | positive | 400 |
| sentiment | phobert | phobert_base | no_accent | full | positive | negative | 125 |
| sentiment | phobert | phobert_base | no_accent | full | neutral | positive | 31 |
| sentiment | phobert | phobert_base | no_accent | full | neutral | negative | 14 |


## Minority and difficult classes

- Sentiment focus: `neutral`.
- Topic focus: `facility`, `others`.
- Detailed examples are saved in `07_minority_class_error_samples.csv`.

## No-accent errors

- No-accent was the strongest noise type in Stage 6.
- Separate tables are exported: `07_no_accent_error_samples.csv` and `07_no_accent_confusion_summary.csv`.

## Tokenization-related errors

- Error rates are summarized by subword inflation bins.
- High-tokenization-shift error examples are exported for qualitative review.

## Limitations

- Error samples are selected automatically and still need human interpretation.
- Noisy data is rule-generated and not fully human-validated.
- Majority-class baseline is excluded from qualitative samples because it ignores input text.
- Stage 7 explains observed errors; it does not introduce a new model or normalization method.
