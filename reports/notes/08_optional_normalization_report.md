# Optional Normalization Experiment Report

- Created at: `2026-06-18T14:48:58`
- Stage: `08_optional_normalization`
- This stage tests simple rule-based normalization on controlled noisy test sets.
- No model is trained in this stage.
- This local run evaluates baseline TF-IDF/SVM models. PhoBERT normalization evaluation can be run separately on GPU if needed.

## Normalization rules

- Reduce elongation, e.g. repeated characters length >= 3.
- Expand domain abbreviations such as `gv`, `sv`, `csvc`, `ctdt`.
- Expand teencode/colloquial forms such as `ko`, `k`, `dc`, `đc`, `lun`.
- Restore a limited set of no-accent Vietnamese phrases.
- Correct a small set of common typo forms.

## Normalization generation summary

| noise_type | num_rows | num_noisy_changed_rows | num_normalization_changed_rows | normalization_changed_percentage | mean_num_normalization_replacements | max_num_normalization_replacements |
| --- | --- | --- | --- | --- | --- | --- |
| domain_abbreviation | 3166 | 1530 | 1531 | 48.3575 | 0.6352 | 3 |
| elongation | 3166 | 3164 | 3164 | 99.9368 | 1.0009 | 2 |
| mixed_noise | 3166 | 3135 | 2673 | 84.4283 | 1.4719 | 4 |
| no_accent | 3166 | 3162 | 2469 | 77.9848 | 1.5474 | 19 |
| teencode_colloquial | 3166 | 1541 | 1541 | 48.6734 | 0.6481 | 3 |
| typo | 3166 | 3074 | 378 | 11.9394 | 0.1197 | 2 |


## Macro-F1 improvement summary, full scope

| task | model_name | noise_type | noisy_macro_f1 | normalized_macro_f1 | macro_f1_improvement | num_eval_samples |
| --- | --- | --- | --- | --- | --- | --- |
| sentiment | majority_class | domain_abbreviation | 0.2229 | 0.2229 | 0.0 | 3166 |
| sentiment | majority_class | elongation | 0.2229 | 0.2229 | 0.0 | 3166 |
| sentiment | majority_class | mixed_noise | 0.2229 | 0.2229 | 0.0 | 3166 |
| sentiment | majority_class | no_accent | 0.2229 | 0.2229 | 0.0 | 3166 |
| sentiment | majority_class | teencode_colloquial | 0.2229 | 0.2229 | 0.0 | 3166 |
| sentiment | majority_class | typo | 0.2229 | 0.2229 | 0.0 | 3166 |
| sentiment | tfidf_char_svm | domain_abbreviation | 0.7397 | 0.7356 | -0.0041 | 3166 |
| sentiment | tfidf_char_svm | elongation | 0.7432 | 0.7354 | -0.0079 | 3166 |
| sentiment | tfidf_char_svm | mixed_noise | 0.7076 | 0.7317 | 0.0241 | 3166 |
| sentiment | tfidf_char_svm | no_accent | 0.405 | 0.5325 | 0.1275 | 3166 |
| sentiment | tfidf_char_svm | teencode_colloquial | 0.7035 | 0.7354 | 0.0319 | 3166 |
| sentiment | tfidf_char_svm | typo | 0.7065 | 0.7145 | 0.008 | 3166 |
| sentiment | tfidf_word_svm | domain_abbreviation | 0.7376 | 0.7292 | -0.0083 | 3166 |
| sentiment | tfidf_word_svm | elongation | 0.7136 | 0.7289 | 0.0153 | 3166 |
| sentiment | tfidf_word_svm | mixed_noise | 0.695 | 0.7244 | 0.0294 | 3166 |
| sentiment | tfidf_word_svm | no_accent | 0.3597 | 0.5505 | 0.1908 | 3166 |
| sentiment | tfidf_word_svm | teencode_colloquial | 0.7004 | 0.7289 | 0.0285 | 3166 |
| sentiment | tfidf_word_svm | typo | 0.7132 | 0.7197 | 0.0065 | 3166 |
| topic | majority_class | domain_abbreviation | 0.2099 | 0.2099 | 0.0 | 3166 |
| topic | majority_class | elongation | 0.2099 | 0.2099 | 0.0 | 3166 |
| topic | majority_class | mixed_noise | 0.2099 | 0.2099 | 0.0 | 3166 |
| topic | majority_class | no_accent | 0.2099 | 0.2099 | 0.0 | 3166 |
| topic | majority_class | teencode_colloquial | 0.2099 | 0.2099 | 0.0 | 3166 |
| topic | majority_class | typo | 0.2099 | 0.2099 | 0.0 | 3166 |
| topic | tfidf_char_svm | domain_abbreviation | 0.6725 | 0.7301 | 0.0576 | 3166 |
| topic | tfidf_char_svm | elongation | 0.7263 | 0.7299 | 0.0036 | 3166 |
| topic | tfidf_char_svm | mixed_noise | 0.6513 | 0.7061 | 0.0548 | 3166 |
| topic | tfidf_char_svm | no_accent | 0.2875 | 0.4716 | 0.1842 | 3166 |
| topic | tfidf_char_svm | teencode_colloquial | 0.7292 | 0.7299 | 0.0007 | 3166 |
| topic | tfidf_char_svm | typo | 0.6763 | 0.6816 | 0.0053 | 3166 |
| topic | tfidf_word_svm | domain_abbreviation | 0.7103 | 0.7511 | 0.0408 | 3166 |
| topic | tfidf_word_svm | elongation | 0.7266 | 0.7509 | 0.0244 | 3166 |
| topic | tfidf_word_svm | mixed_noise | 0.6814 | 0.7315 | 0.05 | 3166 |
| topic | tfidf_word_svm | no_accent | 0.2614 | 0.49 | 0.2286 | 3166 |
| topic | tfidf_word_svm | teencode_colloquial | 0.7538 | 0.7509 | -0.0029 | 3166 |
| topic | tfidf_word_svm | typo | 0.7166 | 0.7208 | 0.0041 | 3166 |


## Interpretation notes

- Positive improvement means normalized input performs better than noisy input.
- Negative improvement means normalization hurt performance.
- No-accent normalization is intentionally limited; it cannot fully restore Vietnamese accents.
- Results should be interpreted as an optional recovery experiment, not as a new robust model.

## Limitations

- The normalizer is rule-based and incomplete.
- Some replacements may be context-insensitive.
- PhoBERT normalization evaluation is not included in this local baseline run.
- The noisy data is rule-generated and not fully human-validated.
