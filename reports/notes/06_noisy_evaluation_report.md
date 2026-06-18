# Clean vs Noisy Evaluation Report

- Created at: `2026-06-18T13:30:04`
- Stage: `06_noisy_evaluation_all_models`
- Clean test split: `d:\project-ml-engineering\nlp-phobert-student-feedback\data\processed\test.csv`
- Noisy data directory: `d:\project-ml-engineering\nlp-phobert-student-feedback\data\noisy`
- Primary metric: `Macro-F1`
- No model is trained in this stage; all models are loaded from previous stages.

## Evaluation setup

- Clean test set is evaluated once for each model.
- Each Stage 4 noisy test set is evaluated using `noisy_text`.
- Changed-only subsets are also evaluated when `include_changed_only=true`.
- Robustness drop is computed against clean test Macro-F1 for the same task/model.

## Models evaluated

| task | model_type | model_name | path |
| --- | --- | --- | --- |
| sentiment | baseline | majority_class | d:\project-ml-engineering\nlp-phobert-student-feedback\models\baselines\sentiment\majority_class.joblib |
| sentiment | baseline | tfidf_char_svm | d:\project-ml-engineering\nlp-phobert-student-feedback\models\baselines\sentiment\tfidf_char_svm.joblib |
| sentiment | baseline | tfidf_word_svm | d:\project-ml-engineering\nlp-phobert-student-feedback\models\baselines\sentiment\tfidf_word_svm.joblib |
| topic | baseline | majority_class | d:\project-ml-engineering\nlp-phobert-student-feedback\models\baselines\topic\majority_class.joblib |
| topic | baseline | tfidf_char_svm | d:\project-ml-engineering\nlp-phobert-student-feedback\models\baselines\topic\tfidf_char_svm.joblib |
| topic | baseline | tfidf_word_svm | d:\project-ml-engineering\nlp-phobert-student-feedback\models\baselines\topic\tfidf_word_svm.joblib |


## Clean performance

| task | model_type | model_name | accuracy | macro_f1 | weighted_f1 | num_eval_samples |
| --- | --- | --- | --- | --- | --- | --- |
| sentiment | baseline | majority_class | 0.5022 | 0.2229 | 0.3358 | 3166 |
| sentiment | baseline | tfidf_char_svm | 0.874 | 0.7354 | 0.8755 | 3166 |
| sentiment | baseline | tfidf_word_svm | 0.892 | 0.7289 | 0.887 | 3166 |
| topic | baseline | majority_class | 0.7233 | 0.2099 | 0.6072 | 3166 |
| topic | baseline | tfidf_char_svm | 0.8326 | 0.7299 | 0.8396 | 3166 |
| topic | baseline | tfidf_word_svm | 0.8585 | 0.7509 | 0.8598 | 3166 |


## Full noisy evaluation summary

| task | model_name | noise_type | accuracy | macro_f1 | weighted_f1 | num_eval_samples |
| --- | --- | --- | --- | --- | --- | --- |
| sentiment | majority_class | domain_abbreviation | 0.5022 | 0.2229 | 0.3358 | 3166 |
| sentiment | majority_class | elongation | 0.5022 | 0.2229 | 0.3358 | 3166 |
| sentiment | majority_class | mixed_noise | 0.5022 | 0.2229 | 0.3358 | 3166 |
| sentiment | majority_class | no_accent | 0.5022 | 0.2229 | 0.3358 | 3166 |
| sentiment | majority_class | teencode_colloquial | 0.5022 | 0.2229 | 0.3358 | 3166 |
| sentiment | majority_class | typo | 0.5022 | 0.2229 | 0.3358 | 3166 |
| sentiment | tfidf_char_svm | domain_abbreviation | 0.8749 | 0.7397 | 0.8777 | 3166 |
| sentiment | tfidf_char_svm | elongation | 0.8774 | 0.7432 | 0.8797 | 3166 |
| sentiment | tfidf_char_svm | mixed_noise | 0.8377 | 0.7076 | 0.8457 | 3166 |
| sentiment | tfidf_char_svm | no_accent | 0.4618 | 0.405 | 0.5134 | 3166 |
| sentiment | tfidf_char_svm | teencode_colloquial | 0.8405 | 0.7035 | 0.845 | 3166 |
| sentiment | tfidf_char_svm | typo | 0.8484 | 0.7065 | 0.853 | 3166 |
| sentiment | tfidf_word_svm | domain_abbreviation | 0.8929 | 0.7376 | 0.8884 | 3166 |
| sentiment | tfidf_word_svm | elongation | 0.8733 | 0.7136 | 0.8703 | 3166 |
| sentiment | tfidf_word_svm | mixed_noise | 0.8452 | 0.695 | 0.846 | 3166 |
| sentiment | tfidf_word_svm | no_accent | 0.4068 | 0.3597 | 0.4474 | 3166 |
| sentiment | tfidf_word_svm | teencode_colloquial | 0.8632 | 0.7004 | 0.86 | 3166 |
| sentiment | tfidf_word_svm | typo | 0.874 | 0.7132 | 0.8706 | 3166 |
| topic | majority_class | domain_abbreviation | 0.7233 | 0.2099 | 0.6072 | 3166 |
| topic | majority_class | elongation | 0.7233 | 0.2099 | 0.6072 | 3166 |
| topic | majority_class | mixed_noise | 0.7233 | 0.2099 | 0.6072 | 3166 |
| topic | majority_class | no_accent | 0.7233 | 0.2099 | 0.6072 | 3166 |
| topic | majority_class | teencode_colloquial | 0.7233 | 0.2099 | 0.6072 | 3166 |
| topic | majority_class | typo | 0.7233 | 0.2099 | 0.6072 | 3166 |
| topic | tfidf_char_svm | domain_abbreviation | 0.7903 | 0.6725 | 0.8046 | 3166 |
| topic | tfidf_char_svm | elongation | 0.8263 | 0.7263 | 0.8351 | 3166 |
| topic | tfidf_char_svm | mixed_noise | 0.7666 | 0.6513 | 0.7856 | 3166 |
| topic | tfidf_char_svm | no_accent | 0.4172 | 0.2875 | 0.4832 | 3166 |
| topic | tfidf_char_svm | teencode_colloquial | 0.8332 | 0.7292 | 0.8396 | 3166 |
| topic | tfidf_char_svm | typo | 0.7843 | 0.6763 | 0.8001 | 3166 |
| topic | tfidf_word_svm | domain_abbreviation | 0.8266 | 0.7103 | 0.8317 | 3166 |
| topic | tfidf_word_svm | elongation | 0.8395 | 0.7266 | 0.8438 | 3166 |
| topic | tfidf_word_svm | mixed_noise | 0.7994 | 0.6814 | 0.8104 | 3166 |
| topic | tfidf_word_svm | no_accent | 0.3298 | 0.2614 | 0.4065 | 3166 |
| topic | tfidf_word_svm | teencode_colloquial | 0.8601 | 0.7538 | 0.8615 | 3166 |
| topic | tfidf_word_svm | typo | 0.8291 | 0.7166 | 0.8356 | 3166 |


## Robustness ranking

| task | model_type | model_name | mean_absolute_macro_f1_drop | max_absolute_macro_f1_drop | mean_relative_macro_f1_drop_pct |
| --- | --- | --- | --- | --- | --- |
| sentiment | baseline | majority_class | 0.0 | 0.0 | 0.0 |
| sentiment | baseline | tfidf_char_svm | 0.0678 | 0.3303 | 9.2147 |
| sentiment | baseline | tfidf_word_svm | 0.0757 | 0.3692 | 10.3811 |
| topic | baseline | majority_class | 0.0 | 0.0 | 0.0 |
| topic | baseline | tfidf_char_svm | 0.1061 | 0.4424 | 14.5305 |
| topic | baseline | tfidf_word_svm | 0.1092 | 0.4896 | 14.546 |


## Interpretation notes

- Positive drop means the model performs worse than on clean test data.
- Negative drop means the noisy variant happened to score higher than clean; this should be interpreted cautiously.
- Full noisy evaluation keeps all test rows. Changed-only evaluation isolates rows actually modified by Stage 4 rules.
- This stage reports quantitative robustness. Detailed error examples are left for Stage 7.

## Limitations

- Noisy data is rule-generated and not fully human-validated.
- Domain abbreviation and teencode sets include unchanged rows when no rule matched.
- Results should be interpreted together with Stage 5 tokenization analysis and Stage 7 error analysis.
