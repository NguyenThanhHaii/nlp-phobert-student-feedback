# PhoBERT Clean vs Noisy Evaluation Report

- Created at: `2026-06-18T07:20:58`
- Stage: `06b_phobert_noisy_evaluation`
- Stage 3 model zip: `/kaggle/input/notebooks/thanhhainguyn/stage-3-phobert-clean-fine-tuning/stage3_phobert_outputs.zip`
- Device: `Tesla T4`
- Primary metric: `Macro-F1`
- No model is trained in this stage; fine-tuned PhoBERT checkpoints from Stage 3 are loaded.
- Clean/noisy test data is regenerated inside Kaggle from the Hugging Face parquet test split and Stage 4 rule configuration.

## Clean performance

| task      | model_type | model_name   | accuracy | macro_f1 | weighted_f1 | num_eval_samples |
| --------- | ---------- | ------------ | -------- | -------- | ----------- | ---------------- |
| sentiment | phobert    | phobert_base | 0.9324   | 0.8295   | 0.9312      | 3166             |
| topic     | phobert    | phobert_base | 0.8992   | 0.8052   | 0.8969      | 3166             |

## Full noisy evaluation summary

| task      | noise_type          | accuracy | macro_f1 | weighted_f1 | num_eval_samples |
| --------- | ------------------- | -------- | -------- | ----------- | ---------------- |
| sentiment | no_accent           | 0.3588   | 0.3165   | 0.4003      | 3166             |
| sentiment | domain_abbreviation | 0.928    | 0.8257   | 0.9277      | 3166             |
| sentiment | teencode_colloquial | 0.9182   | 0.8058   | 0.9184      | 3166             |
| sentiment | typo                | 0.9002   | 0.7848   | 0.9015      | 3166             |
| sentiment | elongation          | 0.9182   | 0.7995   | 0.9181      | 3166             |
| sentiment | mixed_noise         | 0.8917   | 0.7708   | 0.8961      | 3166             |
| topic     | no_accent           | 0.4719   | 0.263    | 0.5122      | 3166             |
| topic     | domain_abbreviation | 0.8515   | 0.7417   | 0.8517      | 3166             |
| topic     | teencode_colloquial | 0.8964   | 0.7941   | 0.8933      | 3166             |
| topic     | typo                | 0.8686   | 0.7645   | 0.8682      | 3166             |
| topic     | elongation          | 0.8863   | 0.7798   | 0.8845      | 3166             |
| topic     | mixed_noise         | 0.8244   | 0.7099   | 0.8275      | 3166             |

## Robustness drop summary

| task      | noise_type          | clean_macro_f1 | noisy_macro_f1 | absolute_macro_f1_drop | relative_macro_f1_drop_pct |
| --------- | ------------------- | -------------- | -------------- | ---------------------- | -------------------------- |
| sentiment | no_accent           | 0.8295         | 0.3165         | 0.5129                 | 61.8396                    |
| sentiment | domain_abbreviation | 0.8295         | 0.8257         | 0.0037                 | 0.449                      |
| sentiment | teencode_colloquial | 0.8295         | 0.8058         | 0.0237                 | 2.8593                     |
| sentiment | typo                | 0.8295         | 0.7848         | 0.0447                 | 5.3835                     |
| sentiment | elongation          | 0.8295         | 0.7995         | 0.03                   | 3.6169                     |
| sentiment | mixed_noise         | 0.8295         | 0.7708         | 0.0586                 | 7.0683                     |
| topic     | no_accent           | 0.8052         | 0.263          | 0.5422                 | 67.337                     |
| topic     | domain_abbreviation | 0.8052         | 0.7417         | 0.0635                 | 7.8833                     |
| topic     | teencode_colloquial | 0.8052         | 0.7941         | 0.0111                 | 1.3751                     |
| topic     | typo                | 0.8052         | 0.7645         | 0.0407                 | 5.0575                     |
| topic     | elongation          | 0.8052         | 0.7798         | 0.0253                 | 3.1477                     |
| topic     | mixed_noise         | 0.8052         | 0.7099         | 0.0952                 | 11.8282                    |

## Interpretation notes

- Positive drop means PhoBERT performs worse than on clean test data.
- Negative drop means the noisy variant happened to score higher than clean and should be interpreted cautiously.
- Changed-only results are available in CSV outputs for rule types that do not affect every row.
- Results should be merged with Stage 6A baseline results for final comparison.

## Limitations

- Noisy data is rule-generated and not fully human-validated.
- This notebook only evaluates PhoBERT. Baseline results are produced by Stage 6A.
- Detailed error examples are left for Stage 7.
