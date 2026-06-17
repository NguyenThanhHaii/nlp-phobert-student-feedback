# PhoBERT Clean Fine-tuning Report

- Created at: `2026-06-17T20:51:38`
- Stage: `03_phobert_clean_finetuning`
- Dataset: processed UIT-VSFC splits from Stage 1
- Main metric: Macro-F1
- Model selection split: dev
- Final reporting split: test

## Model and preprocessing

- Model name: `vinai/phobert-base`
- Max length: `128`
- Word segmentation enabled: `False`
- Word segmentation method: `none`

## Hyperparameters

| parameter | value |
| --- | --- |
| seed | 42 |
| learning_rate | 2e-05 |
| num_train_epochs | 4 |
| per_device_train_batch_size | 16 |
| per_device_eval_batch_size | 32 |
| weight_decay | 0.01 |
| warmup_ratio | 0.06 |
| logging_steps | 50 |
| evaluation_strategy | epoch |
| save_strategy | epoch |
| load_best_model_at_end | True |
| metric_for_best_model | macro_f1 |
| greater_is_better | True |
| fp16 | True |
| report_to | none |


## PhoBERT results

| task | model_name | split | accuracy | macro_f1 | weighted_f1 | num_eval_samples | train_time_sec | predict_time_sec | num_train_samples | model_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sentiment | phobert_base | dev | 0.9419 | 0.8423 | 0.9402 | 1583 | 742.256 | 7.4622 | 11426 | /kaggle/working/nlp-phobert-student-feedback/models/phobert/sentiment/best_model |
| sentiment | phobert_base | test | 0.9324 | 0.8295 | 0.9312 | 3166 | 742.256 | 14.0307 | 11426 | /kaggle/working/nlp-phobert-student-feedback/models/phobert/sentiment/best_model |
| topic | phobert_base | dev | 0.897 | 0.7995 | 0.8937 | 1583 | 748.1372 | 7.5017 | 11426 | /kaggle/working/nlp-phobert-student-feedback/models/phobert/topic/best_model |
| topic | phobert_base | test | 0.8992 | 0.8052 | 0.8969 | 3166 | 748.1372 | 14.1736 | 11426 | /kaggle/working/nlp-phobert-student-feedback/models/phobert/topic/best_model |


## PhoBERT vs best Stage 2 baseline

| task | best_baseline_model | baseline_test_accuracy | baseline_test_macro_f1 | baseline_test_weighted_f1 | phobert_test_accuracy | phobert_test_macro_f1 | phobert_test_weighted_f1 | delta_macro_f1 | delta_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sentiment | tfidf_char_svm | 0.874 | 0.7354 | 0.8755 | 0.9324 | 0.8295 | 0.9312 | 0.0941 | 0.0584 |
| topic | tfidf_word_svm | 0.8585 | 0.7509 | 0.8598 | 0.8992 | 0.8052 | 0.8969 | 0.0543 | 0.0407 |


## Notes

- Test results are used only for final reporting, not for model selection.
- Macro-F1 remains the primary metric because both tasks are class-imbalanced.
- No noisy-data evaluation is performed in Stage 3.
- The current default config does not apply external Vietnamese word segmentation; this should be reported as a controlled implementation choice and revisited in the segmentation/tokenization analysis stage.
