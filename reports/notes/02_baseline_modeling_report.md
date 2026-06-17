# Baseline Modeling Report

- Created at: `2026-06-18T01:04:45`
- Data source: processed UIT-VSFC splits from Stage 1
- Main metric: Macro-F1
- Model selection split: dev
- Final reporting split: test

## Tasks

- Sentiment classification
- Topic classification

## Models

1. Majority Class Classifier
2. Word-level TF-IDF + Linear SVM
3. Character-level TF-IDF + Linear SVM

## Preprocessing policy

Only minimal text preprocessing is used: convert to string and strip whitespace. No accent removal, spelling correction, abbreviation normalization, or stopword removal is applied in this clean baseline stage.

## Baseline results on test split

| task | model_name | accuracy | macro_f1 | weighted_f1 | train_time_sec | predict_time_sec |
| --- | --- | --- | --- | --- | --- | --- |
| sentiment | majority_class | 0.5022 | 0.2229 | 0.3358 | 0.004 | 0.0004 |
| sentiment | tfidf_word_svm | 0.892 | 0.7289 | 0.887 | 0.4663 | 0.0873 |
| sentiment | tfidf_char_svm | 0.874 | 0.7354 | 0.8755 | 1.2359 | 0.2026 |
| topic | majority_class | 0.7233 | 0.2099 | 0.6072 | 0.0041 | 0.0001 |
| topic | tfidf_word_svm | 0.8585 | 0.7509 | 0.8598 | 0.3988 | 0.0551 |
| topic | tfidf_char_svm | 0.8326 | 0.7299 | 0.8396 | 1.3946 | 0.2103 |


## Best baseline by dev Macro-F1

| task | model_name | accuracy | macro_f1 | weighted_f1 |
| --- | --- | --- | --- | --- |
| sentiment | tfidf_char_svm | 0.8989 | 0.7706 | 0.9027 |
| topic | tfidf_word_svm | 0.8648 | 0.7615 | 0.8657 |


## Notes

- Accuracy is reported but not used as the main metric because both tasks are class-imbalanced.
- Macro-F1 is used to reflect minority-class performance more clearly.
- Confusion matrices and per-class reports are exported for later error analysis.
