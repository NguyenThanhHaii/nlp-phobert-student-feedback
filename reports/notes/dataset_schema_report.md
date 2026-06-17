# Dataset Schema Report

- Dataset name: `uitnlp/vietnamese_students_feedback`
- Checked at: `2026-06-18T00:31:35`
- Project root: `d:\project-ml-engineering\nlp-phobert-student-feedback`

## Available splits

| split | num_samples | has_sentiment | has_topic | num_columns |
| --- | --- | --- | --- | --- |
| train | 11426 | True | True | 9 |
| dev | 1583 | True | True | 9 |
| test | 3166 | True | True | 9 |


## Raw columns by split

### train
- Shape: `(11426, 3)`
- Columns: `['sentence', 'sentiment', 'topic']`

### validation
- Shape: `(1583, 3)`
- Columns: `['sentence', 'sentiment', 'topic']`

### test
- Shape: `(3166, 3)`
- Columns: `['sentence', 'sentiment', 'topic']`

## Selected project schema

| project_field | source_column |
| --- | --- |
| text | sentence |
| sentiment_label | sentiment |
| topic_label | topic |


## Label mapping

```json
{
  "dataset_name": "uitnlp/vietnamese_students_feedback",
  "created_at": "2026-06-18T00:31:29",
  "sentiment": {
    "source_column": "sentiment",
    "verified_from_hf_features": true,
    "id_to_name": {
      "0": "negative",
      "1": "neutral",
      "2": "positive"
    }
  },
  "topic": {
    "source_column": "topic",
    "verified_from_hf_features": true,
    "id_to_name": {
      "0": "lecturer",
      "1": "training_program",
      "2": "facility",
      "3": "others"
    }
  }
}
```

## Missing values

| column | missing_count | missing_percentage |
| --- | --- | --- |
| id | 0 | 0.0 |
| split | 0 | 0.0 |
| text | 0 | 0.0 |
| sentiment_label_raw | 0 | 0.0 |
| sentiment_label | 0 | 0.0 |
| topic_label_raw | 0 | 0.0 |
| topic_label | 0 | 0.0 |
| char_count | 0 | 0.0 |
| raw_word_count | 0 | 0.0 |


## Duplicate summary

```json
{
  "total_rows": 16175,
  "duplicated_full_rows": 0,
  "duplicated_text_rows": 1
}
```

## Label availability

- Sentiment label available: `True`
- Topic label available: `True`

## Preliminary PhoBERT tokenizer length analysis

- Available: `True`


## Stage 1 decision

Proceed with sentiment and topic classification.

