# Demo App Guide

## Purpose

This app is a lightweight academic demo for the NLP project:

**Sentiment Analysis and Topic Classification of Noisy Vietnamese Student Feedback using PhoBERT**

It demonstrates:

1. Input Vietnamese student feedback.
2. Optional rule-based normalization.
3. Sentiment prediction.
4. Topic prediction.
5. Model fallback from PhoBERT to TF-IDF/SVM baseline when PhoBERT is unavailable.

## How to run

From project root:

```powershell
.\scripts\run_demo.ps1
```

Then open:

```text
http://127.0.0.1:8000
```

## API endpoints

```text
GET  /health
GET  /model-info
POST /predict
```

Example request:

```json
{
  "text": "gv day de hieu nhung bt nhieu qua",
  "task": "both",
  "model": "auto",
  "normalize": true
}
```

## Model behavior

- `model = auto`: use PhoBERT if available, otherwise use baseline.
- `model = phobert`: force PhoBERT.
- `model = baseline`: force TF-IDF/SVM baseline.

Default baseline models:

```text
sentiment: tfidf_char_svm
topic: tfidf_word_svm
```

## Notes

This demo is not the main research result. The main evidence is in Stage 1–8 reports and tables.

The demo is for presentation only and should not be interpreted as an official teaching-quality evaluation tool.
