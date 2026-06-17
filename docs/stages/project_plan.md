# Project Plan

## Stage 0 — Project setup

Create project structure, conda environment, config files, and notebook skeletons.

## Stage 1 — Dataset verification and EDA

Load the real dataset, verify columns, labels, encoding, and split policy.

## Stage 2 — Clean baselines

Train Majority Class, TF-IDF word-level + Linear SVM, and TF-IDF char-level + Linear SVM.

## Stage 3 — PhoBERT fine-tuning

Fine-tune PhoBERT on Kaggle GPU after max_length is selected from EDA.

## Stage 4 — Controlled noise generation

Generate clean/noisy test sets with rule-based transformations.

## Stage 5 — Segmentation/tokenization analysis

Measure segmentation change and subword inflation.

## Stage 6 — Clean vs noisy evaluation

Evaluate all models on clean and noisy test sets.

## Stage 7 — Error analysis

Analyze model errors by class, noise type, and segmentation/tokenization changes.

## Stage 8 — Optional normalization

Evaluate whether simple normalization recovers performance.

## Stage 9 — App prototype

Build a FastAPI + React demo under pp/.
