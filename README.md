# Vietnamese Student Feedback Classification with PhoBERT

This project studies sentiment analysis and topic classification for Vietnamese student feedback.

The main research focus is NLP experimentation:

- Dataset verification
- Exploratory data analysis
- Traditional text classification baselines
- PhoBERT fine-tuning
- Controlled noisy text evaluation
- Word segmentation and subword tokenization analysis
- Error analysis

The pp/ folder is reserved for a later FastAPI + React demonstration. It is not the main research contribution.

## Project structure

`
nlp-phobert-student-feedback/
├── app/
├── configs/
├── data/
├── docs/
├── models/
├── notebooks/
├── reports/
├── scripts/
├── src/
└── tests/
`

## Local environment

Recommended local environment:

- Windows
- VS Code
- Conda
- Python 3.10

Create the environment manually:

`powershell
conda env create -f environment.yml
conda activate nlp-phobert-feedback
python -m ipykernel install --user --name nlp-phobert-feedback --display-name "Python (nlp-phobert-feedback)"
`

Or run:

`powershell
.\setup_project.ps1 -CreateCondaEnv
`

## GPU strategy

Local development is CPU-based. PhoBERT fine-tuning will be done later on Kaggle GPU.

## Stage roadmap

1. Stage 0 — Project setup
2. Stage 1 — Dataset verification and EDA
3. Stage 2 — Clean baselines
4. Stage 3 — PhoBERT fine-tuning
5. Stage 4 — Controlled noise generation
6. Stage 5 — Segmentation and tokenization analysis
7. Stage 6 — Clean vs noisy evaluation
8. Stage 7 — Error analysis
9. Stage 8 — Optional normalization
10. Stage 9 — App prototype

## Reproducibility rules

- Do not assume dataset schema before verification.
- Do not commit raw data or model checkpoints.
- Keep all random seeds fixed where possible.
- Save tables to eports/tables/.
- Save figures to eports/figures/.
- Write project notes in English.
