from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.backend.schemas import PredictRequest, PredictResponse, TaskPrediction
from app.backend.model_service import (
    available_models,
    label_display,
    predict,
    simple_rule_normalize,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = PROJECT_ROOT / "app" / "frontend"

app = FastAPI(
    title="Vietnamese Student Feedback NLP Demo",
    description="Sentiment analysis and topic classification demo for noisy Vietnamese student feedback.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def index():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Frontend files not found. API is running."}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "nlp-feedback-demo",
    }


@app.get("/model-info")
def model_info():
    return available_models()


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    raw_text = request.text.strip()
    normalized_text = simple_rule_normalize(raw_text) if request.normalize else None
    used_text = normalized_text if request.normalize and normalized_text is not None else raw_text

    tasks = ["sentiment", "topic"] if request.task == "both" else [request.task]

    predictions = []
    warnings = []

    for task in tasks:
        result, task_warnings = predict(task=task, text=used_text, model_choice=request.model)
        warnings.extend(task_warnings)

        label = result["label"]
        scores = result.get("scores")

        if scores:
            scores = {
                f"{key} ({label_display(key)})": value
                for key, value in scores.items()
            }

        predictions.append(
            TaskPrediction(
                task=task,
                model_type=result["model_type"],
                model_name=result["model_name"],
                label=f"{label} ({label_display(label)})",
                confidence=result.get("confidence"),
                scores=scores,
            )
        )

    return PredictResponse(
        input_text=raw_text,
        normalized_text=normalized_text,
        used_text=used_text,
        model_choice=request.model,
        normalize=request.normalize,
        predictions=predictions,
        warnings=warnings,
        disclaimer=(
            "Demo chỉ phục vụ minh họa học thuật. Kết quả không phải đánh giá chính thức "
            "về chất lượng giảng dạy hoặc quyết định hành chính."
        ),
    )
