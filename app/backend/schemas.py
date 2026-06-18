from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


TaskName = Literal["sentiment", "topic", "both"]
ModelChoice = Literal["auto", "phobert", "baseline"]


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Vietnamese student feedback text.")
    task: TaskName = Field(default="both")
    model: ModelChoice = Field(default="auto")
    normalize: bool = Field(default=False)


class TaskPrediction(BaseModel):
    task: Literal["sentiment", "topic"]
    model_type: str
    model_name: str
    label: str
    confidence: float | None = None
    scores: dict[str, float] | None = None


class PredictResponse(BaseModel):
    input_text: str
    normalized_text: str | None = None
    used_text: str
    model_choice: ModelChoice
    normalize: bool
    predictions: list[TaskPrediction]
    warnings: list[str] = []
    disclaimer: str
