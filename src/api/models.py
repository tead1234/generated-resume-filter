from pydantic import BaseModel, Field
from typing import List, Dict, Union, Optional


class TextRequest(BaseModel):
    text: str = Field(..., description="분석할 텍스트")


class BatchTextRequest(BaseModel):
    texts: List[str] = Field(..., description="분석할 텍스트 목록")


class SentenceResult(BaseModel):
    text: str
    perplexity: float
    position: int
    classification: str
    confidence: float
    ai_suspicious: bool


class OverallStats(BaseModel):
    total_sentences: int
    ai_suspicious_count: int
    natural_count: int
    error_count: int
    ai_ratio: float


class AnalysisResult(BaseModel):
    ai_suspicious_sentences: List[SentenceResult]
    natural_sentences: List[SentenceResult]
    error_sentences: List[SentenceResult]
    overall_stats: OverallStats
    recommendations: List[str]


class BatchAnalysisResult(BaseModel):
    text_id: int
    result: AnalysisResult


class ModelInfo(BaseModel):
    model_name: str
    device: str
    max_length: int
    perplexity_threshold: float


class HealthResponse(BaseModel):
    status: str
    message: str