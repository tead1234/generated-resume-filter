from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import logging
from typing import Dict, Any

from ..perplexity_analyzer.analyzer import PerplexityAnalyzer
from .models import (
    TextRequest, BatchTextRequest, AnalysisResult, 
    BatchAnalysisResult, ModelInfo, HealthResponse
)

# 전역 analyzer 변수
analyzer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작시 모델 로드
    global analyzer
    logging.info("Loading PerplexityAnalyzer...")
    analyzer = PerplexityAnalyzer(model_name='kogpt2')
    logging.info("PerplexityAnalyzer loaded successfully")
    
    yield
    
    # 종료시 정리 (필요시)
    logging.info("Shutting down...")

app = FastAPI(
    title="Resume AI Filter API",
    description="AI로 생성된 자소서를 탐지하는 API",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """서버 상태 확인"""
    return HealthResponse(
        status="healthy",
        message="Resume AI Filter API is running"
    )


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """현재 사용 중인 모델 정보 조회"""
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")
    
    return ModelInfo(**analyzer.get_model_info())


@app.post("/analyze", response_model=AnalysisResult)
async def analyze_text(request: TextRequest):
    """단일 텍스트의 AI 생성 여부 분석"""
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")
    
    try:
        result = analyzer.analyze_sentences(request.text)
        return AnalysisResult(**result)
    except Exception as e:
        logging.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/analyze/batch", response_model=list[BatchAnalysisResult])
async def analyze_batch(request: BatchTextRequest):
    """여러 텍스트의 AI 생성 여부 배치 분석"""
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")
    
    try:
        results = analyzer.analyze_batch(request.texts)
        return [
            BatchAnalysisResult(text_id=result["text_id"], result=AnalysisResult(**result))
            for result in results
        ]
    except Exception as e:
        logging.error(f"Batch analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "Resume AI Filter API", 
        "docs": "/docs",
        "health": "/health"
    }