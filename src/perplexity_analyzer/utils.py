import re
import logging
from typing import List


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """로거 설정"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def preprocess_text(text: str) -> str:
    """텍스트 전처리"""
    # 불필요한 공백 제거
    text = re.sub(r'\s+', ' ', text.strip())
    
    # 특수문자 정리 (기본적인 것만)
    text = re.sub(r'[^\w\s가-힣.,!?]', '', text)
    
    return text


def split_into_sentences(text: str) -> List[str]:
    """텍스트를 문장 단위로 분할"""
    # 한국어 문장 분리 패턴
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def normalize_score(perplexity: float, min_ppl: float = 1.0, max_ppl: float = 1000.0) -> float:
    """Perplexity를 0-1 스케일로 정규화"""
    # 로그 스케일 적용
    import math
    log_ppl = math.log(max(perplexity, min_ppl))
    log_min = math.log(min_ppl)
    log_max = math.log(max_ppl)
    
    # 0-1 정규화 (1에 가까울수록 AI 생성 의심)
    normalized = (log_max - log_ppl) / (log_max - log_min)
    return max(0.0, min(1.0, normalized))