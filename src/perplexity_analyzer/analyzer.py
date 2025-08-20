import torch
import math
from typing import List, Dict, Optional, Union
from tqdm import tqdm
from .models import ModelManager
from .utils import setup_logger, preprocess_text, split_into_sentences

logger = setup_logger(__name__)


class PerplexityAnalyzer:
    """텍스트의 Perplexity를 분석하여 AI 생성 문장을 탐지하고 분류하는 클래스"""
    
    # 로그 perplexity 임계값 (1.474 이하면 AI 생성 의심)
    LOG_PPL_THRESHOLD = 1.474
    
    def __init__(self, model_name: str = 'gpt2', max_length: int = 512):
        """
        Args:
            model_name: 사용할 모델명 ('gpt2', 'kogpt2' 등)
            max_length: 토큰화시 최대 길이
        """
        self.model_name = model_name
        self.max_length = max_length
        self.model_manager = ModelManager()
        self.model, self.tokenizer = self.model_manager.load_model(model_name)
        
        logger.info(f"PerplexityAnalyzer initialized with {model_name}")
    
    def calculate_log_perplexity(self, text: str) -> float:
        """단일 텍스트의 로그 perplexity 계산"""
        if not text or not text.strip():
            return float('inf')
        
        # 텍스트 전처리
        processed_text = preprocess_text(text)
        
        # 토큰화
        inputs = self.tokenizer(
            processed_text,
            return_tensors='pt',
            max_length=self.max_length,
            truncation=True,
            padding=True
        )
        
        inputs = {k: v.to(self.model_manager.device) for k, v in inputs.items()}
        
        try:
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                log_perplexity = loss.item()  # 이미 자연로그값
                
            return log_perplexity
            
        except Exception as e:
            logger.error(f"Error calculating log perplexity: {e}")
            return float('inf')
    
    def classify_sentence(self, log_ppl: float) -> Dict[str, Union[str, float]]:
        """로그 perplexity 기반으로 문장 분류"""
        if log_ppl == float('inf'):
            return {
                'classification': 'ERROR',
                'confidence': 0.0,
                'ai_suspicious': False
            }
        
        if log_ppl <= self.LOG_PPL_THRESHOLD:
            # AI 생성 의심
            confidence = (self.LOG_PPL_THRESHOLD - log_ppl) / self.LOG_PPL_THRESHOLD
            confidence = max(0.0, min(1.0, confidence))
            
            return {
                'classification': 'AI_SUSPICIOUS',
                'confidence': confidence,
                'ai_suspicious': True
            }
        else:
            # 자연스러운 문장
            confidence = min(1.0, (log_ppl - self.LOG_PPL_THRESHOLD) / 2.0)
            
            return {
                'classification': 'NATURAL',
                'confidence': confidence,
                'ai_suspicious': False
            }
    
    def analyze_sentences(self, text: str) -> Dict:
        """문장별로 분석하고 AI 의심 문장들을 분류"""
        sentences = split_into_sentences(text)
        
        if not sentences:
            return {
                'ai_suspicious_sentences': [],
                'natural_sentences': [],
                'error_sentences': [],
                'overall_stats': {
                    'total_sentences': 0,
                    'ai_suspicious_count': 0,
                    'natural_count': 0,
                    'ai_ratio': 0.0
                }
            }
        
        ai_suspicious = []
        natural = []
        errors = []
        
        for i, sentence in enumerate(tqdm(sentences, desc="Analyzing sentences")):
            log_ppl = self.calculate_log_perplexity(sentence)
            classification = self.classify_sentence(log_ppl)
            
            sentence_data = {
                'text': sentence,
                'log_perplexity': log_ppl,
                'position': i,
                **classification
            }
            
            if classification['classification'] == 'AI_SUSPICIOUS':
                ai_suspicious.append(sentence_data)
            elif classification['classification'] == 'NATURAL':
                natural.append(sentence_data)
            else:
                errors.append(sentence_data)
        
        # AI 의심 문장들을 confidence 순으로 정렬 (높은 의심도부터)
        ai_suspicious.sort(key=lambda x: x['confidence'], reverse=True)
        
        # 통계 계산
        total_count = len(sentences)
        ai_count = len(ai_suspicious)
        ai_ratio = ai_count / total_count if total_count > 0 else 0.0
        
        return {
            'ai_suspicious_sentences': ai_suspicious,
            'natural_sentences': natural,
            'error_sentences': errors,
            'overall_stats': {
                'total_sentences': total_count,
                'ai_suspicious_count': ai_count,
                'natural_count': len(natural),
                'error_count': len(errors),
                'ai_ratio': ai_ratio
            },
            'recommendations': self._generate_recommendations(ai_suspicious, ai_ratio)
        }
    
    def _generate_recommendations(self, ai_suspicious: List[Dict], ai_ratio: float) -> List[str]:
        """수정 권장사항 생성"""
        recommendations = []
        
        if ai_ratio >= 0.5:
            recommendations.append("⚠️  전체 문장의 50% 이상이 AI 생성으로 의심됩니다. 전면 수정을 권장합니다.")
        elif ai_ratio >= 0.3:
            recommendations.append("⚠️  상당수 문장이 AI 생성으로 의심됩니다. 주요 문장들을 수정해주세요.")
        elif ai_ratio > 0:
            recommendations.append("✓ 일부 문장만 AI 생성으로 의심됩니다. 해당 문장들을 확인해주세요.")
        else:
            recommendations.append("✅ AI 생성이 의심되는 문장이 없습니다.")
        
        # 우선순위가 높은 문장들 (confidence > 0.8) 알림
        high_priority = [s for s in ai_suspicious if s['confidence'] > 0.8]
        if high_priority:
            positions = [str(s['position'] + 1) for s in high_priority[:3]]  # 최대 3개만
            recommendations.append(f"🔥 우선 수정 필요 문장: {', '.join(positions)}번")
        
        return recommendations
    
    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """여러 텍스트를 배치로 분석"""
        results = []
        
        for i, text in enumerate(tqdm(texts, desc="Analyzing batch")):
            logger.info(f"Analyzing text {i+1}/{len(texts)}")
            result = self.analyze_sentences(text)
            result['text_id'] = i
            results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict[str, Union[str, float]]:
        """현재 사용중인 모델 정보 반환"""
        return {
            'model_name': self.model_name,
            'device': str(self.model_manager.device),
            'max_length': self.max_length,
            'log_ppl_threshold': self.LOG_PPL_THRESHOLD
        }