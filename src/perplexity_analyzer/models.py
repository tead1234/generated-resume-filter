from typing import Dict, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .utils import setup_logger

logger = setup_logger(__name__)


class ModelManager:
    """사전훈련된 모델들을 관리하는 클래스"""
    
    SUPPORTED_MODELS = {
        'kogpt2': 'skt/kogpt2-base-v2'
    }
    
    def __init__(self):
        self.models: Dict[str, tuple] = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def load_model(self, model_name: str) -> tuple:
        """모델과 토크나이저 로드"""
        if model_name in self.models:
            return self.models[model_name]
        
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_name}")
        
        model_path = self.SUPPORTED_MODELS[model_name]
        logger.info(f"Loading model: {model_path}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path)
            
            # 패딩 토큰 설정
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model.to(self.device)
            model.eval()
            
            self.models[model_name] = (model, tokenizer)
            logger.info(f"Successfully loaded {model_name}")
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
