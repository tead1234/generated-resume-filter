import unittest
import sys
import os

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.perplexity_analyzer import PerplexityAnalyzer


class TestPerplexityAnalyzer(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """테스트 클래스 초기화 (모델 로딩은 한 번만)"""
        cls.analyzer = PerplexityAnalyzer(model_name='gpt2')
    
    def test_analyzer_initialization(self):
        """분석기 초기화 테스트"""
        self.assertIsNotNone(self.analyzer.model)
        self.assertIsNotNone(self.analyzer.tokenizer)
        self.assertEqual(self.analyzer.LOG_PPL_THRESHOLD, 1.474)
    
    def test_log_perplexity_calculation(self):
        """로그 perplexity 계산 테스트"""
        test_text = "This is a simple test sentence."
        log_ppl = self.analyzer.calculate_log_perplexity(test_text)
        
        self.assertIsInstance(log_ppl, float)
        self.assertGreater(log_ppl, 0)
        self.assertNotEqual(log_ppl, float('inf'))
    
    def test_sentence_classification(self):
        """문장 분류 테스트"""
        # AI 의심 문장 (낮은 로그 perplexity)
        ai_result = self.analyzer.classify_sentence(1.0)
        self.assertEqual(ai_result['classification'], 'AI_SUSPICIOUS')
        self.assertTrue(ai_result['ai_suspicious'])
        self.assertGreater(ai_result['confidence'], 0)
        
        # 자연스러운 문장 (높은 로그 perplexity)
        natural_result = self.analyzer.classify_sentence(2.0)
        self.assertEqual(natural_result['classification'], 'NATURAL')
        self.assertFalse(natural_result['ai_suspicious'])
        
        # 에러 케이스
        error_result = self.analyzer.classify_sentence(float('inf'))
        self.assertEqual(error_result['classification'], 'ERROR')
    
    def test_analyze_sentences(self):
        """전체 텍스트 분석 테스트"""
        test_text = """
        안녕하세요. 저는 컴퓨터 공학을 전공했습니다. 
        프로그래밍에 대한 열정이 있습니다. 
        새로운 기술을 배우는 것을 좋아합니다.
        """
        
        result = self.analyzer.analyze_sentences(test_text)
        
        # 결과 구조 검증
        self.assertIn('ai_suspicious_sentences', result)
        self.assertIn('natural_sentences', result)
        self.assertIn('overall_stats', result)
        self.assertIn('recommendations', result)
        
        # 통계 검증
        stats = result['overall_stats']
        self.assertGreater(stats['total_sentences'], 0)
        self.assertEqual(
            stats['total_sentences'],
            stats['ai_suspicious_count'] + stats['natural_count'] + stats['error_count']
        )
    
    def test_empty_text(self):
        """빈 텍스트 처리 테스트"""
        result = self.analyzer.analyze_sentences("")
        
        self.assertEqual(len(result['ai_suspicious_sentences']), 0)
        self.assertEqual(len(result['natural_sentences']), 0)
        self.assertEqual(result['overall_stats']['total_sentences'], 0)
    
    def test_batch_analysis(self):
        """배치 분석 테스트"""
        test_texts = [
            "첫 번째 테스트 문장입니다.",
            "두 번째 테스트 문장입니다.",
            "세 번째 테스트 문장입니다."
        ]
        
        results = self.analyzer.analyze_batch(test_texts)
        
        self.assertEqual(len(results), 3)
        for i, result in enumerate(results):
            self.assertEqual(result['text_id'], i)
            self.assertIn('overall_stats', result)
    
    def test_model_info(self):
        """모델 정보 테스트"""
        info = self.analyzer.get_model_info()
        
        self.assertIn('model_name', info)
        self.assertIn('device', info)
        self.assertIn('log_ppl_threshold', info)
        self.assertEqual(info['model_name'], 'gpt2')


if __name__ == '__main__':
    unittest.main()