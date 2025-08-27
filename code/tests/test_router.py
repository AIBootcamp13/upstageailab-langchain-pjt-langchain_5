"""
QueryRouter 테스트 모듈

라우터의 질문 분류 기능을 테스트합니다.
"""

import pytest
import sys
import os
from pathlib import Path

# 테스트를 위한 경로 설정
test_dir = Path(__file__).parent
code_dir = test_dir.parent
sys.path.insert(0, str(code_dir))

from modules.router import QueryRouter


class TestQueryRouter:
    """QueryRouter 테스트 클래스"""
    
    @pytest.fixture
    def router(self):
        """QueryRouter 인스턴스 생성"""
        try:
            return QueryRouter()
        except ValueError:
            pytest.skip("UPSTAGE_API_KEY가 설정되지 않아 테스트를 스킵합니다.")
    
    def test_bakery_keywords_detection(self, router):
        """제빵 관련 키워드 감지 테스트"""
        bakery_questions = [
            "빵 만드는 방법을 알려주세요",
            "반죽 발효 시간은 얼마나 되나요?",
            "크로와상 만들기 어려운가요?",
            "제빵기능사 시험 준비 중입니다",
            "버터 크림 만들기",
            "케이크 구우는 온도",
            "이스트 종류에 대해 설명해주세요"
        ]
        
        for question in bakery_questions:
            has_keywords = router.check_keywords(question)
            assert has_keywords, f"제빵 관련 질문에서 키워드를 감지하지 못했습니다: {question}"
    
    def test_general_keywords_detection(self, router):
        """일반 질문 키워드 감지 테스트"""
        general_questions = [
            "오늘 날씨가 어떤가요?",
            "파이썬 프로그래밍을 배우고 싶어요",
            "주식 투자 방법을 알려주세요", 
            "축구 경기 결과가 궁금합니다",
            "영화 추천해주세요",
            "건강한 운동법을 알려주세요"
        ]
        
        for question in general_questions:
            has_keywords = router.check_keywords(question)
            assert not has_keywords, f"일반 질문에서 제빵 키워드가 감지되었습니다: {question}"
    
    def test_routing_bakery_questions(self, router):
        """제빵 관련 질문 라우팅 테스트"""
        bakery_questions = [
            "식빵 만드는 법을 알려주세요",
            "반죽이 잘 부풀지 않아요",
            "제과기능사 시험 팁이 있나요?",
            "오븐 온도는 몇 도로 설정해야 하나요?"
        ]
        
        for question in bakery_questions:
            result = router.route(question)
            assert result["type"] == "BAKERY", f"제빵 질문이 잘못 분류되었습니다: {question}"
            assert result["use_rag"] == True, f"제빵 질문에서 RAG 사용이 False입니다: {question}"
            assert result["model"] == "solar-pro2", f"제빵 질문에서 잘못된 모델이 선택되었습니다: {question}"
            assert result["confidence"] > 0.5, f"신뢰도가 너무 낮습니다: {question}"
    
    def test_routing_general_questions(self, router):
        """일반 질문 라우팅 테스트"""
        general_questions = [
            "안녕하세요",
            "오늘 날씨는 어때요?",
            "파이썬 배우는 방법",
            "좋은 책 추천해주세요"
        ]
        
        for question in general_questions:
            result = router.route(question)
            assert result["type"] == "GENERAL", f"일반 질문이 잘못 분류되었습니다: {question}"
            assert result["use_rag"] == False, f"일반 질문에서 RAG 사용이 True입니다: {question}"
            assert result["model"] == "upstage/solar-1-mini-chat", f"일반 질문에서 잘못된 모델이 선택되었습니다: {question}"
            assert result["confidence"] > 0.5, f"신뢰도가 너무 낮습니다: {question}"
    
    def test_edge_cases(self, router):
        """경계 사례 테스트"""
        edge_cases = [
            ("", "GENERAL"),  # 빈 문자열
            ("빵", "BAKERY"),  # 단일 키워드
            ("빵집에서 일하는데 날씨가 궁금해요", "BAKERY"),  # 혼합 질문
            ("제과제빵과는 전혀 관련없는 이야기", "GENERAL")  # 부정적 언급
        ]
        
        for question, expected_type in edge_cases:
            result = router.route(question)
            assert result["type"] == expected_type, f"경계 사례 분류 실패: '{question}' -> 예상: {expected_type}, 실제: {result['type']}"
    
    def test_explain_route(self, router):
        """라우팅 설명 기능 테스트"""
        bakery_question = "빵 만드는 방법"
        general_question = "날씨 정보"
        
        bakery_explanation = router.explain_route(bakery_question)
        general_explanation = router.explain_route(general_question)
        
        assert "제빵 관련" in bakery_explanation, "제빵 질문 설명에 적절한 내용이 없습니다"
        assert "일반 질문" in general_explanation, "일반 질문 설명에 적절한 내용이 없습니다"
        assert "%" in bakery_explanation, "신뢰도 정보가 없습니다"
        assert "%" in general_explanation, "신뢰도 정보가 없습니다"
    
    def test_router_consistency(self, router):
        """라우터 일관성 테스트 - 같은 질문에 대해 일관된 결과를 반환하는지 확인"""
        test_question = "빵 반죽 하는 방법을 알려주세요"
        
        results = []
        for _ in range(3):
            result = router.route(test_question)
            results.append(result["type"])
        
        # 모든 결과가 동일해야 함
        assert all(r == results[0] for r in results), f"라우터 결과가 일관되지 않습니다: {results}"
    
    def test_confidence_scores(self, router):
        """신뢰도 점수 테스트"""
        high_confidence_bakery = "제빵기능사 실기시험 준비방법"
        high_confidence_general = "오늘 주식시장 동향"
        
        bakery_result = router.route(high_confidence_bakery)
        general_result = router.route(high_confidence_general)
        
        assert bakery_result["confidence"] > 0.7, "제빵 질문의 신뢰도가 낮습니다"
        assert general_result["confidence"] > 0.7, "일반 질문의 신뢰도가 낮습니다"


class TestQueryRouterIntegration:
    """QueryRouter 통합 테스트"""
    
    @pytest.fixture
    def router(self):
        """QueryRouter 인스턴스 생성"""
        try:
            return QueryRouter()
        except ValueError:
            pytest.skip("UPSTAGE_API_KEY가 설정되지 않아 테스트를 스킵합니다.")
    
    def test_real_world_scenarios(self, router):
        """실제 사용 시나리오 테스트"""
        scenarios = [
            # 제빵 관련 실제 질문들
            {
                "question": "크로와상을 만들 때 버터가 새어나오는 이유는 뭔가요?",
                "expected_type": "BAKERY",
                "expected_rag": True
            },
            {
                "question": "제과제빵기능사 실기시험에서 자주 나오는 문제는?",
                "expected_type": "BAKERY", 
                "expected_rag": True
            },
            {
                "question": "반죽 발효가 잘 안될 때 해결방법",
                "expected_type": "BAKERY",
                "expected_rag": True
            },
            # 일반 질문들
            {
                "question": "안녕하세요! 처음 사용해봅니다",
                "expected_type": "GENERAL",
                "expected_rag": False
            },
            {
                "question": "내일 비가 올까요?",
                "expected_type": "GENERAL",
                "expected_rag": False
            },
            {
                "question": "추천할 만한 영화가 있나요?",
                "expected_type": "GENERAL",
                "expected_rag": False
            }
        ]
        
        for scenario in scenarios:
            result = router.route(scenario["question"])
            
            assert result["type"] == scenario["expected_type"], \
                f"시나리오 분류 실패: '{scenario['question']}'"
            assert result["use_rag"] == scenario["expected_rag"], \
                f"RAG 사용 설정 실패: '{scenario['question']}'"


if __name__ == "__main__":
    pytest.main([__file__])