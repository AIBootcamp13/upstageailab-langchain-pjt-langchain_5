"""
질문 라우팅 모듈

사용자 질문을 분석하여 제빵 관련 질문인지 일반 질문인지 판별하는 기능을 제공합니다.

주요 기능:
1. 질문 분류 (제빵 관련 / 일반)
2. 키워드 기반 사전 필터링
3. LLM 기반 정밀 분류
"""

import os
from typing import Dict, Any, Optional
from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

from .logger import LoggerManager


class QueryRouter:
    """질문 라우팅 클래스"""
    
    # 제빵/제과 관련 키워드
    BAKERY_KEYWORDS = [
        # 제빵 관련
        "빵", "제빵", "베이킹", "반죽", "발효", "굽기", "오븐",
        "이스트", "효모", "글루텐", "밀가루", "버터", "계란",
        "바게트", "크로와상", "식빵", "단팥빵", "소보로", "모카빵",
        "도우", "믹싱", "니딩", "프루핑", "벤치타임", "성형",
        
        # 제과 관련
        "제과", "케이크", "쿠키", "마카롱", "타르트", "파이",
        "초콜릿", "생크림", "커스터드", "머랭", "슈크림",
        "스펀지", "시폰", "무스", "가나슈", "버터크림",
        
        # 재료 및 도구
        "박력분", "중력분", "강력분", "베이킹파우더", "베이킹소다",
        "바닐라", "코코아", "시나몬", "설탕", "소금",
        "믹서기", "반죽기", "오븐", "발효기", "온도계",
        
        # 기능사 관련
        "기능사", "실기", "시험", "자격증", "제과기능사", "제빵기능사"
    ]
    
    def __init__(self, 
                 api_key: str = None,
                 router_model: str = "upstage/solar-1-mini-chat",
                 temperature: float = 0.1):
        """
        QueryRouter 초기화
        
        Args:
            api_key: Upstage API 키
            router_model: 라우팅에 사용할 모델
            temperature: 응답 다양성 (낮을수록 일관성 높음)
        """
        self.logger = LoggerManager("QueryRouter")
        self.api_key = api_key or os.getenv("UPSTAGE_API_KEY")
        
        if not self.api_key:
            raise ValueError("UPSTAGE_API_KEY가 설정되지 않았습니다.")
        
        # 라우팅용 경량 모델 초기화 (reasoning_effort 제외)
        self.router_llm = ChatUpstage(
            api_key=self.api_key,
            model=router_model,
            temperature=temperature
        )
        
        # 분류 프롬프트 템플릿
        self.classification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query classifier for a bakery assistant system.
Your task is to determine if a user question is related to baking/bakery topics.

Bakery-related topics include:
- Bread making, baking techniques, recipes
- Pastry, cakes, cookies, desserts
- Baking ingredients (flour, yeast, butter, etc.)
- Baking equipment and tools
- Baking certifications and exams
- Any questions about the bakery documents in the system

Reply with ONLY one word:
- "BAKERY" if the question is related to baking/bakery
- "GENERAL" if the question is not related to baking/bakery

Examples:
Q: "How to make croissant dough?" -> BAKERY
Q: "What's the weather today?" -> GENERAL
Q: "Tell me about fermentation process" -> BAKERY
Q: "Who is the president?" -> GENERAL"""),
            ("human", "{question}")
        ])
        
        self.logger.log_success("Query Router 초기화 완료")
    
    def check_keywords(self, question: str) -> bool:
        """
        키워드 기반 사전 필터링
        
        Args:
            question: 사용자 질문
            
        Returns:
            bool: 제빵 관련 키워드 포함 여부
        """
        question_lower = question.lower()
        
        for keyword in self.BAKERY_KEYWORDS:
            if keyword.lower() in question_lower:
                self.logger.log_step("키워드 매칭", f"발견된 키워드: {keyword}")
                return True
        
        return False
    
    def classify_with_llm(self, question: str) -> str:
        """
        LLM을 사용한 정밀 분류
        
        Args:
            question: 사용자 질문
            
        Returns:
            str: "BAKERY" 또는 "GENERAL"
        """
        try:
            # 프롬프트 포맷팅
            messages = self.classification_prompt.format_messages(question=question)
            
            # LLM 호출
            response = self.router_llm.invoke(messages)
            classification = response.content.strip().upper()
            
            # 유효성 검사
            if classification not in ["BAKERY", "GENERAL"]:
                self.logger.log_warning("예상치 못한 분류 결과", classification)
                # 키워드 체크로 폴백
                return "BAKERY" if self.check_keywords(question) else "GENERAL"
            
            return classification
            
        except Exception as e:
            self.logger.log_error("LLM 분류 오류", e)
            # 에러 시 키워드 체크로 폴백
            return "BAKERY" if self.check_keywords(question) else "GENERAL"
    
    def route(self, question: str) -> Dict[str, Any]:
        """
        질문 라우팅 수행
        
        Args:
            question: 사용자 질문
            
        Returns:
            Dict: {
                "type": "BAKERY" or "GENERAL",
                "use_rag": bool,
                "model": str (추천 모델),
                "confidence": float (0-1)
            }
        """
        self.logger.log_function_start("route", question=question[:50] + "..." if len(question) > 50 else question)
        
        # 1단계: 키워드 체크 (빠른 필터링)
        has_keywords = self.check_keywords(question)
        
        # 2단계: LLM 분류 (정밀 분석)
        llm_classification = self.classify_with_llm(question)
        
        # 3단계: 최종 결정
        if has_keywords and llm_classification == "BAKERY":
            # 높은 신뢰도로 제빵 관련
            result = {
                "type": "BAKERY",
                "use_rag": True,
                "model": "solar-pro2",
                "confidence": 0.95
            }
        elif has_keywords or llm_classification == "BAKERY":
            # 중간 신뢰도로 제빵 관련
            result = {
                "type": "BAKERY", 
                "use_rag": True,
                "model": "solar-pro2",
                "confidence": 0.7
            }
        else:
            # 일반 질문
            result = {
                "type": "GENERAL",
                "use_rag": False,
                "model": "upstage/solar-1-mini-chat",
                "confidence": 0.9
            }
        
        self.logger.log_function_end("route", f"분류 결과: {result['type']} (신뢰도: {result['confidence']})")
        return result
    
    def explain_route(self, question: str) -> str:
        """
        라우팅 결정에 대한 설명 생성
        
        Args:
            question: 사용자 질문
            
        Returns:
            str: 라우팅 결정 설명
        """
        result = self.route(question)
        
        if result["type"] == "BAKERY":
            return f"제빵 관련 질문으로 분류되었습니다. (신뢰도: {result['confidence']*100:.0f}%) RAG 시스템을 사용하여 문서에서 정보를 검색합니다."
        else:
            return f"일반 질문으로 분류되었습니다. (신뢰도: {result['confidence']*100:.0f}%) 일반 대화 모델로 응답합니다."