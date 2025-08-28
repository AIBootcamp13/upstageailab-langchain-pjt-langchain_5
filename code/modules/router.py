"""
질문 라우팅 모듈

사용자 질문을 분석하여 제빵 관련 질문인지 일반 질문인지 판별하는 기능을 제공합니다.

주요 기능:
1. 질문 분류 (제빵 관련 / 일반)
2. 키워드 기반 사전 필터링
3. LLM 기반 정밀 분류
"""

import os
from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate

from .logger import LoggerManager
from .prompt_loader import get_prompt_loader
from .llm import LLMManager


class QueryRouter:
    """질문 라우팅 클래스"""

    def __init__(self, llm_manager: LLMManager):
        """
        QueryRouter 초기화

        Args:
            llm_manager (LLMManager): LLMManager 인스턴스
        """
        self.logger = LoggerManager("QueryRouter")
        self.llm_manager = llm_manager
        self.router_llm = self.llm_manager.router_llm
        
        # 라우터 모델 정보 로깅
        router_config = self.llm_manager.llm_config.get("router", {})
        self.router_model = router_config.get("model", "N/A")
        self.logger.log_step("Router LLM 설정", f"모델: {self.router_model}")

        # 프롬프트 로더 초기화
        self.prompt_loader = get_prompt_loader()
        
        self.logger.log_success("Query Router 초기화 완료")
    
    def route(self, question: str) -> Dict[str, Any]:
        """
        질문 라우팅 수행 (분류만 수행, 답변 생성 안함)
        
        Args:
            question: 사용자 질문
            
        Returns:
            Dict: {
                "type": "GENERAL" or "RAG",
                "use_rag": bool,
                "response": None (답변은 별도 LLM에서 생성)
            }
        """
        self.logger.log_function_start("route", question=question[:50] + "..." if len(question) > 50 else question)
        
        try:
            # Jinja2 템플릿을 사용하여 시스템 프롬프트 렌더링
            system_prompt = self.prompt_loader.render_routing_prompt(question)
            
            # ChatPromptTemplate 생성 
            classification_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{question}")
            ])
            
            # 프롬프트 포맷팅
            messages = classification_prompt.format_messages(question=question)
            
            # 최종 프롬프트를 로그에 출력
            self.logger.log_step("최종 라우팅 프롬프트", f"프롬프트 길이: {len(str(messages))}자")
            self.logger.log_step("시스템 프롬프트", system_prompt)
            self.logger.log_step("사용자 질문", question)
            
            # LLM 호출 전 대기
            self.llm_manager._apply_delay()

            # LLM 호출
            self.logger.log_step("라우터용 LLM 호출", f"모델: {self.router_model}")
            response = self.router_llm.invoke(messages)
            response_text = response.content.strip()
            
            # 분류 결과 확인
            if response_text == "일반답변":
                self.logger.log_step("일반답변 분류", "일반답변으로 분류됨")
                route_result = {
                    "type": "GENERAL",
                    "use_rag": False,
                    "response": None  # 일반답변용 LLM에서 생성할 예정
                }
            elif response_text == "RAG답변":
                self.logger.log_step("RAG답변 분류", "RAG답변으로 분류됨")
                route_result = {
                    "type": "RAG",
                    "use_rag": True,
                    "response": None  # RAG용 LLM에서 생성할 예정
                }
            else:
                # 예상치 못한 응답인 경우 기본값으로 RAG로 라우팅
                self.logger.log_warning("예상치 못한 라우팅 응답", f"응답: {response_text}, RAG로 라우팅")
                route_result = {
                    "type": "RAG",
                    "use_rag": True,
                    "response": None
                }
            
            self.logger.log_function_end("route", f"분류 결과: {route_result['type']}")
            return route_result
            
        except Exception as e:
            self.logger.log_error("라우팅 오류", e)
            # 에러 시 RAG로 폴백
            return {
                "type": "RAG",
                "use_rag": True,
                "response": None
            }
    
    def explain_route(self, question: str) -> str:
        """
        라우팅 결정에 대한 설명 생성
        
        Args:
            question: 사용자 질문
            
        Returns:
            str: 라우팅 결정 설명
        """
        result = self.route(question)
        
        if result["type"] == "RAG":
            return f"제빵 전문 지식이 필요한 질문으로 판단되어 RAG 시스템을 사용합니다."
        else:
            return f"일반 질문으로 판단되어 일반답변 시스템을 사용합니다."