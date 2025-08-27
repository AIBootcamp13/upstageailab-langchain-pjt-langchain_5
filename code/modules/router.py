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
from datetime import datetime
import pytz
from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

from .logger import LoggerManager
from .prompt_loader import get_prompt_loader
from .config_loader import get_config_loader


class QueryRouter:
    """질문 라우팅 클래스"""
    
    def __init__(self, 
                 api_key: str = None,
                 use_config: bool = True,
                 router_model: str = None,
                 temperature: float = None):
        """
        QueryRouter 초기화
        
        Args:
            api_key: Upstage API 키
            use_config (bool): config.yaml 사용 여부 (기본: True)
            router_model: 라우팅에 사용할 모델 (config 우선)
            temperature: 응답 다양성 (config 우선, 낮을수록 일관성 높음)
        """
        self.logger = LoggerManager("QueryRouter")
        self.api_key = api_key or os.getenv("UPSTAGE_API_KEY")
        
        if not self.api_key:
            raise ValueError("UPSTAGE_API_KEY가 설정되지 않았습니다.")
        
        # 설정 로드
        if use_config:
            try:
                config_loader = get_config_loader()
                llm_config = config_loader.get_llm_config()
                
                # 새로운 설정 구조 사용
                router_config = llm_config.get("router", {})
                self.router_model = router_config.get("model", router_model or "solar-pro2")
                self.temperature = router_config.get("temperature", temperature or 0.1)
                
                self.logger.log_step("Config 기반 Router 설정", 
                                   f"모델: {self.router_model}, temperature: {self.temperature}")
            except Exception as e:
                self.logger.log_warning(f"Config 로드 실패, 기본값 사용", str(e))
                self.router_model = router_model or "solar-pro2"
                self.temperature = temperature or 0.1
        else:
            self.router_model = router_model or "solar-pro2"
            self.temperature = temperature or 0.1
        
        # 라우팅용 모델 초기화
        self.router_llm = ChatUpstage(
            api_key=self.api_key,
            model=self.router_model,
            temperature=self.temperature
        )
        
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
            
            # LLM 호출
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