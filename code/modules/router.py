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
        
        # 설정 로드
        if use_config:
            try:
                config_loader = get_config_loader()
                router_config = config_loader.get_router_config()
                
                self.router_model = router_config.get("router_model", router_model or "upstage/solar-1-mini-chat")
                self.temperature = router_config.get("temperature", temperature or 0.1)
                
                self.logger.log_step("Config 기반 Router 설정", 
                                   f"모델: {self.router_model}, temperature: {self.temperature}")
            except Exception as e:
                self.logger.log_warning(f"Config 로드 실패, 기본값 사용", str(e))
                self.router_model = router_model or "upstage/solar-1-mini-chat"
                self.temperature = temperature or 0.1
        else:
            self.router_model = router_model or "upstage/solar-1-mini-chat"
            self.temperature = temperature or 0.1
        
        if not self.api_key:
            raise ValueError("UPSTAGE_API_KEY가 설정되지 않았습니다.")
        
        # 라우팅용 경량 모델 초기화 (reasoning_effort 제외)
        self.router_llm = ChatUpstage(
            api_key=self.api_key,
            model=self.router_model,
            temperature=self.temperature
        )
        
        # 프롬프트 로더 초기화
        self.prompt_loader = get_prompt_loader()
        
        self.logger.log_success("Query Router 초기화 완료")
    
    def get_response_or_route(self, question: str) -> Dict[str, Any]:
        """
        질문에 대해 직접 답변하거나 RAG 라우팅 신호를 반환
        
        Args:
            question: 사용자 질문
            
        Returns:
            Dict: {
                "response": str (직접 답변 또는 "BAKERY-RAG"),
                "is_bakery_rag": bool (RAG 필요 여부)
            }
        """
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
            
            # BAKERY-RAG 신호 체크 (여러 패턴 지원)
            response_clean = response_text.strip()
            
            # 1. 정확한 일치 (가장 명확한 경우)
            if response_clean == "BAKERY-RAG" or response_clean == "BAKERY-RAG.":
                self.logger.log_step("RAG 라우팅", "명확한 BAKERY-RAG 신호 감지")
                return {
                    "response": None,
                    "is_bakery_rag": True
                }
            
            # 2. 시작 패턴 (LLM이 추가 설명을 붙인 경우)
            elif response_clean.startswith("BAKERY-RAG"):
                self.logger.log_warning("RAG 라우팅", f"BAKERY-RAG로 시작하는 응답: {response_text[:50]}...")
                return {
                    "response": None,
                    "is_bakery_rag": True
                }
            
            # 3. 처음 몇 단어에 포함 (실수로 앞에 나온 경우)
            elif "BAKERY-RAG" in response_text.split()[:3]:
                self.logger.log_warning("RAG 라우팅", f"처음 3단어에 BAKERY-RAG 포함: {response_text[:50]}...")
                return {
                    "response": None,
                    "is_bakery_rag": True
                }
            
            # 4. 응답에 포함되어 있지만 일반 답변으로 보이는 경우 (마지막 수단)
            elif "BAKERY-RAG" in response_text:
                self.logger.log_warning("RAG 라우팅", f"응답 중간에 BAKERY-RAG 포함, RAG로 라우팅: {response_text[:100]}...")
                return {
                    "response": None,
                    "is_bakery_rag": True
                }
            
            else:
                # 완전한 일반 답변
                self.logger.log_step("일반 답변", f"직접 응답 생성: {response_text[:50]}...")
                return {
                    "response": response_text,
                    "is_bakery_rag": False
                }
            
        except Exception as e:
            self.logger.log_error("LLM 라우팅 오류", e)
            # 에러 시 일반 응답으로 폴백
            return {
                "response": "죄송합니다. 현재 응답을 처리하는 중 문제가 발생했습니다. 다시 시도해주세요.",
                "is_bakery_rag": False
            }
    
    def route(self, question: str) -> Dict[str, Any]:
        """
        질문 라우팅 수행 (호환성을 위해 유지)
        
        Args:
            question: 사용자 질문
            
        Returns:
            Dict: {
                "type": "BAKERY" or "GENERAL",
                "use_rag": bool,
                "response": str or None,
                "model": str (사용된 모델)
            }
        """
        self.logger.log_function_start("route", question=question[:50] + "..." if len(question) > 50 else question)
        
        # 새로운 방식으로 라우팅
        result = self.get_response_or_route(question)
        
        if result["is_bakery_rag"]:
            # RAG가 필요한 제빵 관련 질문
            route_result = {
                "type": "BAKERY",
                "use_rag": True,
                "response": None,  # RAG에서 생성할 예정
                "model": "solar-pro2"
            }
        else:
            # 직접 답변 가능한 일반 질문
            route_result = {
                "type": "GENERAL",
                "use_rag": False,
                "response": result["response"],  # 이미 생성된 답변
                "model": "upstage/solar-1-mini-chat"
            }
        
        self.logger.log_function_end("route", f"분류 결과: {route_result['type']} (응답: {'생성됨' if route_result['response'] else 'RAG 필요'})")
        return route_result
    
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
            return f"제빵 전문 지식이 필요한 질문으로 판단되어 RAG 시스템을 사용합니다. 모델: {result['model']}"
        else:
            return f"일반 질문으로 판단되어 직접 답변을 생성했습니다. 모델: {result['model']}"