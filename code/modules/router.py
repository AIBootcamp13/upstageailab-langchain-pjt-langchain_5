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


class QueryRouter:
    """질문 라우팅 클래스"""
    
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
        
        # 현재 시간 설정 (한국 시간)
        tz = pytz.timezone("Asia/Seoul")
        self.current_time = datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')
        
        # 라우팅용 경량 모델 초기화 (reasoning_effort 제외)
        self.router_llm = ChatUpstage(
            api_key=self.api_key,
            model=router_model,
            temperature=temperature
        )
        
        # 답변 생성 및 라우팅 프롬프트 템플릿
        self.classification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant that can answer general questions directly.
However, for specialized bakery and pastry technical questions, you need to access a knowledge base.
The current time is {current_time}.

YOUR TASK:
1. If the question is general (greetings, weather, programming, etc.) - ANSWER DIRECTLY in Korean
2. If the question requires specialized bakery/pastry knowledge - RESPOND WITH EXACTLY: "BAKERY-RAG"

IMPORTANT RULES FOR BAKERY-RAG:
- Output ONLY the text "BAKERY-RAG" - nothing else
- Do NOT include "BAKERY-RAG" in a sentence or explanation
- Do NOT say "This needs BAKERY-RAG" or similar phrases
- Do NOT add periods, explanations, or additional text
- Just output: BAKERY-RAG

BAKERY-RAG is needed for:
- Technical baking questions (recipes, techniques, troubleshooting)
- Baking ingredients and their functions  
- Professional bakery equipment and processes
- Bakery certifications and exams
- Food science related to baking
- Specific baking problem solving

ANSWER DIRECTLY for:
- General greetings and conversation
- Non-baking topics (weather, news, programming, etc.)
- Simple opinions about food
- Business questions (locations, hours, prices)
- General life advice

Examples:
Q: "안녕하세요"
A: 안녕하세요! 무엇을 도와드릴까요?

Q: "오늘 날씨 어때?"
A: 죄송하지만 실시간 날씨 정보는 제공할 수 없습니다. 날씨 앱이나 웹사이트를 확인해보시는 것을 추천드립니다.

Q: "크로와상 반죽이 왜 층이 안 생기죠?"
A: BAKERY-RAG

Q: "파이썬 배우고 싶어요"
A: 파이썬은 배우기 쉽고 강력한 프로그래밍 언어입니다. 온라인 강의나 공식 튜토리얼부터 시작하시는 것을 추천드립니다.

Q: "빵이 맛있어 보이네요"
A: 맛있어 보이는 빵을 발견하셨군요! 좋은 빵을 즐기시길 바랍니다.

Q: "이스트와 베이킹파우더 차이점이 뭐예요?"
A: BAKERY-RAG

Q: "좋은 하루 되세요"
A: 감사합니다! 좋은 하루 되세요!"""),
            ("human", "{question}")
        ])
        
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
            # 프롬프트 포맷팅
            messages = self.classification_prompt.format_messages(
                question=question,
                current_time=self.current_time
            )
            
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