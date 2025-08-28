"""
LLM API 호출 및 응답 처리 모듈

이 모듈은 LangChain을 통해 LLM API를 호출하고 응답을 처리하는 기능을 제공합니다.

주요 기능:
1. LLM 모델 초기화 및 설정 (router, general, rag 분리)
2. 프롬프트 템플릿 관리
3. API 호출 및 응답 처리
4. 스트리밍 응답 지원
"""

import os
from typing import List, Dict, Optional, Generator, Any
from datetime import datetime
import pytz

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_upstage import ChatUpstage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_upstage import UpstageEmbeddings

from .logger import LoggerManager
from .prompt_loader import get_prompt_loader
from .config_loader import get_config_loader


class LLMManager:
    """LLM API 호출 및 응답 처리 클래스"""
    
    def __init__(self, use_config: bool = True):
        """
        LLMManager 초기화
        
        Args:
            use_config (bool): config.yaml 사용 여부 (기본: True)
        """
        self.logger = LoggerManager("LLM")
        self.upstage_api_key = os.getenv("UPSTAGE_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        
        if use_config:
            try:
                config_loader = get_config_loader()
                self.llm_config = config_loader.get_llm_config()
            except Exception as e:
                self.logger.log_warning(f"Config 로드 실패, 기본값 사용", str(e))
                self.llm_config = self._get_default_llm_config()
        else:
            self.llm_config = self._get_default_llm_config()
        
        self._init_llms()
        self.prompt_loader = get_prompt_loader()
        self.logger.log_success("LLM Manager 초기화 완료")

    def _get_default_llm_config(self) -> Dict[str, Any]:
        """기본 LLM 설정 반환"""
        return {
            "router": {"provider": "upstage", "model": "solar-pro2", "temperature": 0.1},
            "general": {"provider": "upstage", "model": "solar-pro2", "temperature": 0.7},
            "rag": {"provider": "upstage", "model": "solar-pro2", "temperature": 0.7, "reasoning_effort": "high"}
        }

    def _create_llm_instance(self, role: str) -> Any:
        """설정에 따라 LLM 인스턴스 생성"""
        config = self.llm_config.get(role, {})
        provider = config.get("provider", "upstage")
        model_name = config.get("model")
        temperature = config.get("temperature")
        reasoning_effort = config.get("reasoning_effort")

        log_message = f"Provider: {provider}, Model: {model_name}"
        if "solar-pro2" in model_name.lower() and reasoning_effort:
            log_message += f", Reasoning Effort: {reasoning_effort}"

        self.logger.log_step(f"{role} LLM 생성", log_message)

        if provider == "google":
            if not self.google_api_key:
                raise ValueError(f"{role}에 google provider를 사용하려면 GOOGLE_API_KEY가 필요합니다.")
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                google_api_key=self.google_api_key,
            )

        elif provider == "upstage":
            if not self.upstage_api_key:
                raise ValueError(f"{role}에 upstage provider를 사용하려면 UPSTAGE_API_KEY가 필요합니다.")
            params = {
                "api_key": self.upstage_api_key,
                "model": model_name,
                "temperature": temperature,
            }
            if "solar-pro2" in model_name.lower() and reasoning_effort:
                params["reasoning_effort"] = reasoning_effort
            return ChatUpstage(**params)

        else:
            raise ValueError(f"지원하지 않는 LLM provider입니다: {provider}")

    def _init_llms(self):
        """각 용도별 LLM 모델 초기화"""
        try:
            self.router_llm = self._create_llm_instance("router")
            self.general_llm = self._create_llm_instance("general")
            self.rag_llm = self._create_llm_instance("rag")
            self.logger.log_success("모든 LLM 인스턴스 생성 완료")
        except Exception as e:
            self.logger.log_error("LLM 초기화 실패", e)
            raise
    
    def create_custom_prompt(self, 
                           system_message: str,
                           include_context: bool = True,
                           include_history: bool = True) -> ChatPromptTemplate:
        """
        커스텀 프롬프트 템플릿 생성
        
        Args:
            system_message (str): 시스템 메시지
            include_context (bool): 컨텍스트 포함 여부
            include_history (bool): 채팅 히스토리 포함 여부
            
        Returns:
            ChatPromptTemplate: 생성된 프롬프트 템플릿
        """
        messages = []
        
        # 시스템 메시지 구성
        if include_context:
            system_msg = f"{system_message}\n\nContext: {{context}}"
        else:
            system_msg = system_message
        
        messages.append(("system", system_msg))
        
        # 채팅 히스토리 포함
        if include_history:
            messages.append(MessagesPlaceholder(variable_name="chat_history"))
        
        # 사용자 메시지
        messages.append(("human", "{question}"))
        
        return ChatPromptTemplate.from_messages(messages)
    
    def format_chat_history(self, messages: List[Dict]) -> List:
        """
        메시지 리스트를 LangChain 메시지 객체로 변환
        
        Args:
            messages (List[Dict]): 메시지 리스트 [{"role": "user", "content": "..."}, ...]
            
        Returns:
            List: LangChain 메시지 객체 리스트
        """
        chat_history = []
        for msg in messages:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                chat_history.append(AIMessage(content=msg["content"]))
        
        return chat_history
    
    def generate_general_response(self, 
                                question: str,
                                chat_history: List[Dict] = None) -> str:
        """
        일반답변 생성
        
        Args:
            question (str): 사용자 질문
            chat_history (List[Dict], optional): 채팅 히스토리
            
        Returns:
            str: 생성된 일반답변
        """
        self.logger.log_function_start("generate_general_response", 
                                     question=question[:50] + "..." if len(question) > 50 else question)
        
        try:
            # 일반답변용 프롬프트 렌더링
            system_prompt = self.prompt_loader.render_template("general_response.jinja2", {"question": question})
            
            # 채팅 히스토리 변환
            formatted_history = []
            if chat_history:
                formatted_history = self.format_chat_history(chat_history)
            
            # ChatPromptTemplate 생성
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}")
            ])
            
            # 프롬프트 포맷팅
            formatted_prompt = prompt.format_messages(
                chat_history=formatted_history,
                question=question
            )
            
            # 일반답변용 LLM 호출
            self.logger.log_step("일반답변용 LLM 호출", f"모델: {self.general_model}")
            response = self.general_llm.invoke(formatted_prompt)
            
            self.logger.log_function_end("generate_general_response", "일반답변 생성 완료")
            return response.content
            
        except Exception as e:
            self.logger.log_error("generate_general_response", e)
            return f"죄송합니다. 일반답변을 생성하는 중 오류가 발생했습니다: {str(e)}"
    
    def generate_rag_response(self, 
                            question: str,
                            context: str = "",
                            chat_history: List[Dict] = None,
                            prompt_template: ChatPromptTemplate = None) -> str:
        """
        RAG 답변 생성 (기존 generate_response와 동일)
        
        Args:
            question (str): 사용자 질문
            context (str): 검색된 컨텍스트
            chat_history (List[Dict], optional): 채팅 히스토리
            prompt_template (ChatPromptTemplate, optional): 커스텀 프롬프트
            
        Returns:
            str: 생성된 RAG 답변
        """
        self.logger.log_function_start("generate_rag_response", 
                                     question=question[:50] + "..." if len(question) > 50 else question)
        
        try:
            if prompt_template:
                # 커스텀 프롬프트 템플릿 사용
                prompt = prompt_template
                
                # 채팅 히스토리 변환
                formatted_history = []
                if chat_history:
                    formatted_history = self.format_chat_history(chat_history)
                
                # 프롬프트 포맷팅
                formatted_prompt = prompt.format_messages(
                    context=context,
                    chat_history=formatted_history,
                    question=question
                )
            else:
                # 기본 Jinja2 템플릿 사용
                system_prompt = self.prompt_loader.render_rag_prompt(context, question)
                
                # 채팅 히스토리 변환
                formatted_history = []
                if chat_history:
                    formatted_history = self.format_chat_history(chat_history)
                
                # ChatPromptTemplate 생성
                default_prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{question}")
                ])
                
                # 프롬프트 포맷팅
                formatted_prompt = default_prompt.format_messages(
                    chat_history=formatted_history,
                    question=question
                )
            
            # RAG용 LLM 호출
            self.logger.log_step("RAG용 LLM 호출", f"모델: {self.rag_model}")
            response = self.rag_llm.invoke(formatted_prompt)
            
            self.logger.log_function_end("generate_rag_response", "RAG 답변 생성 완료")
            return response.content
            
        except Exception as e:
            self.logger.log_error("generate_rag_response", e)
            return f"죄송합니다. RAG 답변을 생성하는 중 오류가 발생했습니다: {str(e)}"
    
    def generate_response(self, 
                         question: str,
                         context: str = "",
                         chat_history: List[Dict] = None,
                         prompt_template: ChatPromptTemplate = None) -> str:
        """
        질문에 대한 응답 생성 (호환성을 위해 유지, RAG용으로 사용)
        
        Args:
            question (str): 사용자 질문
            context (str): 검색된 컨텍스트
            chat_history (List[Dict], optional): 채팅 히스토리
            prompt_template (ChatPromptTemplate, optional): 커스텀 프롬프트
            
        Returns:
            str: 생성된 응답
        """
        return self.generate_rag_response(question, context, chat_history, prompt_template)
    
    def generate_response_stream(self, 
                               question: str,
                               context: str = "",
                               chat_history: List[Dict] = None,
                               prompt_template: ChatPromptTemplate = None) -> Generator[str, None, None]:
        """
        스트리밍 응답 생성 (RAG용)
        
        Args:
            question (str): 사용자 질문
            context (str): 검색된 컨텍스트
            chat_history (List[Dict], optional): 채팅 히스토리
            prompt_template (ChatPromptTemplate, optional): 커스텀 프롬프트
            
        Yields:
            str: 스트리밍 응답 청크
        """
        self.logger.log_function_start("generate_response_stream")
        
        try:
            if prompt_template:
                # 커스텀 프롬프트 템플릿 사용
                prompt = prompt_template
                
                # 채팅 히스토리 변환
                formatted_history = []
                if chat_history:
                    formatted_history = self.format_chat_history(chat_history)
                
                # 프롬프트 포맷팅
                formatted_prompt = prompt.format_messages(
                    context=context,
                    chat_history=formatted_history,
                    question=question
                )
            else:
                # 기본 Jinja2 템플릿 사용
                system_prompt = self.prompt_loader.render_rag_prompt(context, question)
                
                # 채팅 히스토리 변환
                formatted_history = []
                if chat_history:
                    formatted_history = self.format_chat_history(chat_history)
                
                # ChatPromptTemplate 생성
                default_prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{question}")
                ])
                
                # 프롬프트 포맷팅
                formatted_prompt = default_prompt.format_messages(
                    chat_history=formatted_history,
                    question=question
                )
            
            # 스트리밍 응답
            self.logger.log_step("RAG용 LLM 스트리밍 호출", f"모델: {self.rag_model}")
            for chunk in self.rag_llm.stream(formatted_prompt):
                if chunk.content:
                    yield chunk.content
            
            self.logger.log_function_end("generate_response_stream")
            
        except Exception as e:
            self.logger.log_error("generate_response_stream", e)
            yield f"죄송합니다. 응답을 생성하는 중 오류가 발생했습니다: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """현재 모델 정보 반환"""
        return {
            "router": {
                "model": self.router_model,
                "temperature": self.router_temperature
            },
            "general": {
                "model": self.general_model,
                "temperature": self.general_temperature
            },
            "rag": {
                "model": self.rag_model,
                "temperature": self.rag_temperature,
                "reasoning_effort": self.rag_reasoning_effort
            },
            "api_key_set": bool(self.api_key)
        }
    
    def validate_api_connection(self) -> bool:
        """API 연결 상태 확인"""
        try:
            test_response = self.general_llm.invoke([HumanMessage(content="Hello")])
            self.logger.log_success("API 연결 확인 완료")
            return True
        except Exception as e:
            self.logger.log_error("API 연결 확인", e)
            return False