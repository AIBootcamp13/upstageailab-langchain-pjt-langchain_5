"""
LLM API 호출 및 응답 처리 모듈

이 모듈은 LangChain을 통해 LLM API를 호출하고 응답을 처리하는 기능을 제공합니다.

주요 기능:
1. LLM 모델 초기화 및 설정
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
from langchain_upstage import UpstageEmbeddings

from .logger import LoggerManager
from .prompt_loader import get_prompt_loader


class LLMManager:
    """LLM API 호출 및 응답 처리 클래스"""
    
    def __init__(self, 
                 api_key: str = None,
                 model: str = "solar-pro2",
                 reasoning_effort: str = "high",
                 temperature: float = 0.7):
        """
        LLMManager 초기화
        
        Args:
            api_key (str, optional): Upstage API 키. None이면 환경변수에서 가져옴
            model (str): 사용할 모델명
            reasoning_effort (str): 추론 노력 수준
            temperature (float): 응답 다양성 조절 (0.0~1.0)
        """
        self.logger = LoggerManager("LLM")
        self.api_key = api_key or os.getenv("UPSTAGE_API_KEY")
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        
        if not self.api_key:
            raise ValueError("UPSTAGE_API_KEY가 설정되지 않았습니다.")
        
        # LLM 초기화
        self._init_llm()
        
        # 프롬프트 로더 초기화
        self.prompt_loader = get_prompt_loader()
        
        # 기본 프롬프트 템플릿 설정
        self._init_default_prompt()
        
        self.logger.log_success("LLM Manager 초기화 완료")
    
    def _is_reasoning_model(self, model: str) -> bool:
        """
        모델이 reasoning 기능을 지원하는지 확인
        
        Args:
            model (str): 모델명
            
        Returns:
            bool: reasoning 모델 여부
        """
        reasoning_models = ["solar-pro", "solar-1-pro"]
        return any(rm in model.lower() for rm in reasoning_models)
    
    def _init_llm(self):
        """LLM 모델 초기화"""
        try:
            # 기본 파라미터 설정
            params = {
                "api_key": self.api_key,
                "model": self.model,
                "temperature": self.temperature
            }
            
            # reasoning 모델인 경우에만 reasoning_effort 추가
            if self._is_reasoning_model(self.model):
                params["reasoning_effort"] = self.reasoning_effort
                self.logger.log_step("Reasoning 모델 감지", f"reasoning_effort: {self.reasoning_effort}")
            else:
                self.logger.log_step("일반 모델 감지", "reasoning_effort 파라미터 제외")
            
            self.llm = ChatUpstage(**params)
            self.logger.log_step("LLM 모델 초기화", f"모델: {self.model}")
        except Exception as e:
            self.logger.log_error("LLM 초기화", e)
            raise
    
    def _init_default_prompt(self):
        """기본 프롬프트 템플릿 초기화 (Jinja2 템플릿 사용)"""
        self.logger.log_step("기본 프롬프트 템플릿 설정", "Jinja2 템플릿 기반으로 설정")
    
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
    
    def generate_response(self, 
                         question: str,
                         context: str = "",
                         chat_history: List[Dict] = None,
                         prompt_template: ChatPromptTemplate = None) -> str:
        """
        질문에 대한 응답 생성
        
        Args:
            question (str): 사용자 질문
            context (str): 검색된 컨텍스트
            chat_history (List[Dict], optional): 채팅 히스토리
            prompt_template (ChatPromptTemplate, optional): 커스텀 프롬프트
            
        Returns:
            str: 생성된 응답
        """
        self.logger.log_function_start("generate_response", 
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
            
            # LLM 호출
            response = self.llm.invoke(formatted_prompt)
            
            self.logger.log_function_end("generate_response", "응답 생성 완료")
            return response.content
            
        except Exception as e:
            self.logger.log_error("generate_response", e)
            return f"죄송합니다. 응답을 생성하는 중 오류가 발생했습니다: {str(e)}"
    
    def generate_response_stream(self, 
                               question: str,
                               context: str = "",
                               chat_history: List[Dict] = None,
                               prompt_template: ChatPromptTemplate = None) -> Generator[str, None, None]:
        """
        스트리밍 응답 생성
        
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
            for chunk in self.llm.stream(formatted_prompt):
                if chunk.content:
                    yield chunk.content
            
            self.logger.log_function_end("generate_response_stream")
            
        except Exception as e:
            self.logger.log_error("generate_response_stream", e)
            yield f"죄송합니다. 응답을 생성하는 중 오류가 발생했습니다: {str(e)}"
    
    def update_model_settings(self, 
                            model: str = None,
                            reasoning_effort: str = None,
                            temperature: float = None):
        """
        모델 설정 업데이트
        
        Args:
            model (str, optional): 새 모델명
            reasoning_effort (str, optional): 새 추론 노력 수준
            temperature (float, optional): 새 temperature 값
        """
        if model and model != self.model:
            self.model = model
            self.logger.log_step("모델 변경", f"새 모델: {model}")
        
        if reasoning_effort and reasoning_effort != self.reasoning_effort:
            self.reasoning_effort = reasoning_effort
            self.logger.log_step("추론 노력 수준 변경", f"새 수준: {reasoning_effort}")
        
        if temperature is not None and temperature != self.temperature:
            self.temperature = temperature
            self.logger.log_step("Temperature 변경", f"새 값: {temperature}")
        
        # LLM 재초기화
        self._init_llm()
    
    def get_model_info(self) -> Dict[str, Any]:
        """현재 모델 정보 반환"""
        return {
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "temperature": self.temperature,
            "api_key_set": bool(self.api_key)
        }
    
    def validate_api_connection(self) -> bool:
        """API 연결 상태 확인"""
        try:
            test_response = self.llm.invoke([HumanMessage(content="Hello")])
            self.logger.log_success("API 연결 확인 완료")
            return True
        except Exception as e:
            self.logger.log_error("API 연결 확인", e)
            return False