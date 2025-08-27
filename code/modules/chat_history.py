"""
채팅 기록 관리 모듈

이 모듈은 사용자와 AI 간의 대화 기록을 관리하고 컨텍스트 윈도우를 조절하는 기능을 제공합니다.

주요 기능:
1. 메모리 관리 (ConversationBufferWindowMemory 래핑)
2. 채팅 히스토리 저장 및 로드
3. 컨텍스트 윈도우 크기 조절
4. SQLite와 연동한 영구 저장
"""

from typing import List, Dict, Optional, Tuple
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, AIMessage

from .sql import SQLManager
from .logger import LoggerManager


class ChatHistoryManager:
    """채팅 기록 관리 클래스"""
    
    def __init__(self, 
                 session_id: str = None,
                 memory_k: int = 3,
                 sql_manager: SQLManager = None,
                 auto_save: bool = True):
        """
        ChatHistoryManager 초기화
        
        Args:
            session_id (str, optional): 세션 ID. None이면 새 세션 생성 (sql_manager가 있을 때만)
            memory_k (int): 메모리에 보관할 최근 대화 수
            sql_manager (SQLManager, optional): SQL 관리자. None이면 메모리만 사용
            auto_save (bool): 자동 저장 여부
        """
        self.logger = LoggerManager("ChatHistory")
        self.memory_k = memory_k
        self.sql_manager = sql_manager
        self.auto_save = auto_save
        
        # 세션 ID 설정
        if session_id:
            self.session_id = session_id
            self.logger.log_step("기존 세션 로드", f"세션 ID: {session_id}")
        elif self.sql_manager:
            # SQL 관리자가 있을 때만 새 세션 생성
            self.session_id = self.sql_manager.create_conversation()
            self.logger.log_step("새 세션 생성", f"세션 ID: {self.session_id}")
        else:
            # 메모리만 사용하는 경우 (CLI/평가용)
            self.session_id = None
            self.logger.log_step("메모리 전용 모드", "세션 ID 없음")
        
        # 메모리 초기화
        self._init_memory()
        
        # 기존 채팅 히스토리 로드 (SQL 관리자가 있을 때만)
        if self.sql_manager and self.session_id:
            self._load_chat_history()
        
        self.logger.log_success("Chat History Manager 초기화 완료")
    
    def _init_memory(self):
        """LangChain 메모리 초기화"""
        try:
            self.memory = ConversationBufferWindowMemory(
                k=self.memory_k,
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            self.logger.log_step("메모리 초기화", f"윈도우 크기: {self.memory_k}")
        except Exception as e:
            self.logger.log_error("메모리 초기화", e)
            raise
    
    def _load_chat_history(self):
        """SQLite에서 채팅 히스토리 로드"""
        if not self.sql_manager or not self.session_id:
            return
            
        try:
            # 최근 메시지들을 메모리에 로드 (memory_k * 2개, 즉 대화 쌍 기준)
            recent_messages = self.sql_manager.get_recent_messages(
                self.session_id, 
                count=self.memory_k * 2
            )
            
            # 메모리에 대화 추가
            for i in range(0, len(recent_messages), 2):
                if i + 1 < len(recent_messages):
                    user_role, user_content = recent_messages[i]
                    ai_role, ai_content = recent_messages[i + 1]
                    
                    if user_role == "user" and ai_role == "assistant":
                        self.memory.save_context(
                            {"input": user_content},
                            {"answer": ai_content}
                        )
            
            self.logger.log_step("채팅 히스토리 로드", 
                               f"{len(recent_messages)}개 메시지 로드")
            
        except Exception as e:
            self.logger.log_error("채팅 히스토리 로드", e)
    
    def add_user_message(self, message: str) -> int:
        """
        사용자 메시지 추가
        
        Args:
            message (str): 사용자 메시지
            
        Returns:
            int: 메시지 ID (SQLite)
        """
        self.logger.log_function_start("add_user_message")
        
        try:
            # SQLite에 저장 (auto_save가 True이고 SQL 관리자가 있는 경우)
            message_id = None
            if self.auto_save and self.sql_manager and self.session_id:
                message_id = self.sql_manager.add_message(
                    self.session_id, "user", message
                )
            
            self.logger.log_function_end("add_user_message", f"메시지 ID: {message_id}")
            return message_id
            
        except Exception as e:
            self.logger.log_error("add_user_message", e)
            return None
    
    def add_ai_message(self, message: str, user_input: str = None, sources: List[str] = None) -> int:
        """
        AI 메시지 추가 및 메모리 업데이트
        
        Args:
            message (str): AI 응답 메시지
            user_input (str, optional): 대응하는 사용자 입력 (메모리 업데이트용)
            sources (List[str], optional): 참조한 문서 소스 목록
            
        Returns:
            int: 메시지 ID (SQLite)
        """
        self.logger.log_function_start("add_ai_message")
        
        try:
            # 메타데이터 준비
            metadata = {}
            if sources:
                metadata["sources"] = sources
                self.logger.log_step("소스 정보 추가", f"{len(sources)}개 소스")
            
            # SQLite에 저장 (auto_save가 True이고 SQL 관리자가 있는 경우)
            message_id = None
            if self.auto_save and self.sql_manager and self.session_id:
                message_id = self.sql_manager.add_message(
                    self.session_id, "assistant", message, metadata if metadata else None
                )
            
            # 메모리 업데이트 (user_input이 제공된 경우)
            if user_input:
                self.memory.save_context(
                    {"input": user_input},
                    {"answer": message}
                )
                self.logger.log_step("메모리 업데이트", "대화 쌍 추가")
            
            self.logger.log_function_end("add_ai_message", f"메시지 ID: {message_id}")
            return message_id
            
        except Exception as e:
            self.logger.log_error("add_ai_message", e)
            return None
    
    def add_conversation_pair(self, user_message: str, ai_message: str) -> Tuple[int, int]:
        """
        대화 쌍 추가 (사용자 메시지 + AI 응답)
        
        Args:
            user_message (str): 사용자 메시지
            ai_message (str): AI 응답
            
        Returns:
            Tuple[int, int]: (사용자 메시지 ID, AI 메시지 ID)
        """
        user_id = self.add_user_message(user_message)
        ai_id = self.add_ai_message(ai_message, user_message)
        
        return user_id, ai_id
    
    def get_chat_history_for_llm(self) -> List:
        """
        LLM용 채팅 히스토리 반환 (LangChain 메시지 형식)
        
        Returns:
            List: LangChain 메시지 객체 리스트
        """
        return self.memory.chat_memory.messages
    
    def get_chat_history_as_dicts(self) -> List[Dict]:
        """
        채팅 히스토리를 딕셔너리 리스트로 반환
        
        Returns:
            List[Dict]: [{"role": "user", "content": "..."}, ...] 형식
        """
        messages = []
        for msg in self.memory.chat_memory.messages:
            if isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"role": "assistant", "content": msg.content})
        
        return messages
    
    def get_full_conversation_history(self) -> List[Dict]:
        """
        전체 대화 기록을 SQLite에서 가져오기
        
        Returns:
            List[Dict]: 전체 메시지 리스트
        """
        try:
            if self.sql_manager and self.session_id:
                return self.sql_manager.get_messages(self.session_id)
            else:
                # 메모리 전용 모드: 메모리에서 가져오기
                memory_messages = self.get_chat_history_as_dicts()
                return memory_messages
        except Exception as e:
            self.logger.log_error("get_full_conversation_history", e)
            return []
    
    def clear_memory(self):
        """메모리 초기화 (SQLite 데이터는 유지)"""
        try:
            self.memory.clear()
            self.logger.log_step("메모리 초기화", "메모리 내용 삭제")
        except Exception as e:
            self.logger.log_error("clear_memory", e)
    
    def update_memory_window_size(self, new_k: int):
        """
        메모리 윈도우 크기 변경
        
        Args:
            new_k (int): 새로운 윈도우 크기
        """
        if new_k != self.memory_k:
            self.memory_k = new_k
            
            # 새로운 크기로 메모리 재초기화
            self._init_memory()
            
            # 히스토리 다시 로드
            self._load_chat_history()
            
            self.logger.log_step("메모리 윈도우 크기 변경", f"새 크기: {new_k}")
    
    def get_conversation_summary(self) -> Dict:
        """
        현재 대화 요약 정보 반환
        
        Returns:
            Dict: 대화 요약 정보
        """
        try:
            full_history = self.get_full_conversation_history()
            memory_history = self.get_chat_history_as_dicts()
            
            return {
                "session_id": self.session_id,
                "total_messages": len(full_history),
                "memory_messages": len(memory_history),
                "memory_window_size": self.memory_k,
                "auto_save": self.auto_save
            }
        except Exception as e:
            self.logger.log_error("get_conversation_summary", e)
            return {}
    
    def export_conversation(self, format: str = "json") -> str:
        """
        대화 내용 내보내기
        
        Args:
            format (str): 내보내기 형식 ("json", "text")
            
        Returns:
            str: 내보내진 대화 내용
        """
        try:
            full_history = self.get_full_conversation_history()
            
            if format == "json":
                import json
                return json.dumps(full_history, ensure_ascii=False, indent=2)
            
            elif format == "text":
                text_lines = []
                for msg in full_history:
                    role = "사용자" if msg["role"] == "user" else "AI"
                    timestamp = msg["timestamp"]
                    content = msg["content"]
                    text_lines.append(f"[{timestamp}] {role}: {content}")
                
                return "\n\n".join(text_lines)
            
            else:
                raise ValueError(f"지원하지 않는 형식: {format}")
                
        except Exception as e:
            self.logger.log_error("export_conversation", e)
            return ""
    
    def delete_conversation(self) -> bool:
        """
        현재 세션의 모든 대화 삭제
        
        Returns:
            bool: 삭제 성공 여부
        """
        try:
            # 메모리 초기화
            self.clear_memory()
            
            # SQLite에서 삭제
            success = self.sql_manager.delete_conversation(self.session_id)
            
            if success:
                self.logger.log_success("대화 삭제 완료")
            else:
                self.logger.log_warning_with_icon("대화 삭제 실패")
            
            return success
            
        except Exception as e:
            self.logger.log_error("delete_conversation", e)
            return False
    
    def switch_session(self, new_session_id: str):
        """
        다른 세션으로 전환
        
        Args:
            new_session_id (str): 새 세션 ID
        """
        try:
            self.session_id = new_session_id
            
            # 메모리 초기화 후 새 히스토리 로드
            self.clear_memory()
            self._load_chat_history()
            
            self.logger.log_step("세션 전환", f"새 세션 ID: {new_session_id}")
            
        except Exception as e:
            self.logger.log_error("switch_session", e)