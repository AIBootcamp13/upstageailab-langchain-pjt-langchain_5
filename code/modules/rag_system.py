"""
RAG 시스템 공통 모듈

이 모듈은 main.py, cli.py, evaluate.py에서 공통으로 사용하는
RAG 시스템 초기화 및 질의 처리 로직을 제공합니다.

주요 기능:
1. RAG 시스템 표준 초기화
2. 질의 처리 공통 로직
3. 프로젝트 경로 관리
4. 에러 처리 통합
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Any
from langchain_upstage import UpstageEmbeddings

from .vector_store import VectorStoreManager
from .llm import LLMManager
from .retriever import RetrieverManager
from .sql import SQLManager
from .chat_history import ChatHistoryManager
from .logger import LoggerManager
from .config_loader import get_config_loader
from .router import QueryRouter


class RAGSystemInitializer:
    """RAG 시스템 공통 초기화 클래스"""
    
    @staticmethod
    def get_project_paths(current_file_path: Path) -> Tuple[str, str, str]:
        """
        현재 파일 경로를 기준으로 프로젝트 경로들을 계산
        
        Args:
            current_file_path (Path): 현재 실행 중인 파일의 경로
            
        Returns:
            Tuple[str, str, str]: (project_root, pdf_dir, vectorstore_dir)
        """
        # code 폴더 내부에서 실행되는 경우를 고려
        if current_file_path.name == "code":
            project_root = current_file_path.parent
        else:
            project_root = current_file_path.parent
            
        pdf_dir = str(project_root / "data" / "pdf")
        vectorstore_dir = str(project_root / "data" / "vectorstore")
        
        return str(project_root), pdf_dir, vectorstore_dir
    
    @staticmethod
    def initialize_embeddings(use_config: bool = True):
        """
        임베딩 모델 초기화 (config 기반)
        
        Args:
            use_config (bool): config.yaml 사용 여부
        """
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        
        logger = LoggerManager("RAGSystem")
        
        if use_config:
            try:
                config_loader = get_config_loader()
                embeddings_config = config_loader.get_embeddings_config()
                provider = embeddings_config.get("provider", "upstage")
                model_name = embeddings_config.get("model")
            except Exception as e:
                logger.log_warning("임베딩 설정 로드 실패, Upstage 기본값 사용", str(e))
                provider = "upstage"
                model_name = "embedding-query"
        else:
            provider = "upstage"
            model_name = "embedding-query"

        logger.log_step("임베딩 모델 초기화", f"Provider: {provider}, Model: {model_name}")

        if provider == "google":
            if not os.getenv("GOOGLE_API_KEY"):
                raise ValueError("임베딩에 google provider를 사용하려면 GOOGLE_API_KEY가 필요합니다.")
            return GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
        elif provider == "upstage":
            if not os.getenv("UPSTAGE_API_KEY"):
                raise ValueError("임베딩에 upstage provider를 사용하려면 UPSTAGE_API_KEY가 필요합니다.")
            return UpstageEmbeddings(
                api_key=os.getenv("UPSTAGE_API_KEY"),
                model=model_name
            )
        else:
            raise ValueError(f"지원하지 않는 임베딩 provider입니다: {provider}")
    
    @staticmethod
    def initialize_vector_manager(pdf_dir: str, vectorstore_dir: str, 
                                embeddings: UpstageEmbeddings,
                                use_config: bool = True,
                                chunk_size: int = None,
                                chunk_overlap: int = None) -> VectorStoreManager:
        """
        벡터스토어 관리자 초기화 (config 기반)
        
        Args:
            pdf_dir (str): PDF 디렉토리 경로
            vectorstore_dir (str): 벡터스토어 디렉토리 경로
            embeddings (UpstageEmbeddings): 임베딩 모델
            use_config (bool): config.yaml 사용 여부
            chunk_size (int, optional): 청크 크기 (config 우선)
            chunk_overlap (int, optional): 청크 겹침 (config 우선)
        """
        if use_config:
            try:
                config_loader = get_config_loader()
                vectorstore_config = config_loader.get_vectorstore_config()
                
                chunk_size = vectorstore_config.get("chunk_size", chunk_size or 1000)
                chunk_overlap = vectorstore_config.get("chunk_overlap", chunk_overlap or 50)
                include_filename_in_chunk = vectorstore_config.get("include_filename_in_chunk", True)
            except Exception:
                chunk_size = chunk_size or 1000
                chunk_overlap = chunk_overlap or 50
                include_filename_in_chunk = True
        else:
            chunk_size = chunk_size or 1000
            chunk_overlap = chunk_overlap or 50
            include_filename_in_chunk = True
        
        return VectorStoreManager(
            pdf_dir=pdf_dir,
            vectorstore_dir=vectorstore_dir, 
            embeddings=embeddings,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            include_filename_in_chunk=include_filename_in_chunk
        )
    
    @classmethod
    def initialize_system(cls, 
                         current_file_path: Path,
                         include_sql: bool = False,
                         use_config: bool = True,
                         chunk_size: int = None,
                         chunk_overlap: int = None,
                         logger_name: str = "RAGSystem",
                         enable_db_memory: bool = False) -> Tuple:
        """
        표준 RAG 시스템 초기화 (config 기반)
        
        Args:
            current_file_path (Path): 현재 실행 파일 경로
            include_sql (bool): SQL 관리자 포함 여부
            use_config (bool): config.yaml 사용 여부 (기본: True)
            chunk_size (int, optional): 청크 크기 (config 우선)
            chunk_overlap (int, optional): 청크 겹침 (config 우선)
            logger_name (str): 로거 이름
            enable_db_memory (bool): RAGQueryProcessor에 DB 메모리 기능 활성화 여부
            
        Returns:
            Tuple: (vector_manager, llm_manager, retriever_manager[, sql_manager])
        """
        logger = LoggerManager(logger_name)
        logger.log_function_start("initialize_system")
        
        try:
            # 1. 프로젝트 경로 계산
            project_root, pdf_dir, vectorstore_dir = cls.get_project_paths(current_file_path)
            logger.log_step("프로젝트 경로 설정", f"root: {project_root}")
            
            # 2. 임베딩 모델 초기화 (config 기반)
            embeddings = cls.initialize_embeddings(use_config=use_config)
            logger.log_step("임베딩 모델 초기화 완료")
            
            # 3. 벡터스토어 관리자 초기화 (config 기반)
            vector_manager = cls.initialize_vector_manager(
                pdf_dir, vectorstore_dir, embeddings, 
                use_config=use_config, 
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
            )
            logger.log_step("벡터스토어 관리자 초기화 완료")
            
            # 4. 벡터스토어 로드/생성
            vectorstore = vector_manager.get_or_create_vectorstore()
            if vectorstore is None:
                logger.log_error_with_icon("벡터스토어를 생성하거나 로드할 수 없습니다.")
                return None
            
            logger.log_step("벡터스토어 로드/생성 완료")
            
            # 5. LLM 관리자 초기화 (config 기반)
            llm_manager = LLMManager(use_config=use_config)
            logger.log_step("LLM 관리자 초기화 완료")
            
            # 6. 검색기 관리자 초기화 (config 기반)
            retriever_manager = RetrieverManager(vectorstore=vectorstore, use_config=use_config)
            logger.log_step("검색기 관리자 초기화 완료")
            
            # 7. RAG 질의 처리기 초기화
            # 라우터 초기화
            router = None
            try:
                router = QueryRouter(use_config=use_config)
                logger.log_step("라우터 초기화", "QueryRouter 생성 완료")
            except Exception as e:
                logger.log_warning("라우터 초기화 실패", str(e))
            
            # RAGQueryProcessor 생성 (라우터 포함)
            query_processor = RAGQueryProcessor(
                llm_manager=llm_manager, 
                retriever_manager=retriever_manager, 
                router=router,  # 라우터 주입
                logger_name=logger_name + "_Query",
                db_save=enable_db_memory,
                project_root=str(project_root) if enable_db_memory else None
            )
            logger.log_step("RAG 질의 처리기 초기화 완료")
            
            # 8. SQL 관리자 초기화 (선택사항)
            sql_manager = None
            if include_sql:
                db_path = str(Path(project_root) / "data" / "chat.db")
                sql_manager = SQLManager(db_path=db_path)
                logger.log_step("SQL 관리자 초기화 완료")
            
            logger.log_function_end("initialize_system", "모든 컴포넌트 초기화 완료")
            
            if include_sql:
                return vector_manager, llm_manager, retriever_manager, sql_manager, query_processor
            else:
                return vector_manager, llm_manager, retriever_manager, query_processor
            
        except Exception as e:
            logger.log_error("initialize_system", e)
            return None


class RAGQueryProcessor:
    """통합 질의 처리 클래스 (라우팅 + RAG/일반 + 메모리)"""
    
    def __init__(self, llm_manager: LLMManager, retriever_manager: RetrieverManager,
                 router: QueryRouter = None,  # 라우터 추가
                 logger_name: str = "RAGQueryProcessor", 
                 db_save: bool = False,
                 project_root: str = None):
        """
        RAGQueryProcessor 초기화
        
        Args:
            llm_manager: LLM 관리자
            retriever_manager: 검색기 관리자
            router (QueryRouter, optional): 질문 라우터. None이면 기본적으로 RAG 사용
            logger_name: 로거 이름
            db_save: 데이터베이스 저장 여부
            project_root: 프로젝트 루트 경로 (db_save=True일 때 필요)
        """
        self.llm_manager = llm_manager
        self.retriever_manager = retriever_manager
        self.router = router  # 라우터 추가
        self.logger = LoggerManager(logger_name)
        self.db_save = db_save
        
        # 메모리 관리자 초기화
        self._init_memory_manager(project_root)
    
    def unified_query(self, question: str, chat_history_manager: ChatHistoryManager = None,
                     auto_save: bool = True, return_sources: bool = False) -> Dict[str, Any]:
        """
        라우팅 + 분기 처리 + 메모리 저장을 통합한 질의 처리
        
        Args:
            question (str): 사용자 질문
            chat_history_manager (ChatHistoryManager, optional): 채팅 히스토리 관리자
            auto_save (bool): 메모리에 자동 저장 여부
            return_sources (bool): 소스 정보 반환 여부
            
        Returns:
            Dict[str, Any]: {
                "response": str,
                "success": bool,
                "error": Optional[str],
                "processing_type": str ("RAG" or "GENERAL"),
                "sources": List[str] (if return_sources=True),
                "documents": List[Document] (if return_sources=True),
                "routing_info": Dict (라우팅 결과 정보)
            }
        """
        self.logger.log_function_start("unified_query", 
                                     question=question[:50] + "..." if len(question) > 50 else question)
        
        try:
            # 1. 라우팅 (라우터가 있는 경우)
            if self.router:
                routing_result = self.router.route(question)
                use_rag = routing_result["use_rag"]
                self.logger.log_step("라우팅 완료", f"타입: {routing_result['type']}")
            else:
                # 라우터가 없으면 기본적으로 RAG 사용
                use_rag = True
                routing_result = {"type": "RAG", "use_rag": True, "confidence": 1.0}
                self.logger.log_step("라우터 없음", "기본 RAG 모드 사용")
            
            # 2. 분기 처리 및 응답 생성
            if use_rag:
                result = self._process_rag_query(question, chat_history_manager, return_sources)
            else:
                result = self._process_general_query(question, chat_history_manager)
            
            # 3. 메모리에 저장 (선택사항)
            if auto_save and chat_history_manager:
                self._save_to_memory(question, result, chat_history_manager)
            
            # 4. 라우팅 정보 추가
            result["routing_info"] = routing_result
            
            self.logger.log_function_end("unified_query", f"처리 완료: {result['processing_type']}")
            return result
            
        except Exception as e:
            self.logger.log_error("unified_query", e)
            return {
                "response": f"질의 처리 중 오류가 발생했습니다: {str(e)}",
                "success": False,
                "error": str(e),
                "processing_type": "ERROR",
                "routing_info": {"type": "ERROR", "use_rag": False, "confidence": 0.0}
            }
    
    def _process_rag_query(self, question: str, chat_history_manager: ChatHistoryManager = None,
                          return_sources: bool = False) -> Dict[str, Any]:
        """RAG 질의 처리 (기존 process_query와 동일)"""
        return self.process_query(question, chat_history_manager, return_sources)
    
    def _process_general_query(self, question: str, chat_history_manager: ChatHistoryManager = None) -> Dict[str, Any]:
        """일반 질의 처리"""
        self.logger.log_function_start("_process_general_query", 
                                     question=question[:50] + "..." if len(question) > 50 else question)
        
        try:
            # 채팅 히스토리 가져오기
            chat_history = []
            if chat_history_manager:
                chat_history = chat_history_manager.get_chat_history_as_dicts()
                self.logger.log_step("채팅 히스토리 로드", f"{len(chat_history)}개 메시지")
            
            # 일반답변용 LLM 응답 생성
            response = self.llm_manager.generate_general_response(
                question=question,
                chat_history=chat_history
            )
            
            self.logger.log_step("일반답변 생성 완료")
            
            result = {
                "response": response,
                "success": True,
                "error": None,
                "processing_type": "GENERAL",
                "sources": [],
                "documents": []
            }
            
            self.logger.log_function_end("_process_general_query", "일반답변 처리 완료")
            return result
            
        except Exception as e:
            self.logger.log_error("_process_general_query", e)
            return {
                "response": f"일반답변 생성 중 오류가 발생했습니다: {str(e)}",
                "success": False,
                "error": str(e),
                "processing_type": "ERROR"
            }
    
    def _save_to_memory(self, question: str, result: Dict[str, Any], 
                        chat_history_manager: ChatHistoryManager):
        """메모리에 대화 저장"""
        try:
            if result["success"]:
                sources = result.get("sources", [])
                chat_history_manager.add_user_message(question)
                chat_history_manager.add_ai_message(result["response"], question, sources)
                self.logger.log_step("대화 기록 저장 완료")
        except Exception as e:
            self.logger.log_warning("대화 기록 저장 실패", str(e))
    
    def _init_memory_manager(self, project_root: str = None):
        """메모리 관리자 초기화"""
        if self.db_save and project_root:
            # 데이터베이스 기반 메모리 (WebUI용)
            from .sql import SQLManager
            db_path = str(Path(project_root) / "data" / "chat.db")
            sql_manager = SQLManager(db_path=db_path)
            # 설정에서 메모리 윈도우 크기를 불러와 주입
            try:
                config_loader = get_config_loader()
                db_conf = config_loader.get_database_config()
                memory_k = db_conf.get("memory_window", 3)
            except Exception:
                memory_k = 3

            self.chat_history = ChatHistoryManager(
                sql_manager=sql_manager,
                auto_save=True,
                memory_k=memory_k
            )
        else:
            # CLI/평가용: 메모리 기반만 사용 (세션별 관리 없음)
            # 설정에서 메모리 윈도우 크기를 불러와 주입
            try:
                config_loader = get_config_loader()
                db_conf = config_loader.get_database_config()
                memory_k = db_conf.get("memory_window", 3)
            except Exception:
                memory_k = 3

            self.chat_history = ChatHistoryManager(
                sql_manager=None,
                auto_save=False,
                memory_k=memory_k
            )
    
    def query(self, question: str, return_sources: bool = False) -> Dict[str, Any]:
        """
        자동 메모리 기능이 포함된 간단한 질의 처리 (기존 호환성 유지)
        
        Args:
            question (str): 사용자 질문
            return_sources (bool): 소스 정보 반환 여부
            
        Returns:
            Dict[str, Any]: process_query와 동일한 형식
        """
        return self.process_query_with_memory(
            question=question,
            chat_history_manager=self.chat_history,
            auto_save=True,
            return_sources=return_sources
        )
    
    def process_query(self, 
                     question: str,
                     chat_history_manager: Optional[ChatHistoryManager] = None,
                     return_sources: bool = False) -> Dict[str, Any]:
        """
        표준 RAG 질의 처리
        
        Args:
            question (str): 사용자 질문
            chat_history_manager (Optional[ChatHistoryManager]): 채팅 히스토리 관리자
            return_sources (bool): 소스 정보 반환 여부
            
        Returns:
            Dict[str, Any]: {
                "response": str,
                "sources": List[str] (if return_sources=True),
                "documents": List[Document] (if return_sources=True),
                "success": bool,
                "error": Optional[str]
            }
        """
        self.logger.log_function_start("process_query", 
                                     question=question[:50] + "..." if len(question) > 50 else question)
        
        try:
            # 1. 문서 검색
            documents = self.retriever_manager.search_documents(question)
            context = self.retriever_manager.format_documents_for_context(documents)
            
            self.logger.log_step("문서 검색 완료", f"{len(documents)}개 문서 찾음")
            
            # 2. 채팅 히스토리 가져오기
            chat_history = []
            if chat_history_manager:
                chat_history = chat_history_manager.get_chat_history_as_dicts()
                self.logger.log_step("채팅 히스토리 로드", f"{len(chat_history)}개 메시지")
            
            # 3. LLM 응답 생성
            response = self.llm_manager.generate_response(
                question=question,
                context=context,
                chat_history=chat_history
            )
            
            self.logger.log_step("LLM 응답 생성 완료")
            
            # 4. 결과 구성
            result = {
                "response": response,
                "success": True,
                "error": None,
                "processing_type": "RAG"  # RAG 처리 타입 추가
            }
            
            # 5. 소스 정보 추가 (선택사항)
            if return_sources:
                sources = self.retriever_manager.get_unique_sources(documents)
                result["sources"] = sources
                result["documents"] = documents
                self.logger.log_step("소스 정보 추가", f"{len(sources)}개 소스")
            
            self.logger.log_function_end("process_query", "질의 처리 완료")
            return result
            
        except Exception as e:
            self.logger.log_error("process_query", e)
            return {
                "response": f"질의 처리 중 오류가 발생했습니다: {str(e)}",
                "success": False,
                "error": str(e),
                "processing_type": "ERROR"  # 에러 처리 타입 추가
            }
    
    def process_query_with_memory(self,
                                question: str,
                                chat_history_manager: ChatHistoryManager,
                                auto_save: bool = True,
                                return_sources: bool = False) -> Dict[str, Any]:
        """
        메모리 기능을 포함한 질의 처리
        
        Args:
            question (str): 사용자 질문
            chat_history_manager (ChatHistoryManager): 채팅 히스토리 관리자
            auto_save (bool): 자동 저장 여부
            return_sources (bool): 소스 정보 반환 여부
            
        Returns:
            Dict[str, Any]: process_query와 동일한 형식
        """
        # 질의 처리
        result = self.process_query(question, chat_history_manager, return_sources)
        
        # 성공적으로 처리된 경우 메모리에 추가
        if result["success"] and auto_save:
            try:
                # 소스 정보가 있는 경우 함께 저장
                sources = result.get("sources", [])
                
                if hasattr(chat_history_manager, 'add_ai_message') and sources:
                    # main.py 스타일 (소스 포함)
                    chat_history_manager.add_user_message(question)
                    chat_history_manager.add_ai_message(result["response"], question, sources)
                elif hasattr(chat_history_manager, 'add_conversation_pair'):
                    # evaluate.py 스타일
                    chat_history_manager.add_conversation_pair(question, result["response"])
                else:
                    # 기본 방식
                    chat_history_manager.add_user_message(question)
                    chat_history_manager.add_ai_message(result["response"])
                
                self.logger.log_step("대화 기록 저장 완료")
                
            except Exception as e:
                self.logger.log_warning("대화 기록 저장 실패", str(e))
        
        return result


