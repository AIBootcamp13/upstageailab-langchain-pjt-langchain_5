"""
벡터 DB 검색 기능 모듈

이 모듈은 FAISS 벡터 데이터베이스에서 사용자 질문과 관련된 문서를 검색하는 기능을 제공합니다.

주요 기능:
1. 벡터 데이터베이스 검색
2. 검색 결과 필터링 및 랭킹
3. 검색 매개변수 조정
4. 검색 결과 후처리
"""

from typing import List, Dict, Optional, Tuple, Any
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_upstage import UpstageEmbeddings

from .logger import LoggerManager
from .config_loader import get_config_loader


class RetrieverManager:
    """벡터 DB 검색 관리 클래스"""
    
    def __init__(self, 
                 vectorstore: FAISS = None,
                 use_config: bool = True,
                 search_type: str = None,
                 k: int = None,
                 score_threshold: float = None):
        """
        RetrieverManager 초기화
        
        Args:
            vectorstore (FAISS, optional): FAISS 벡터스토어
            use_config (bool): config.yaml 사용 여부 (기본: True)
            search_type (str, optional): 검색 타입 (config 우선)
            k (int, optional): 반환할 문서 수 (config 우선)
            score_threshold (float, optional): 유사도 임계값 (config 우선)
        """
        self.logger = LoggerManager("Retriever")
        self.vectorstore = vectorstore
        
        # 설정 로드
        if use_config:
            try:
                config_loader = get_config_loader()
                retriever_config = config_loader.get_retriever_config()
                
                self.search_type = retriever_config.get("search_type", search_type or "similarity")
                self.k = retriever_config.get("k", k or 5)
                self.score_threshold = retriever_config.get("score_threshold", score_threshold or 0.8)
                
                self.logger.log_step("Config 기반 Retriever 설정", 
                                   f"타입: {self.search_type}, k: {self.k}, threshold: {self.score_threshold}")
            except Exception as e:
                self.logger.log_warning(f"Config 로드 실패, 기본값 사용", str(e))
                self.search_type = search_type or "similarity"
                self.k = k or 5
                self.score_threshold = score_threshold or 0.8
        else:
            self.search_type = search_type or "similarity"
            self.k = k or 5
            self.score_threshold = score_threshold or 0.8
        self.retriever = None
        
        # 검색기 초기화
        if vectorstore:
            self._init_retriever()
        
        self.logger.log_success("Retriever Manager 초기화 완료")
    
    def _init_retriever(self):
        """검색기 초기화"""
        try:
            if self.search_type == "similarity_score_threshold" and self.score_threshold:
                self.retriever = self.vectorstore.as_retriever(
                    search_type=self.search_type,
                    search_kwargs={
                        "k": self.k,
                        "score_threshold": self.score_threshold
                    }
                )
            elif self.search_type == "mmr":
                self.retriever = self.vectorstore.as_retriever(
                    search_type=self.search_type,
                    search_kwargs={
                        "k": self.k,
                        "fetch_k": self.k * 2,  # MMR을 위해 더 많은 문서 가져오기
                        "lambda_mult": 0.7  # 다양성 조절 (0.0~1.0)
                    }
                )
            else:  # similarity (기본값)
                self.retriever = self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": self.k}
                )
            
            self.logger.log_step("검색기 초기화", 
                               f"타입: {self.search_type}, k: {self.k}")
        except Exception as e:
            self.logger.log_error("검색기 초기화", e)
            raise
    
    def set_vectorstore(self, vectorstore: FAISS):
        """벡터스토어 설정"""
        self.vectorstore = vectorstore
        self._init_retriever()
        self.logger.log_step("벡터스토어 설정", "새 벡터스토어로 업데이트")
    
    def search_documents(self, query: str) -> List[Document]:
        """
        문서 검색
        
        Args:
            query (str): 검색 쿼리
            
        Returns:
            List[Document]: 검색된 문서 리스트
        """
        if not self.retriever:
            self.logger.log_warning_with_icon("검색기가 초기화되지 않았습니다.")
            return []
        
        self.logger.log_function_start("search_documents", 
                                     query=query[:50] + "..." if len(query) > 50 else query)
        
        try:
            documents = self.retriever.invoke(query)
            self.logger.log_function_end("search_documents", 
                                       f"{len(documents)}개 문서 검색")
            return documents
        except Exception as e:
            self.logger.log_error("search_documents", e)
            return []
    
    def search_with_scores(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """
        유사도 점수와 함께 문서 검색
        
        Args:
            query (str): 검색 쿼리
            k (int, optional): 반환할 문서 수 (None이면 기본값 사용)
            
        Returns:
            List[Tuple[Document, float]]: (문서, 유사도 점수) 튜플 리스트
        """
        if not self.vectorstore:
            self.logger.log_warning_with_icon("벡터스토어가 설정되지 않았습니다.")
            return []
        
        search_k = k or self.k
        self.logger.log_function_start("search_with_scores", 
                                     query=query[:50] + "..." if len(query) > 50 else query,
                                     k=search_k)
        
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=search_k)
            self.logger.log_function_end("search_with_scores", 
                                       f"{len(results)}개 문서 검색 (점수 포함)")
            return results
        except Exception as e:
            self.logger.log_error("search_with_scores", e)
            return []
    
    def search_by_vector(self, embedding: List[float], k: int = None) -> List[Document]:
        """
        임베딩 벡터로 직접 검색
        
        Args:
            embedding (List[float]): 검색할 임베딩 벡터
            k (int, optional): 반환할 문서 수
            
        Returns:
            List[Document]: 검색된 문서 리스트
        """
        if not self.vectorstore:
            self.logger.log_warning_with_icon("벡터스토어가 설정되지 않았습니다.")
            return []
        
        search_k = k or self.k
        self.logger.log_function_start("search_by_vector", k=search_k)
        
        try:
            documents = self.vectorstore.similarity_search_by_vector(embedding, k=search_k)
            self.logger.log_function_end("search_by_vector", 
                                       f"{len(documents)}개 문서 검색")
            return documents
        except Exception as e:
            self.logger.log_error("search_by_vector", e)
            return []
    
    def filter_documents_by_source(self, documents: List[Document], 
                                 source_filter: str) -> List[Document]:
        """
        소스 파일로 문서 필터링
        
        Args:
            documents (List[Document]): 원본 문서 리스트
            source_filter (str): 필터링할 소스 파일명 (부분 일치)
            
        Returns:
            List[Document]: 필터링된 문서 리스트
        """
        filtered_docs = []
        for doc in documents:
            source = doc.metadata.get('source', '')
            source_file = doc.metadata.get('source_file', '')
            
            if source_filter.lower() in source.lower() or source_filter.lower() in source_file.lower():
                filtered_docs.append(doc)
        
        self.logger.log_step("문서 필터링", 
                           f"'{source_filter}' 기준으로 {len(documents)} → {len(filtered_docs)}개")
        return filtered_docs
    
    def get_unique_sources(self, documents: List[Document]) -> List[str]:
        """
        문서 리스트에서 고유한 소스 파일 목록 추출
        
        Args:
            documents (List[Document]): 문서 리스트
            
        Returns:
            List[str]: 고유한 소스 파일 목록 (파일명만)
        """
        import os
        
        sources = set()
        for doc in documents:
            # source_file 우선 사용, 없으면 source에서 파일명 추출
            source_file = doc.metadata.get('source_file', '')
            if source_file:
                sources.add(source_file)
            else:
                source = doc.metadata.get('source', '')
                if source:
                    sources.add(os.path.basename(source))
        
        return sorted(list(sources))
    
    def format_documents_for_context(self, documents: List[Document]) -> str:
        """
        문서 리스트를 컨텍스트 문자열로 포맷팅
        
        Args:
            documents (List[Document]): 문서 리스트
            
        Returns:
            str: 포맷팅된 컨텍스트 문자열
        """
        if not documents:
            return ""
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source_file', doc.metadata.get('source', 'Unknown'))
            page = doc.metadata.get('page', '')
            
            header = f"[문서 {i}]"
            if source != 'Unknown':
                header += f" 출처: {source}"
            if page:
                header += f" (페이지: {page})"
            
            context_parts.append(f"{header}\n{doc.page_content}\n")
        
        return "\n".join(context_parts)
    
    def update_search_params(self, 
                           search_type: str = None,
                           k: int = None,
                           score_threshold: float = None):
        """
        검색 매개변수 업데이트
        
        Args:
            search_type (str, optional): 새 검색 타입
            k (int, optional): 새 k 값
            score_threshold (float, optional): 새 점수 임계값
        """
        updated = False
        
        if search_type and search_type != self.search_type:
            self.search_type = search_type
            updated = True
            self.logger.log_step("검색 타입 변경", f"새 타입: {search_type}")
        
        if k and k != self.k:
            self.k = k
            updated = True
            self.logger.log_step("검색 문서 수 변경", f"새 k: {k}")
        
        if score_threshold is not None and score_threshold != self.score_threshold:
            self.score_threshold = score_threshold
            updated = True
            self.logger.log_step("점수 임계값 변경", f"새 임계값: {score_threshold}")
        
        # 매개변수가 변경되었으면 검색기 재초기화
        if updated and self.vectorstore:
            self._init_retriever()
    
    def get_search_info(self) -> Dict[str, Any]:
        """현재 검색 설정 정보 반환"""
        return {
            "search_type": self.search_type,
            "k": self.k,
            "score_threshold": self.score_threshold,
            "vectorstore_available": self.vectorstore is not None,
            "retriever_available": hasattr(self, 'retriever') and self.retriever is not None
        }
    
    def test_search(self, query: str = "테스트") -> bool:
        """
        검색 기능 테스트
        
        Args:
            query (str): 테스트 쿼리
            
        Returns:
            bool: 테스트 성공 여부
        """
        try:
            documents = self.search_documents(query)
            self.logger.log_success(f"검색 테스트 완료 - {len(documents)}개 문서 반환")
            return True
        except Exception as e:
            self.logger.log_error("검색 테스트", e)
            return False