"""
벡터 데이터베이스 관리 모듈

이 모듈은 FAISS 벡터스토어의 생성, 저장, 로딩 및 증분 업데이트 기능을 제공합니다.

주요 기능:
1. 벡터스토어 영구 저장/로딩: FAISS 인덱스를 디스크에 저장하고 재시작 시 로드
2. 파일 변경 감지: PDF 파일의 수정시간과 해시값을 추적하여 변경사항 자동 감지  
3. 증분 업데이트: 새로운/수정된 파일만 처리하여 벡터스토어를 효율적으로 업데이트
4. 삭제 파일 처리: 삭제된 파일의 벡터 데이터를 제거하기 위한 전체 재구성 기능
5. 메타데이터 관리: 파일별 처리 이력을 JSON으로 관리

사용법:
    from vector_store import VectorStoreManager
    from langchain_upstage import UpstageEmbeddings
    
    embeddings = UpstageEmbeddings(api_key="your_key", model="embedding-query")
    # 삭제 시 즉시 재구성 (기본값)
    manager = VectorStoreManager(embeddings=embeddings, rebuild_on_delete=True)
    # 또는 삭제 시 재구성하지 않음
    # manager = VectorStoreManager(embeddings=embeddings, rebuild_on_delete=False)
    vectorstore = manager.get_or_create_vectorstore()
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_upstage import UpstageEmbeddings
from .logger import LoggerManager

log = LoggerManager("VectorStore")


class VectorStoreManager:
    def __init__(self, 
                 pdf_dir: str = "../data/pdf",
                 vectorstore_dir: str = "../data/vectorstore",
                 embeddings: Optional[UpstageEmbeddings] = None,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 50,
                 rebuild_on_delete: bool = True,
                 delete_threshold: int = 1):
        self.pdf_dir = pdf_dir
        self.vectorstore_dir = vectorstore_dir
        self.embeddings = embeddings
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 삭제 파일 처리 옵션
        self.rebuild_on_delete = rebuild_on_delete  # 삭제 시 즉시 재구성 여부
        self.delete_threshold = delete_threshold    # 재구성을 위한 삭제 파일 임계값
        
        # 디렉토리 생성
        os.makedirs(self.vectorstore_dir, exist_ok=True)
        
        # 메타데이터 파일 경로
        self.metadata_file = os.path.join(self.vectorstore_dir, "file_metadata.json")
        
        # 텍스트 분할기 초기화
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
    
    def _get_file_metadata(self) -> Dict[str, Dict]:
        """저장된 파일 메타데이터를 로드합니다."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                log.warning("메타데이터 파일을 읽는 중 오류 발생. 새로 생성합니다.")
                return {}
        return {}
    
    def _save_file_metadata(self, metadata: Dict[str, Dict]):
        """파일 메타데이터를 저장합니다."""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    def _get_file_hash(self, file_path: str) -> str:
        """파일의 해시값을 계산합니다."""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _scan_pdf_files(self) -> Dict[str, Dict]:
        """PDF 디렉토리의 모든 파일을 스캔하고 메타데이터를 수집합니다."""
        current_files = {}
        pdf_path = Path(self.pdf_dir)
        
        if not pdf_path.exists():
            log.warning(f"PDF 디렉토리가 존재하지 않습니다: {self.pdf_dir}")
            return current_files
        
        for pdf_file in pdf_path.rglob("*.pdf"):
            file_path = str(pdf_file)
            relative_path = os.path.relpath(file_path, self.pdf_dir)
            
            stat = pdf_file.stat()
            current_files[relative_path] = {
                "absolute_path": file_path,
                "size": stat.st_size,
                "modified_time": stat.st_mtime,
                "hash": self._get_file_hash(file_path),
                "last_processed": None
            }
        
        return current_files
    
    def check_file_changes(self) -> Tuple[List[str], List[str], List[str]]:
        """파일 변경사항을 확인하고 새로운/수정된/삭제된 파일 목록을 반환합니다."""
        stored_metadata = self._get_file_metadata()
        current_files = self._scan_pdf_files()
        
        new_files = []
        modified_files = []
        deleted_files = []
        
        # 새로운 파일과 수정된 파일 확인
        for file_path, file_info in current_files.items():
            if file_path not in stored_metadata:
                new_files.append(file_path)
                log.info(f"새로운 파일 발견: {file_path}")
            else:
                stored_info = stored_metadata[file_path]
                if (file_info["hash"] != stored_info.get("hash") or 
                    file_info["modified_time"] != stored_info.get("modified_time")):
                    modified_files.append(file_path)
                    log.info(f"수정된 파일 발견: {file_path}")
        
        # 삭제된 파일 확인
        for file_path in stored_metadata:
            if file_path not in current_files:
                deleted_files.append(file_path)
                log.info(f"삭제된 파일 발견: {file_path}")
        
        return new_files, modified_files, deleted_files
    
    def _load_and_split_documents(self, file_paths: List[str]) -> List:
        """지정된 파일들을 로드하고 분할합니다."""
        all_documents = []
        
        for file_path in file_paths:
            absolute_path = os.path.join(self.pdf_dir, file_path)
            if not os.path.exists(absolute_path):
                log.warning(f"파일을 찾을 수 없습니다: {absolute_path}")
                continue
            
            try:
                loader = PyMuPDFLoader(absolute_path)
                docs = loader.load()
                
                # 문서에 파일 경로 메타데이터 추가
                for doc in docs:
                    doc.metadata["source_file"] = file_path
                
                split_docs = self.text_splitter.split_documents(docs)
                all_documents.extend(split_docs)
                
                log.info(f"파일 처리 완료: {file_path} ({len(split_docs)} 청크)")
                
            except Exception as e:
                log.error(f"파일 로드 중 오류 발생 ({file_path}): {str(e)}")
        
        return all_documents
    
    def vectorstore_exists(self) -> bool:
        """벡터스토어가 이미 존재하는지 확인합니다."""
        index_path = os.path.join(self.vectorstore_dir, "index.faiss")
        pkl_path = os.path.join(self.vectorstore_dir, "index.pkl")
        return os.path.exists(index_path) and os.path.exists(pkl_path)
    
    def save_vectorstore(self, vectorstore: FAISS):
        """벡터스토어를 디스크에 저장합니다."""
        try:
            vectorstore.save_local(self.vectorstore_dir)
            log.info(f"벡터스토어가 저장되었습니다: {self.vectorstore_dir}")
        except Exception as e:
            log.error(f"벡터스토어 저장 중 오류 발생: {str(e)}")
            raise
    
    def load_vectorstore(self) -> Optional[FAISS]:
        """저장된 벡터스토어를 로드합니다."""
        if not self.vectorstore_exists():
            log.warning("저장된 벡터스토어가 없습니다.")
            return None
        
        try:
            vectorstore = FAISS.load_local(
                self.vectorstore_dir, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            log.info("벡터스토어가 성공적으로 로드되었습니다.")
            return vectorstore
        except Exception as e:
            log.error(f"벡터스토어 로드 중 오류 발생: {str(e)}")
            return None
    
    def create_vectorstore_from_files(self, file_paths: List[str]) -> Optional[FAISS]:
        """지정된 파일들로부터 새로운 벡터스토어를 생성합니다."""
        if not file_paths:
            log.warning("처리할 파일이 없습니다.")
            return None
        
        documents = self._load_and_split_documents(file_paths)
        if not documents:
            log.warning("처리된 문서가 없습니다.")
            return None
        
        try:
            vectorstore = FAISS.from_documents(documents, self.embeddings)
            log.info(f"벡터스토어가 생성되었습니다. ({len(documents)} 청크)")
            return vectorstore
        except Exception as e:
            log.error(f"벡터스토어 생성 중 오류 발생: {str(e)}")
            return None
    
    def _rebuild_vectorstore_from_existing_files(self) -> Optional[FAISS]:
        """현재 존재하는 파일들로부터 벡터스토어를 재구성합니다."""
        current_files = self._scan_pdf_files()
        all_files = list(current_files.keys())
        
        if not all_files:
            log.warning("벡터스토어 재구성을 위한 파일이 없습니다.")
            return None
        
        log.info(f"벡터스토어를 재구성합니다. ({len(all_files)} 파일)")
        return self.create_vectorstore_from_files(all_files)
    
    def update_vectorstore(self, vectorstore: FAISS, 
                          new_files: List[str], 
                          modified_files: List[str], 
                          deleted_files: List[str]) -> FAISS:
        """벡터스토어를 증분 업데이트합니다."""
        
        # 삭제된 파일 처리
        if deleted_files:
            log.info(f"삭제된 파일 {len(deleted_files)}개 감지")
            
            # 삭제 임계값 확인 및 재구성 여부 결정
            should_rebuild = (
                self.rebuild_on_delete and 
                len(deleted_files) >= self.delete_threshold
            )
            
            if should_rebuild:
                log.info(f"삭제 임계값({self.delete_threshold})에 도달하여 벡터스토어를 재구성합니다.")
                # 전체 재구성
                rebuilt_vectorstore = self._rebuild_vectorstore_from_existing_files()
                if rebuilt_vectorstore:
                    vectorstore = rebuilt_vectorstore
                    log.info("벡터스토어가 성공적으로 재구성되었습니다.")
                else:
                    log.error("벡터스토어 재구성에 실패했습니다.")
                    return vectorstore
            else:
                # 재구성하지 않는 경우 경고 메시지
                for file_path in deleted_files:
                    log.warning(f"삭제된 파일: {file_path} (벡터 데이터는 남아있음)")
                log.info("삭제된 파일의 벡터 데이터를 제거하려면 rebuild_on_delete=True로 설정하세요.")
        
        # 새로운/수정된 파일 처리
        files_to_add = new_files + modified_files
        
        if files_to_add:
            log.info(f"새로운/수정된 파일 처리 시작: {len(files_to_add)}개")
            new_documents = self._load_and_split_documents(files_to_add)
            
            if new_documents:
                try:
                    # 새로운 문서들을 기존 벡터스토어에 추가
                    vectorstore.add_documents(new_documents)
                    log.info(f"벡터스토어에 {len(new_documents)} 청크가 추가되었습니다.")
                except Exception as e:
                    log.error(f"벡터스토어 업데이트 중 오류 발생: {str(e)}")
                    # 오류 발생 시 새로운 벡터스토어 생성하여 병합
                    log.info("새로운 벡터스토어를 생성하여 병합합니다.")
                    temp_vectorstore = self.create_vectorstore_from_files(files_to_add)
                    if temp_vectorstore:
                        try:
                            vectorstore.merge_from(temp_vectorstore)
                            log.info("벡터스토어 병합이 완료되었습니다.")
                        except Exception as merge_error:
                            log.error(f"벡터스토어 병합 중 오류 발생: {str(merge_error)}")
        
        return vectorstore
    
    def update_file_metadata(self, processed_files: List[str]):
        """처리된 파일들의 메타데이터를 업데이트합니다."""
        stored_metadata = self._get_file_metadata()
        current_files = self._scan_pdf_files()
        
        # 처리된 파일들의 메타데이터 업데이트
        for file_path in processed_files:
            if file_path in current_files:
                current_files[file_path]["last_processed"] = datetime.now().isoformat()
                stored_metadata[file_path] = current_files[file_path]
        
        # 삭제된 파일들은 메타데이터에서도 제거
        current_file_set = set(current_files.keys())
        stored_metadata = {k: v for k, v in stored_metadata.items() if k in current_file_set}
        
        self._save_file_metadata(stored_metadata)
        log.info("파일 메타데이터가 업데이트되었습니다.")
    
    def get_vectorstore_stats(self) -> Dict[str, any]:
        """벡터스토어의 통계 정보를 반환합니다."""
        stats = {
            "vectorstore_exists": self.vectorstore_exists(),
            "metadata_file_exists": os.path.exists(self.metadata_file),
            "pdf_directory_exists": os.path.exists(self.pdf_dir),
            "total_files_in_metadata": 0,
            "total_pdf_files": 0,
            "config": {
                "rebuild_on_delete": self.rebuild_on_delete,
                "delete_threshold": self.delete_threshold,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap
            }
        }
        
        # 메타데이터 파일 통계
        metadata = self._get_file_metadata()
        stats["total_files_in_metadata"] = len(metadata)
        
        # 실제 PDF 파일 통계
        current_files = self._scan_pdf_files()
        stats["total_pdf_files"] = len(current_files)
        
        return stats
    
    def force_rebuild_vectorstore(self) -> Optional[FAISS]:
        """벡터스토어를 강제로 재구성합니다."""
        log.info("벡터스토어 강제 재구성을 시작합니다.")
        
        # 기존 벡터스토어 파일 삭제
        index_path = os.path.join(self.vectorstore_dir, "index.faiss")
        pkl_path = os.path.join(self.vectorstore_dir, "index.pkl")
        
        for path in [index_path, pkl_path]:
            if os.path.exists(path):
                os.remove(path)
                log.info(f"기존 파일 삭제: {path}")
        
        # 새로 생성
        vectorstore = self._rebuild_vectorstore_from_existing_files()
        if vectorstore:
            self.save_vectorstore(vectorstore)
            current_files = self._scan_pdf_files()
            self.update_file_metadata(list(current_files.keys()))
            log.info("벡터스토어 강제 재구성이 완료되었습니다.")
        
        return vectorstore
    
    def get_or_create_vectorstore(self) -> Optional[FAISS]:
        """벡터스토어를 가져오거나 새로 생성합니다. 증분 업데이트도 수행합니다."""
        log.info("벡터스토어 초기화를 시작합니다...")
        
        # 기존 벡터스토어 로드 시도
        vectorstore = self.load_vectorstore()
        
        # 파일 변경사항 확인
        new_files, modified_files, deleted_files = self.check_file_changes()
        
        # 변경사항이 있거나 벡터스토어가 없는 경우
        if vectorstore is None:
            log.info("기존 벡터스토어가 없습니다. 새로 생성합니다.")
            current_files = self._scan_pdf_files()
            all_files = list(current_files.keys())
            
            if not all_files:
                log.warning("처리할 PDF 파일이 없습니다.")
                return None
            
            vectorstore = self.create_vectorstore_from_files(all_files)
            if vectorstore:
                self.save_vectorstore(vectorstore)
                self.update_file_metadata(all_files)
            
        elif new_files or modified_files or deleted_files:
            log.info("파일 변경사항이 감지되었습니다. 증분 업데이트를 수행합니다.")
            vectorstore = self.update_vectorstore(vectorstore, new_files, modified_files, deleted_files)
            self.save_vectorstore(vectorstore)
            self.update_file_metadata(new_files + modified_files)
            
        else:
            log.info("변경사항이 없습니다. 기존 벡터스토어를 사용합니다.")
        
        return vectorstore