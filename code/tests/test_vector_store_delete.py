"""
삭제된 파일 처리 기능 테스트

이 모듈은 VectorStoreManager의 삭제된 파일 처리 기능을 테스트합니다.
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

from modules.vector_store import VectorStoreManager


class TestVectorStoreDeleteHandling:
    """삭제된 파일 처리 기능 테스트 클래스"""
    
    @pytest.fixture
    def temp_dirs(self):
        """임시 디렉터리 생성"""
        pdf_dir = tempfile.mkdtemp(prefix="test_pdf_")
        vectorstore_dir = tempfile.mkdtemp(prefix="test_vectorstore_")
        yield pdf_dir, vectorstore_dir
        shutil.rmtree(pdf_dir, ignore_errors=True)
        shutil.rmtree(vectorstore_dir, ignore_errors=True)
    
    @pytest.fixture
    def mock_embeddings(self):
        """Mock 임베딩 객체"""
        return Mock()
    
    @pytest.fixture
    def manager_with_rebuild(self, temp_dirs, mock_embeddings):
        """재구성 옵션이 활성화된 VectorStoreManager"""
        pdf_dir, vectorstore_dir = temp_dirs
        return VectorStoreManager(
            pdf_dir=pdf_dir,
            vectorstore_dir=vectorstore_dir,
            embeddings=mock_embeddings,
            rebuild_on_delete=True,
            delete_threshold=1
        )
    
    @pytest.fixture
    def manager_without_rebuild(self, temp_dirs, mock_embeddings):
        """재구성 옵션이 비활성화된 VectorStoreManager"""
        pdf_dir, vectorstore_dir = temp_dirs
        return VectorStoreManager(
            pdf_dir=pdf_dir,
            vectorstore_dir=vectorstore_dir,
            embeddings=mock_embeddings,
            rebuild_on_delete=False,
            delete_threshold=1
        )
    
    def create_test_pdf(self, pdf_dir: str, filename: str) -> str:
        """테스트용 PDF 파일 생성 (실제로는 텍스트 파일)"""
        pdf_path = os.path.join(pdf_dir, filename)
        with open(pdf_path, 'w', encoding='utf-8') as f:
            f.write(f"Test content for {filename}")
        return pdf_path
    
    def create_test_metadata(self, manager: VectorStoreManager, files: list):
        """테스트용 메타데이터 생성"""
        metadata = {}
        for file_rel_path in files:
            file_abs_path = os.path.join(manager.pdf_dir, file_rel_path)
            if os.path.exists(file_abs_path):
                stat = os.stat(file_abs_path)
                metadata[file_rel_path] = {
                    "absolute_path": file_abs_path,
                    "size": stat.st_size,
                    "modified_time": stat.st_mtime,
                    "hash": manager._get_file_hash(file_abs_path),
                    "last_processed": "2024-01-01T00:00:00"
                }
            else:
                # 삭제된 파일을 시뮬레이션하기 위해 더미 데이터 사용
                metadata[file_rel_path] = {
                    "absolute_path": file_abs_path,
                    "size": 100,
                    "modified_time": 1640995200.0,
                    "hash": "dummy_hash",
                    "last_processed": "2024-01-01T00:00:00"
                }
        
        with open(manager.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    def test_delete_detection(self, manager_with_rebuild):
        """삭제된 파일 감지 테스트"""
        # 테스트 파일 생성
        self.create_test_pdf(manager_with_rebuild.pdf_dir, "test1.pdf")
        self.create_test_pdf(manager_with_rebuild.pdf_dir, "test2.pdf")
        
        # 메타데이터에는 3개 파일 등록 (하나는 삭제될 예정)
        self.create_test_metadata(manager_with_rebuild, ["test1.pdf", "test2.pdf", "test3.pdf"])
        
        # 파일 변경사항 확인
        new_files, modified_files, deleted_files = manager_with_rebuild.check_file_changes()
        
        assert len(deleted_files) == 1
        assert "test3.pdf" in deleted_files
        assert len(new_files) == 0
        assert len(modified_files) == 0
    
    def test_rebuild_on_delete_enabled(self, manager_with_rebuild):
        """삭제 시 재구성 활성화된 경우 테스트"""
        # Mock FAISS 벡터스토어
        mock_vectorstore = Mock()
        mock_rebuilt_vectorstore = Mock()
        
        with patch.object(manager_with_rebuild, '_rebuild_vectorstore_from_existing_files') as mock_rebuild:
            mock_rebuild.return_value = mock_rebuilt_vectorstore
            
            # 업데이트 실행
            result = manager_with_rebuild.update_vectorstore(
                vectorstore=mock_vectorstore,
                new_files=[],
                modified_files=[],
                deleted_files=["test1.pdf"]
            )
            
            # 재구성이 호출되었는지 확인
            mock_rebuild.assert_called_once()
            assert result == mock_rebuilt_vectorstore
    
    def test_rebuild_on_delete_disabled(self, manager_without_rebuild):
        """삭제 시 재구성 비활성화된 경우 테스트"""
        # Mock FAISS 벡터스토어
        mock_vectorstore = Mock()
        
        with patch.object(manager_without_rebuild, '_rebuild_vectorstore_from_existing_files') as mock_rebuild:
            # 업데이트 실행
            result = manager_without_rebuild.update_vectorstore(
                vectorstore=mock_vectorstore,
                new_files=[],
                modified_files=[],
                deleted_files=["test1.pdf"]
            )
            
            # 재구성이 호출되지 않았는지 확인
            mock_rebuild.assert_not_called()
            assert result == mock_vectorstore
    
    def test_delete_threshold(self, temp_dirs, mock_embeddings):
        """삭제 임계값 테스트"""
        pdf_dir, vectorstore_dir = temp_dirs
        manager = VectorStoreManager(
            pdf_dir=pdf_dir,
            vectorstore_dir=vectorstore_dir,
            embeddings=mock_embeddings,
            rebuild_on_delete=True,
            delete_threshold=3  # 3개 이상 삭제 시에만 재구성
        )
        
        mock_vectorstore = Mock()
        
        with patch.object(manager, '_rebuild_vectorstore_from_existing_files') as mock_rebuild:
            # 2개 파일 삭제 (임계값 미달)
            result = manager.update_vectorstore(
                vectorstore=mock_vectorstore,
                new_files=[],
                modified_files=[],
                deleted_files=["test1.pdf", "test2.pdf"]
            )
            
            # 재구성이 호출되지 않았는지 확인
            mock_rebuild.assert_not_called()
            assert result == mock_vectorstore
            
            # 3개 파일 삭제 (임계값 도달)
            result = manager.update_vectorstore(
                vectorstore=mock_vectorstore,
                new_files=[],
                modified_files=[],
                deleted_files=["test1.pdf", "test2.pdf", "test3.pdf"]
            )
            
            # 재구성이 호출되었는지 확인
            mock_rebuild.assert_called_once()
    
    def test_metadata_cleanup_after_deletion(self, manager_with_rebuild):
        """삭제 후 메타데이터 정리 테스트"""
        # 테스트 파일 생성
        self.create_test_pdf(manager_with_rebuild.pdf_dir, "test1.pdf")
        self.create_test_pdf(manager_with_rebuild.pdf_dir, "test2.pdf")
        
        # 메타데이터에는 3개 파일 등록
        self.create_test_metadata(manager_with_rebuild, ["test1.pdf", "test2.pdf", "test3.pdf"])
        
        # 메타데이터 업데이트 실행
        manager_with_rebuild.update_file_metadata(["test1.pdf", "test2.pdf"])
        
        # 메타데이터 확인
        metadata = manager_with_rebuild._get_file_metadata()
        assert len(metadata) == 2
        assert "test1.pdf" in metadata
        assert "test2.pdf" in metadata
        assert "test3.pdf" not in metadata
    
    def test_force_rebuild_vectorstore(self, manager_with_rebuild):
        """강제 재구성 테스트"""
        # 테스트 파일 생성
        self.create_test_pdf(manager_with_rebuild.pdf_dir, "test1.pdf")
        
        # 기존 벡터스토어 파일 생성 (더미)
        index_path = os.path.join(manager_with_rebuild.vectorstore_dir, "index.faiss")
        pkl_path = os.path.join(manager_with_rebuild.vectorstore_dir, "index.pkl")
        
        with open(index_path, 'w') as f:
            f.write("dummy faiss index")
        with open(pkl_path, 'w') as f:
            f.write("dummy pkl file")
        
        assert os.path.exists(index_path)
        assert os.path.exists(pkl_path)
        
        with patch.object(manager_with_rebuild, '_rebuild_vectorstore_from_existing_files') as mock_rebuild, \
             patch.object(manager_with_rebuild, 'save_vectorstore') as mock_save:
            
            mock_vectorstore = Mock()
            mock_rebuild.return_value = mock_vectorstore
            
            # 강제 재구성 실행
            result = manager_with_rebuild.force_rebuild_vectorstore()
            
            # 기존 파일이 삭제되었는지 확인
            assert not os.path.exists(index_path)
            assert not os.path.exists(pkl_path)
            
            # 재구성과 저장이 호출되었는지 확인
            mock_rebuild.assert_called_once()
            mock_save.assert_called_once()
            assert result == mock_vectorstore
    
    def test_get_vectorstore_stats(self, manager_with_rebuild):
        """통계 정보 반환 테스트"""
        # 테스트 파일 생성
        self.create_test_pdf(manager_with_rebuild.pdf_dir, "test1.pdf")
        self.create_test_metadata(manager_with_rebuild, ["test1.pdf", "test2.pdf"])
        
        stats = manager_with_rebuild.get_vectorstore_stats()
        
        assert "vectorstore_exists" in stats
        assert "metadata_file_exists" in stats
        assert "pdf_directory_exists" in stats
        assert "total_files_in_metadata" in stats
        assert "total_pdf_files" in stats
        assert "config" in stats
        
        assert stats["total_files_in_metadata"] == 2
        assert stats["total_pdf_files"] == 1
        assert stats["config"]["rebuild_on_delete"] == True
        assert stats["config"]["delete_threshold"] == 1