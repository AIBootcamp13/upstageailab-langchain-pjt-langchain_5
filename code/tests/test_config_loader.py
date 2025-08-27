"""
ConfigLoader 모듈 테스트

설정 파일 로더의 기능을 검증하는 테스트 케이스들입니다.
"""

import pytest
import os
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch

# 테스트를 위해 시스템 경로 설정
import sys
current_dir = Path(__file__).parent
code_dir = current_dir.parent
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from modules.config_loader import ConfigLoader, get_config_loader, reset_config_loader, get_config


class TestConfigLoader:
    """ConfigLoader 클래스 테스트"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """각 테스트 전에 실행되는 설정"""
        # 싱글톤 인스턴스 초기화
        reset_config_loader()
        
        yield
        
        # 테스트 후 정리
        reset_config_loader()
    
    @pytest.fixture
    def temp_config_file(self):
        """임시 설정 파일 생성"""
        config_data = {
            "llm": {
                "main_model": "solar-pro2",
                "reasoning_effort": "high",
                "temperature": 0.7
            },
            "router": {
                "model": "upstage/solar-1-mini-chat",
                "temperature": 0.1
            },
            "embeddings": {
                "model": "embedding-query"
            },
            "retriever": {
                "search_type": "similarity",
                "top_k": 5,
                "score_threshold": 0.8
            },
            "vectorstore": {
                "chunk_size": 1000,
                "chunk_overlap": 50
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f, default_flow_style=False)
            temp_path = f.name
        
        yield temp_path
        
        # 임시 파일 삭제
        os.unlink(temp_path)
    
    @pytest.fixture
    def invalid_config_file(self):
        """잘못된 형식의 설정 파일 생성"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [\n")  # 잘못된 YAML 형식
            temp_path = f.name
        
        yield temp_path
        
        # 임시 파일 삭제
        os.unlink(temp_path)
    
    def test_config_loader_initialization(self, temp_config_file):
        """ConfigLoader 초기화 테스트"""
        loader = ConfigLoader(temp_config_file)
        
        assert loader is not None
        assert loader.config_path == Path(temp_config_file)
        assert loader._config is not None
    
    def test_config_loader_file_not_found(self):
        """존재하지 않는 설정 파일 테스트"""
        with pytest.raises(FileNotFoundError):
            ConfigLoader("/nonexistent/config.yaml")
    
    def test_config_loader_invalid_yaml(self, invalid_config_file):
        """잘못된 YAML 형식 테스트"""
        with pytest.raises(ValueError):
            ConfigLoader(invalid_config_file)
    
    def test_get_config_all(self, temp_config_file):
        """전체 설정 조회 테스트"""
        loader = ConfigLoader(temp_config_file)
        config = loader.get_config()
        
        assert isinstance(config, dict)
        assert "llm" in config
        assert "router" in config
        assert "retriever" in config
    
    def test_get_config_section(self, temp_config_file):
        """섹션별 설정 조회 테스트"""
        loader = ConfigLoader(temp_config_file)
        
        llm_config = loader.get_config("llm")
        assert llm_config["main_model"] == "solar-pro2"
        assert llm_config["temperature"] == 0.7
        
        # 존재하지 않는 섹션
        none_section = loader.get_config("nonexistent")
        assert none_section is None
    
    def test_get_config_key(self, temp_config_file):
        """키별 설정 조회 테스트"""
        loader = ConfigLoader(temp_config_file)
        
        model = loader.get_config("llm", "main_model")
        assert model == "solar-pro2"
        
        # 존재하지 않는 키
        none_key = loader.get_config("llm", "nonexistent")
        assert none_key is None
    
    def test_get_llm_config_solar_pro2(self, temp_config_file):
        """Solar-Pro2 모델 LLM 설정 테스트 (reasoning_effort 포함)"""
        loader = ConfigLoader(temp_config_file)
        llm_config = loader.get_llm_config()
        
        assert llm_config["model"] == "solar-pro2"
        assert llm_config["temperature"] == 0.7
        assert llm_config["reasoning_effort"] == "high"  # solar-pro2이므로 포함되어야 함
    
    def test_get_llm_config_other_model(self, temp_config_file):
        """다른 모델 LLM 설정 테스트 (reasoning_effort 제외)"""
        # 임시로 모델을 변경
        with open(temp_config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        config_data["llm"]["main_model"] = "solar-1-mini"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f, default_flow_style=False)
            new_temp_path = f.name
        
        try:
            loader = ConfigLoader(new_temp_path)
            llm_config = loader.get_llm_config()
            
            assert llm_config["model"] == "solar-1-mini"
            assert llm_config["temperature"] == 0.7
            assert "reasoning_effort" not in llm_config  # solar-pro2가 아니므로 제외되어야 함
        
        finally:
            os.unlink(new_temp_path)
    
    def test_get_router_config(self, temp_config_file):
        """라우터 설정 조회 테스트"""
        loader = ConfigLoader(temp_config_file)
        router_config = loader.get_router_config()
        
        assert router_config["router_model"] == "upstage/solar-1-mini-chat"
        assert router_config["temperature"] == 0.1
    
    def test_get_embeddings_config(self, temp_config_file):
        """임베딩 설정 조회 테스트"""
        loader = ConfigLoader(temp_config_file)
        embeddings_config = loader.get_embeddings_config()
        
        assert embeddings_config["model"] == "embedding-query"
    
    def test_get_retriever_config(self, temp_config_file):
        """리트리버 설정 조회 테스트"""
        loader = ConfigLoader(temp_config_file)
        retriever_config = loader.get_retriever_config()
        
        assert retriever_config["search_type"] == "similarity"
        assert retriever_config["k"] == 5
        assert retriever_config["score_threshold"] == 0.8
    
    def test_get_vectorstore_config(self, temp_config_file):
        """벡터스토어 설정 조회 테스트"""
        loader = ConfigLoader(temp_config_file)
        vectorstore_config = loader.get_vectorstore_config()
        
        assert vectorstore_config["chunk_size"] == 1000
        assert vectorstore_config["chunk_overlap"] == 50
    
    def test_config_validation_missing_sections(self):
        """누락된 섹션에 대한 기본값 적용 테스트"""
        # 최소한의 설정 파일
        minimal_config = {"llm": {"main_model": "test-model"}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(minimal_config, f, default_flow_style=False)
            temp_path = f.name
        
        try:
            loader = ConfigLoader(temp_path)
            
            # 기본값이 적용되었는지 확인
            router_config = loader.get_config("router")
            assert router_config is not None
            assert "model" in router_config
            
            retriever_config = loader.get_config("retriever")
            assert retriever_config is not None
            assert "top_k" in retriever_config
            
        finally:
            os.unlink(temp_path)
    
    def test_update_config(self, temp_config_file):
        """설정값 업데이트 테스트"""
        loader = ConfigLoader(temp_config_file)
        
        # 기존값 확인
        old_value = loader.get_config("llm", "temperature")
        assert old_value == 0.7
        
        # 값 업데이트
        loader.update_config("llm", "temperature", 0.9)
        
        # 새값 확인
        new_value = loader.get_config("llm", "temperature")
        assert new_value == 0.9
    
    def test_get_config_summary(self, temp_config_file):
        """설정 요약 정보 테스트"""
        loader = ConfigLoader(temp_config_file)
        summary = loader.get_config_summary()
        
        assert isinstance(summary, str)
        assert "solar-pro2" in summary
        assert "similarity" in summary
        assert "5" in summary  # top_k 값
    

class TestConfigLoaderSingleton:
    """ConfigLoader 싱글톤 패턴 테스트"""
    
    @pytest.fixture
    def temp_config_file(self):
        """임시 설정 파일 생성"""
        config_data = {
            "llm": {
                "main_model": "solar-pro2",
                "reasoning_effort": "high",
                "temperature": 0.7
            },
            "router": {
                "model": "upstage/solar-1-mini-chat",
                "temperature": 0.1
            },
            "embeddings": {
                "model": "embedding-query"
            },
            "retriever": {
                "search_type": "similarity",
                "top_k": 5,
                "score_threshold": 0.8
            },
            "vectorstore": {
                "chunk_size": 1000,
                "chunk_overlap": 50
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f, default_flow_style=False)
            temp_path = f.name
        
        yield temp_path
        
        # 임시 파일 삭제
        os.unlink(temp_path)
    
    def test_get_config_loader_singleton(self, temp_config_file):
        """싱글톤 인스턴스 반환 테스트"""
        reset_config_loader()
        
        loader1 = get_config_loader(temp_config_file)
        loader2 = get_config_loader()
        
        # 같은 인스턴스여야 함
        assert loader1 is loader2
    
    def test_reset_config_loader(self, temp_config_file):
        """설정 로더 초기화 테스트"""
        loader1 = get_config_loader(temp_config_file)
        
        reset_config_loader()
        
        loader2 = get_config_loader(temp_config_file)
        
        # 다른 인스턴스여야 함
        assert loader1 is not loader2
    
    def test_get_config_convenience_function(self, temp_config_file):
        """편의 함수 테스트"""
        reset_config_loader()
        get_config_loader(temp_config_file)
        
        # 편의 함수를 통한 설정 조회
        model = get_config("llm", "main_model")
        assert model == "solar-pro2"
        
        llm_section = get_config("llm")
        assert llm_section["main_model"] == "solar-pro2"


class TestConfigLoaderIntegration:
    """ConfigLoader 통합 테스트"""
    
    def test_integration_with_actual_config(self):
        """실제 config.yaml 파일과의 통합 테스트"""
        # 실제 config.yaml 파일이 존재한다면 테스트
        config_path = code_dir / "config.yaml"
        if config_path.exists():
            loader = ConfigLoader(str(config_path))
            
            # 기본적인 설정들이 있는지 확인
            llm_config = loader.get_llm_config()
            assert "model" in llm_config
            assert "temperature" in llm_config
            
            retriever_config = loader.get_retriever_config()
            assert "k" in retriever_config
            assert "search_type" in retriever_config
    
    def test_auto_path_resolution(self):
        """자동 경로 해결 테스트"""
        # 실제 config.yaml이 존재할 때만 테스트
        config_path = code_dir / "config.yaml"
        if config_path.exists():
            # 경로를 지정하지 않고 초기화
            loader = ConfigLoader()
            assert loader.config_path.exists()
            
            # 설정이 정상적으로 로드되었는지 확인
            config = loader.get_config()
            assert isinstance(config, dict)
    
    def test_error_handling_with_defaults(self):
        """에러 상황에서의 기본값 처리 테스트"""
        # 빈 설정 파일
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")  # 빈 파일
            temp_path = f.name
        
        try:
            loader = ConfigLoader(temp_path)
            
            # 빈 파일이지만 기본값이 적용되어야 함
            llm_config = loader.get_llm_config()
            assert "model" in llm_config
            assert "temperature" in llm_config
            
            retriever_config = loader.get_retriever_config()
            assert "k" in retriever_config
            assert retriever_config["k"] == 5  # 기본값
            
        finally:
            os.unlink(temp_path)
    
    def test_parameter_validation(self):
        """파라미터 유효성 검증 테스트"""
        # 잘못된 값들을 포함한 설정 파일 생성
        invalid_config = {
            "llm": {
                "main_model": "solar-pro2",
                "temperature": 1.5  # 범위 초과
            },
            "retriever": {
                "top_k": -1,  # 음수
                "search_type": "similarity"
            },
            "vectorstore": {
                "chunk_size": 50  # 너무 작음
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_config, f, default_flow_style=False)
            temp_path = f.name
        
        try:
            loader = ConfigLoader(temp_path)
            
            # 잘못된 값들이 기본값으로 수정되었는지 확인
            llm_config = loader.get_llm_config()
            assert llm_config["temperature"] == 0.7  # 기본값으로 수정
            
            retriever_config = loader.get_retriever_config()
            assert retriever_config["k"] == 5  # 기본값으로 수정
            
            vectorstore_config = loader.get_vectorstore_config()
            assert vectorstore_config["chunk_size"] == 1000  # 기본값으로 수정
            
        finally:
            os.unlink(temp_path)