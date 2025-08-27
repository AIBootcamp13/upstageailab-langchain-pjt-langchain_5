"""
설정 파일 로더 모듈

이 모듈은 config.yaml 파일을 로드하고 관리하는 기능을 제공합니다.

주요 기능:
1. YAML 설정 파일 로드
2. 설정 검증 및 기본값 처리
3. 싱글톤 패턴을 통한 전역 설정 관리
4. 모델별 조건부 설정 처리 (solar-pro2 -> reasoning_effort)
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from copy import deepcopy

from .logger import LoggerManager


class ConfigLoader:
    """YAML 기반 설정 파일 로더"""
    
    def __init__(self, config_path: str = None):
        """
        ConfigLoader 초기화
        
        Args:
            config_path (str, optional): 설정 파일 경로. 
                                      None이면 자동으로 config.yaml 탐색
        """
        self.logger = LoggerManager("ConfigLoader")
        
        # 설정 파일 경로 설정
        if config_path:
            self.config_path = Path(config_path)
        else:
            # 자동 경로 탐색: 현재 모듈 기준으로 code/config.yaml 찾기
            current_file = Path(__file__)
            code_dir = current_file.parent.parent  # modules의 상위 디렉토리
            self.config_path = code_dir / "config.yaml"
        
        # 설정 파일 존재 확인
        if not self.config_path.exists():
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {self.config_path}")
        
        # 설정 로드
        self._config = self._load_config()
        self._validate_config()
        
        self.logger.log_success(f"ConfigLoader 초기화 완료 - 설정 파일: {self.config_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """YAML 파일에서 설정을 로드"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            
            self.logger.log_step("설정 파일 로드 완료", str(self.config_path))
            return config or {}
            
        except yaml.YAMLError as e:
            self.logger.log_error("YAML 파싱 오류", e)
            raise ValueError(f"설정 파일 파싱 오류: {e}")
        except Exception as e:
            self.logger.log_error("설정 파일 로드 실패", e)
            raise
    
    def _get_default_config(self) -> Dict[str, Any]:
        """기본 설정값 반환"""
        return {
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
            },
            "logging": {
                "level": "INFO"
            },
            "database": {
                "enable_memory": False,
                "conversation_limit": 20
            }
        }
    
    def _validate_config(self):
        """설정 검증 및 기본값 적용"""
        default_config = self._get_default_config()
        
        # 누락된 섹션이나 키에 대해 기본값 적용
        for section, section_config in default_config.items():
            if section not in self._config:
                self._config[section] = section_config.copy()
                self.logger.log_warning_with_icon(f"설정 섹션 누락으로 기본값 적용: {section}")
            else:
                for key, default_value in section_config.items():
                    if key not in self._config[section]:
                        self._config[section][key] = default_value
                        self.logger.log_warning_with_icon(f"설정 키 누락으로 기본값 적용: {section}.{key}")
        
        # 특정 검증 규칙
        self._validate_specific_rules()
    
    def _validate_specific_rules(self):
        """특정 검증 규칙 적용"""
        # temperature 범위 검증
        for section in ["llm", "router"]:
            temp = self._config[section].get("temperature", 0.7)
            if not (0.0 <= temp <= 1.0):
                self.logger.log_warning_with_icon(f"temperature 값 범위 초과, 기본값으로 설정: {section}.temperature = {temp}")
                self._config[section]["temperature"] = 0.7 if section == "llm" else 0.1
        
        # top_k 양수 검증
        top_k = self._config["retriever"].get("top_k", 5)
        if not isinstance(top_k, int) or top_k < 1:
            self.logger.log_warning_with_icon(f"top_k 값이 유효하지 않음, 기본값으로 설정: top_k = {top_k}")
            self._config["retriever"]["top_k"] = 5
        
        # chunk_size 양수 검증
        chunk_size = self._config["vectorstore"].get("chunk_size", 1000)
        if not isinstance(chunk_size, int) or chunk_size < 100:
            self.logger.log_warning_with_icon(f"chunk_size 값이 유효하지 않음, 기본값으로 설정: chunk_size = {chunk_size}")
            self._config["vectorstore"]["chunk_size"] = 1000
    
    def get_config(self, section: str = None, key: str = None) -> Any:
        """
        설정값 조회
        
        Args:
            section (str, optional): 설정 섹션 (예: 'llm', 'router')
            key (str, optional): 설정 키 (예: 'main_model', 'temperature')
            
        Returns:
            Any: 설정값 또는 설정 딕셔너리
        """
        if section is None:
            return deepcopy(self._config)
        
        if section not in self._config:
            self.logger.log_warning_with_icon(f"존재하지 않는 설정 섹션: {section}")
            return None
        
        if key is None:
            return deepcopy(self._config[section])
        
        if key not in self._config[section]:
            self.logger.log_warning_with_icon(f"존재하지 않는 설정 키: {section}.{key}")
            return None
        
        return self._config[section][key]
    
    def get_llm_config(self) -> Dict[str, Any]:
        """
        LLM 설정 반환 (solar-pro2 조건부 처리 포함)
        
        Returns:
            Dict[str, Any]: LLM 초기화에 필요한 설정
        """
        llm_config = self.get_config("llm")
        model = llm_config.get("main_model", "solar-pro2")
        
        # solar-pro2 모델인 경우에만 reasoning_effort 포함
        if "solar-pro2" in model.lower():
            result = {
                "model": model,
                "temperature": llm_config.get("temperature", 0.7),
                "reasoning_effort": llm_config.get("reasoning_effort", "high")
            }
            self.logger.log_step("LLM 설정 (reasoning_effort 포함)", f"모델: {model}")
        else:
            result = {
                "model": model,
                "temperature": llm_config.get("temperature", 0.7)
            }
            self.logger.log_step("LLM 설정 (reasoning_effort 제외)", f"모델: {model}")
        
        return result
    
    def get_router_config(self) -> Dict[str, Any]:
        """라우터 설정 반환"""
        return {
            "router_model": self.get_config("router", "model"),
            "temperature": self.get_config("router", "temperature")
        }
    
    def get_embeddings_config(self) -> Dict[str, Any]:
        """임베딩 설정 반환"""
        return {
            "model": self.get_config("embeddings", "model")
        }
    
    def get_retriever_config(self) -> Dict[str, Any]:
        """리트리버 설정 반환"""
        return {
            "search_type": self.get_config("retriever", "search_type"),
            "k": self.get_config("retriever", "top_k"),
            "score_threshold": self.get_config("retriever", "score_threshold")
        }
    
    def get_vectorstore_config(self) -> Dict[str, Any]:
        """벡터스토어 설정 반환"""
        return {
            "chunk_size": self.get_config("vectorstore", "chunk_size"),
            "chunk_overlap": self.get_config("vectorstore", "chunk_overlap")
        }
    
    def get_database_config(self) -> Dict[str, Any]:
        """데이터베이스 설정 반환"""
        return {
            "enable_memory": self.get_config("database", "enable_memory"),
            "conversation_limit": self.get_config("database", "conversation_limit")
        }
    
    def update_config(self, section: str, key: str, value: Any):
        """
        설정값 업데이트 (런타임 전용, 파일에는 저장하지 않음)
        
        Args:
            section (str): 설정 섹션
            key (str): 설정 키
            value (Any): 새 값
        """
        if section not in self._config:
            self._config[section] = {}
        
        old_value = self._config[section].get(key, "없음")
        self._config[section][key] = value
        
        self.logger.log_step("설정 업데이트", 
                           f"{section}.{key}: {old_value} -> {value}")
    
    def reload_config(self):
        """설정 파일 다시 로드"""
        try:
            self._config = self._load_config()
            self._validate_config()
            self.logger.log_success("설정 파일 다시 로드 완료")
        except Exception as e:
            self.logger.log_error("설정 파일 다시 로드 실패", e)
            raise
    
    def get_config_summary(self) -> str:
        """설정 요약 정보 반환"""
        llm_config = self.get_llm_config()
        retriever_config = self.get_retriever_config()
        
        summary = f"""
=== RAG 시스템 설정 요약 ===
LLM 모델: {llm_config['model']}
LLM Temperature: {llm_config['temperature']}
Reasoning Effort: {llm_config.get('reasoning_effort', '사용안함')}
라우터 모델: {self.get_config('router', 'model')}
검색 방법: {retriever_config['search_type']}
상위 K개 문서: {retriever_config['k']}
청크 크기: {self.get_config('vectorstore', 'chunk_size')}
설정 파일: {self.config_path}
        """.strip()
        
        return summary


# 전역 인스턴스 (싱글톤 패턴)
_config_loader_instance = None


def get_config_loader(config_path: str = None) -> ConfigLoader:
    """
    ConfigLoader 싱글톤 인스턴스 반환
    
    Args:
        config_path (str, optional): 설정 파일 경로
        
    Returns:
        ConfigLoader: 설정 로더 인스턴스
    """
    global _config_loader_instance
    
    if _config_loader_instance is None:
        _config_loader_instance = ConfigLoader(config_path)
    
    return _config_loader_instance


def reset_config_loader():
    """설정 로더 인스턴스 초기화 (테스트용)"""
    global _config_loader_instance
    _config_loader_instance = None


def get_config(section: str = None, key: str = None) -> Any:
    """
    설정값 조회 (편의 함수)
    
    Args:
        section (str, optional): 설정 섹션
        key (str, optional): 설정 키
        
    Returns:
        Any: 설정값
    """
    loader = get_config_loader()
    return loader.get_config(section, key)