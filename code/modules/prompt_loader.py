"""
프롬프트 템플릿 로더 모듈

이 모듈은 Jinja2 기반 프롬프트 템플릿을 로드하고 관리하는 기능을 제공합니다.

주요 기능:
1. Jinja2 템플릿 파일 로드
2. 템플릿 캐싱
3. 동적 변수 바인딩
4. 템플릿 렌더링
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader, Template, TemplateNotFound
from datetime import datetime
import pytz

from .logger import LoggerManager


class PromptLoader:
    """Jinja2 기반 프롬프트 템플릿 로더"""
    
    def __init__(self, templates_dir: str = None):
        """
        PromptLoader 초기화
        
        Args:
            templates_dir (str, optional): 템플릿 디렉토리 경로. 
                                         None이면 기본 prompts 디렉토리 사용
        """
        self.logger = LoggerManager("PromptLoader")
        
        # 템플릿 디렉토리 설정
        if templates_dir:
            self.templates_dir = Path(templates_dir)
        else:
            # 현재 스크립트 기준으로 prompts 디렉토리 찾기
            current_file = Path(__file__)
            code_dir = current_file.parent.parent  # modules의 상위 디렉토리 (code)
            self.templates_dir = code_dir / "prompts"
        
        # 템플릿 디렉토리 존재 확인
        if not self.templates_dir.exists():
            raise ValueError(f"템플릿 디렉토리를 찾을 수 없습니다: {self.templates_dir}")
        
        # Jinja2 환경 설정
        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True
        )
        
        # 템플릿 캐시
        self._template_cache = {}
        
        # 한국 시간대 설정
        self.tz = pytz.timezone("Asia/Seoul")
        
        self.logger.log_success(f"PromptLoader 초기화 완료 - 템플릿 디렉토리: {self.templates_dir}")
    
    def get_current_time(self) -> str:
        """현재 시간을 한국 시간대로 반환"""
        return datetime.now(self.tz).strftime('%Y-%m-%d %H:%M:%S')
    
    def load_template(self, template_name: str, use_cache: bool = True) -> Template:
        """
        템플릿 파일을 로드
        
        Args:
            template_name (str): 템플릿 파일명 (예: 'routing.jinja2')
            use_cache (bool): 캐시 사용 여부
            
        Returns:
            Template: Jinja2 템플릿 객체
            
        Raises:
            TemplateNotFound: 템플릿 파일이 존재하지 않는 경우
        """
        self.logger.log_function_start("load_template", template_name=template_name)
        
        # 캐시 확인
        if use_cache and template_name in self._template_cache:
            self.logger.log_step("템플릿 캐시 히트", template_name)
            return self._template_cache[template_name]
        
        try:
            # 템플릿 로드
            template = self.env.get_template(template_name)
            
            # 캐시 저장
            if use_cache:
                self._template_cache[template_name] = template
                self.logger.log_step("템플릿 캐시 저장", template_name)
            
            self.logger.log_function_end("load_template", f"템플릿 로드 성공: {template_name}")
            return template
            
        except TemplateNotFound as e:
            self.logger.log_error("템플릿 로드 실패", e)
            raise TemplateNotFound(f"템플릿을 찾을 수 없습니다: {template_name}")
    
    def render_template(self, 
                       template_name: str, 
                       variables: Dict[str, Any] = None, 
                       include_current_time: bool = True) -> str:
        """
        템플릿을 렌더링하여 문자열로 반환
        
        Args:
            template_name (str): 템플릿 파일명
            variables (Dict[str, Any], optional): 템플릿 변수
            include_current_time (bool): 현재 시간 자동 포함 여부
            
        Returns:
            str: 렌더링된 텍스트
        """
        self.logger.log_function_start("render_template", 
                                     template_name=template_name,
                                     variables=list(variables.keys()) if variables else [])
        
        try:
            # 템플릿 로드
            template = self.load_template(template_name)
            
            # 변수 준비
            render_vars = variables.copy() if variables else {}
            
            # 현재 시간 자동 포함
            if include_current_time:
                render_vars['current_time'] = self.get_current_time()
            
            # 템플릿 렌더링
            rendered = template.render(render_vars)
            
            self.logger.log_function_end("render_template", "템플릿 렌더링 성공")
            return rendered
            
        except Exception as e:
            self.logger.log_error("render_template", e)
            raise
    
    def render_routing_prompt(self, question: str) -> str:
        """
        라우팅 프롬프트 렌더링
        
        Args:
            question (str): 사용자 질문
            
        Returns:
            str: 렌더링된 라우팅 프롬프트
        """
        return self.render_template(
            "routing.jinja2",
            {"question": question}
        )
    
    def render_rag_prompt(self, context: str, question: str) -> str:
        """
        RAG 프롬프트 렌더링
        
        Args:
            context (str): 검색된 문서 컨텍스트
            question (str): 사용자 질문
            
        Returns:
            str: 렌더링된 RAG 프롬프트
        """
        return self.render_template(
            "rag.jinja2",
            {
                "context": context,
                "question": question
            }
        )
    
    def list_templates(self) -> list:
        """
        사용 가능한 템플릿 목록 반환
        
        Returns:
            list: 템플릿 파일 목록
        """
        try:
            templates = []
            for file_path in self.templates_dir.glob("*.jinja2"):
                templates.append(file_path.name)
            
            self.logger.log_step("템플릿 목록 조회", f"{len(templates)}개 발견")
            return sorted(templates)
            
        except Exception as e:
            self.logger.log_error("list_templates", e)
            return []
    
    def template_exists(self, template_name: str) -> bool:
        """
        템플릿 파일 존재 여부 확인
        
        Args:
            template_name (str): 템플릿 파일명
            
        Returns:
            bool: 존재 여부
        """
        template_path = self.templates_dir / template_name
        return template_path.exists()
    
    def clear_cache(self):
        """템플릿 캐시 초기화"""
        cache_size = len(self._template_cache)
        self._template_cache.clear()
        self.logger.log_step("캐시 초기화", f"{cache_size}개 템플릿 캐시 삭제")
    
    def reload_template(self, template_name: str) -> Template:
        """
        템플릿을 캐시에서 제거하고 다시 로드
        
        Args:
            template_name (str): 템플릿 파일명
            
        Returns:
            Template: 새로 로드된 템플릿 객체
        """
        # 캐시에서 제거
        if template_name in self._template_cache:
            del self._template_cache[template_name]
            self.logger.log_step("템플릿 캐시 제거", template_name)
        
        # 다시 로드
        return self.load_template(template_name, use_cache=True)
    
    def get_template_info(self, template_name: str) -> Dict[str, Any]:
        """
        템플릿 정보 반환
        
        Args:
            template_name (str): 템플릿 파일명
            
        Returns:
            Dict[str, Any]: 템플릿 정보
        """
        template_path = self.templates_dir / template_name
        
        if not template_path.exists():
            return {"exists": False}
        
        stat = template_path.stat()
        
        return {
            "exists": True,
            "path": str(template_path),
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            "in_cache": template_name in self._template_cache
        }


# 전역 인스턴스 (싱글톤 패턴)
_prompt_loader_instance = None


def get_prompt_loader(templates_dir: str = None) -> PromptLoader:
    """
    PromptLoader 싱글톤 인스턴스 반환
    
    Args:
        templates_dir (str, optional): 템플릿 디렉토리 경로
        
    Returns:
        PromptLoader: 프롬프트 로더 인스턴스
    """
    global _prompt_loader_instance
    
    if _prompt_loader_instance is None:
        _prompt_loader_instance = PromptLoader(templates_dir)
    
    return _prompt_loader_instance


def reset_prompt_loader():
    """프롬프트 로더 인스턴스 초기화 (테스트용)"""
    global _prompt_loader_instance
    _prompt_loader_instance = None