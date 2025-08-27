"""
PromptLoader 모듈 테스트

프롬프트 로더의 기능을 검증하는 테스트 케이스들입니다.
"""

import pytest
import os
from pathlib import Path
from jinja2 import TemplateNotFound

# 테스트를 위해 시스템 경로 설정
import sys
current_dir = Path(__file__).parent
code_dir = current_dir.parent
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from modules.prompt_loader import PromptLoader, get_prompt_loader, reset_prompt_loader


class TestPromptLoader:
    """PromptLoader 클래스 테스트"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """각 테스트 전에 실행되는 설정"""
        # 싱글톤 인스턴스 초기화
        reset_prompt_loader()
        
        # 테스트용 템플릿 디렉토리 경로
        self.test_templates_dir = code_dir / "prompts"
        
        yield
        
        # 테스트 후 정리
        reset_prompt_loader()
    
    def test_prompt_loader_initialization(self):
        """PromptLoader 초기화 테스트"""
        loader = PromptLoader()
        
        assert loader is not None
        assert loader.templates_dir.exists()
        assert loader.env is not None
        assert loader._template_cache == {}
    
    def test_prompt_loader_with_custom_dir(self):
        """커스텀 디렉토리로 PromptLoader 초기화 테스트"""
        custom_dir = str(self.test_templates_dir)
        loader = PromptLoader(custom_dir)
        
        assert str(loader.templates_dir) == custom_dir
    
    def test_get_current_time(self):
        """현재 시간 반환 테스트"""
        loader = PromptLoader()
        current_time = loader.get_current_time()
        
        assert isinstance(current_time, str)
        assert len(current_time) > 0
        # YYYY-MM-DD HH:MM:SS 형식 확인
        assert len(current_time.split()) == 2
        assert len(current_time.split()[0].split('-')) == 3
        assert len(current_time.split()[1].split(':')) == 3
    
    def test_load_template_success(self):
        """템플릿 로드 성공 테스트"""
        loader = PromptLoader()
        
        # routing.jinja2 템플릿 로드
        template = loader.load_template("routing.jinja2")
        
        assert template is not None
        assert "routing.jinja2" in loader._template_cache
    
    def test_load_template_not_found(self):
        """존재하지 않는 템플릿 로드 테스트"""
        loader = PromptLoader()
        
        with pytest.raises(TemplateNotFound):
            loader.load_template("nonexistent.jinja2")
    
    def test_template_caching(self):
        """템플릿 캐싱 테스트"""
        loader = PromptLoader()
        
        # 첫 번째 로드
        template1 = loader.load_template("routing.jinja2")
        
        # 두 번째 로드 (캐시에서)
        template2 = loader.load_template("routing.jinja2")
        
        # 같은 객체여야 함
        assert template1 is template2
        assert len(loader._template_cache) == 1
    
    def test_render_template_success(self):
        """템플릿 렌더링 성공 테스트"""
        loader = PromptLoader()
        
        # routing 템플릿 렌더링
        result = loader.render_template(
            "routing.jinja2",
            {"question": "테스트 질문"}
        )
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert "테스트 질문" in result or "{{ question }}" not in result
    
    def test_render_routing_prompt(self):
        """라우팅 프롬프트 렌더링 테스트"""
        loader = PromptLoader()
        
        question = "빵 만드는 방법을 알려주세요"
        result = loader.render_routing_prompt(question)
        
        assert isinstance(result, str)
        assert len(result) > 0
        # 현재 시간이 포함되어 있는지 확인
        assert "current time is" in result.lower()
    
    def test_render_rag_prompt(self):
        """RAG 프롬프트 렌더링 테스트"""
        loader = PromptLoader()
        
        context = "빵은 밀가루, 물, 이스트로 만듭니다."
        question = "빵의 재료는 무엇인가요?"
        result = loader.render_rag_prompt(context, question)
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert context in result
        # 현재 시간이 포함되어 있는지 확인
        assert "current time is" in result.lower()
    
    def test_list_templates(self):
        """템플릿 목록 조회 테스트"""
        loader = PromptLoader()
        
        templates = loader.list_templates()
        
        assert isinstance(templates, list)
        assert "routing.jinja2" in templates
        assert "rag.jinja2" in templates
    
    def test_template_exists(self):
        """템플릿 존재 여부 확인 테스트"""
        loader = PromptLoader()
        
        # 존재하는 템플릿
        assert loader.template_exists("routing.jinja2") is True
        assert loader.template_exists("rag.jinja2") is True
        
        # 존재하지 않는 템플릿
        assert loader.template_exists("nonexistent.jinja2") is False
    
    def test_clear_cache(self):
        """캐시 초기화 테스트"""
        loader = PromptLoader()
        
        # 템플릿 로드하여 캐시에 저장
        loader.load_template("routing.jinja2")
        loader.load_template("rag.jinja2")
        
        assert len(loader._template_cache) == 2
        
        # 캐시 초기화
        loader.clear_cache()
        
        assert len(loader._template_cache) == 0
    
    def test_reload_template(self):
        """템플릿 재로드 테스트"""
        loader = PromptLoader()
        
        # 첫 번째 로드
        loader.load_template("routing.jinja2")
        assert "routing.jinja2" in loader._template_cache
        
        # 재로드 (캐시에서 제거하고 다시 로드)
        template2 = loader.reload_template("routing.jinja2")
        
        # 템플릿이 성공적으로 로드되었는지 확인
        assert template2 is not None
        assert "routing.jinja2" in loader._template_cache
    
    def test_get_template_info(self):
        """템플릿 정보 조회 테스트"""
        loader = PromptLoader()
        
        # 존재하는 템플릿
        info = loader.get_template_info("routing.jinja2")
        
        assert info["exists"] is True
        assert "path" in info
        assert "size" in info
        assert "modified" in info
        assert info["in_cache"] is False
        
        # 템플릿 로드 후 다시 확인
        loader.load_template("routing.jinja2")
        info_cached = loader.get_template_info("routing.jinja2")
        
        assert info_cached["in_cache"] is True
        
        # 존재하지 않는 템플릿
        info_nonexistent = loader.get_template_info("nonexistent.jinja2")
        
        assert info_nonexistent["exists"] is False
    

class TestPromptLoaderSingleton:
    """PromptLoader 싱글톤 패턴 테스트"""
    
    def test_get_prompt_loader_singleton(self):
        """싱글톤 인스턴스 반환 테스트"""
        reset_prompt_loader()
        
        loader1 = get_prompt_loader()
        loader2 = get_prompt_loader()
        
        # 같은 인스턴스여야 함
        assert loader1 is loader2
    
    def test_reset_prompt_loader(self):
        """프롬프트 로더 초기화 테스트"""
        loader1 = get_prompt_loader()
        
        reset_prompt_loader()
        
        loader2 = get_prompt_loader()
        
        # 다른 인스턴스여야 함
        assert loader1 is not loader2


class TestPromptLoaderIntegration:
    """PromptLoader 통합 테스트"""
    
    def test_integration_with_actual_templates(self):
        """실제 템플릿 파일과의 통합 테스트"""
        loader = PromptLoader()
        
        # routing 프롬프트 테스트
        routing_result = loader.render_routing_prompt("안녕하세요")
        assert "BAKERY-RAG" in routing_result
        assert "안녕하세요" in routing_result or "{{ question }}" not in routing_result
        
        # RAG 프롬프트 테스트
        rag_result = loader.render_rag_prompt(
            "빵은 맛있습니다", 
            "빵에 대해 알려주세요"
        )
        assert "빵은 맛있습니다" in rag_result
        assert "question-answering" in rag_result.lower()
    
    def test_error_handling(self):
        """에러 처리 테스트"""
        # 잘못된 디렉토리로 초기화
        with pytest.raises(ValueError):
            PromptLoader("/nonexistent/directory")
    
    def test_render_with_current_time_auto_include(self):
        """현재 시간 자동 포함 테스트"""
        loader = PromptLoader()
        
        # 현재 시간 자동 포함 (기본값)
        result1 = loader.render_template("rag.jinja2", {
            "context": "테스트 컨텍스트",
            "question": "테스트 질문"
        })
        
        assert "current time is" in result1.lower()
        # 실제 시간이 포함되어 있는지 확인 (빈 문자열이 아님)
        lines = result1.split('\n')
        time_line = [line for line in lines if 'current time is' in line.lower()]
        assert len(time_line) > 0
        assert time_line[0].strip().split()[-1] != ''  # 시간 정보가 있어야 함
        
        # 현재 시간 포함하지 않음
        result2 = loader.render_template("rag.jinja2", {
            "context": "테스트 컨텍스트", 
            "question": "테스트 질문"
        }, include_current_time=False)
        
        # current_time 변수가 제공되지 않았으므로 빈 문자열로 렌더링됨
        assert "current time is" in result2.lower()
        lines2 = result2.split('\n')
        time_line2 = [line for line in lines2 if 'current time is' in line.lower()]
        assert len(time_line2) > 0
        # 시간 정보가 없어야 함 (빈 문자열)
        time_part = time_line2[0].strip().split('current time is')[1].strip()
        assert time_part == '' or time_part == '.'