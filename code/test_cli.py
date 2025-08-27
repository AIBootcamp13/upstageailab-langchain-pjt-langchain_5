"""
CLI RAG 시스템

webui와 동일한 RAG 방식을 사용하여 하드코딩된 질문에 답변하는 CLI 도구입니다.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 환경 설정
script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir))
load_dotenv(script_dir / '.env')

from modules import RAGSystemInitializer
from modules.logger import LoggerManager

def main():
    """메인 함수"""
    # 로거 초기화
    log = LoggerManager("CLI")
    
    # 1. 시스템 초기화 (RAGQueryProcessor 포함, 메모리 기능 활성화, config 기반)
    result = RAGSystemInitializer.initialize_system(
        current_file_path=script_dir, 
        include_sql=False, 
        use_config=True,  # config.yaml 사용
        enable_db_memory=False  # CLI는 메모리만 사용
    )
    if not result: 
        return
    
    _, _, _, processor = result  # processor만 사용
    
    # 2. 첫 번째 질문
    log.info("=== 첫 번째 질문 ===")
    answer1 = processor.query("제과제빵에서 반죽 온도 관리 방법은 무엇인가요?", return_sources=True)
    if answer1["success"]:
        log.info("답변:", answer1["response"])
        log.info("출처:", answer1["sources"])
    else:
        log.error("오류:", answer1["error"])
    
    log.info("\n" + "="*50 + "\n")
    
    # 3. 두 번째 질문 (메모리 기능 확인)
    log.info("=== 두 번째 질문 (메모리 기능 확인) ===")
    answer2 = processor.query("방금 전에 물어본 질문이 뭐였나요?", return_sources=True)
    if answer2["success"]:
        log.info("답변:", answer2["response"])
        log.info("출처:", answer2["sources"])
    else:
        log.error("오류:", answer2["error"])

if __name__ == "__main__": 
    main()