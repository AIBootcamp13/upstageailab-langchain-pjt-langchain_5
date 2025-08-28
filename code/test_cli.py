"""
CLI RAG 시스템

webui와 동일한 라우터 및 메모리 방식을 사용하여 하드코딩된 질문에 답변하는 CLI 도구입니다.
DB 저장 없이 메모리만 사용합니다.
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
    
    # 1. 시스템 초기화 (라우터 포함, 메모리 기능 활성화, config 기반)
    result = RAGSystemInitializer.initialize_system(
        current_file_path=script_dir, 
        include_sql=False, 
        use_config=True,  # config.yaml 사용
        enable_db_memory=False  # CLI는 메모리만 사용
    )
    if not result: 
        return
    
    vector_manager, llm_manager, retriever_manager, query_processor = result
    
    # 2. 메모리 관리자 초기화 (DB 저장 없음, 메모리만 사용)
    try:
        from modules.config_loader import get_config_loader
        config_loader = get_config_loader()
        db_conf = config_loader.get_database_config()
        memory_k = db_conf.get("memory_window", 3)
    except Exception:
        memory_k = 3

    from modules import ChatHistoryManager
    chat_manager = ChatHistoryManager(
        session_id="cli_test",
        memory_k=memory_k,
        sql_manager=None,  # DB 저장 안함
        auto_save=False
    )
    
    # 3. 첫 번째 질문
    log.info("=== 첫 번째 질문 ===")
    question1 = "제과제빵에서 반죽 온도 관리 방법은 무엇인가요?"
    
    # 통합 질의 처리 (라우팅 + 분기 + 메모리)
    result1 = query_processor.unified_query(
        question=question1,
        chat_history_manager=chat_manager,
        auto_save=True  # 메모리에 저장
    )
    
    if result1["success"]:
        log.info("답변:", result1["response"])
        log.info("처리 타입:", result1["processing_type"])
        if result1.get("sources"):
            log.info("출처:", result1["sources"])
    else:
        log.error("오류:", result1["error"])
    
    log.info("\n" + "="*50 + "\n")
    
    # 4. 두 번째 질문 (메모리 기능 확인) - 같은 chat_manager 사용
    log.info("=== 두 번째 질문 (메모리 기능 확인) ===")
    question2 = "방금 전에 물어본 질문이 뭐였나요?"
    
    # 통합 질의 처리 (라우팅 + 분기 + 메모리) - 같은 chat_manager 사용
    result2 = query_processor.unified_query(
        question=question2,
        chat_history_manager=chat_manager,  # 같은 인스턴스 사용
        auto_save=True  # 메모리에 저장
    )
    
    if result2["success"]:
        log.info("답변:", result2["response"])
        log.info("처리 타입:", result2["processing_type"])
        if result2.get("sources"):
            log.info("출처:", result2["sources"])
    else:
        log.error("오류:", result2["error"])
    
    # 5. 메모리 상태 확인
    log.info("\n" + "="*50)
    log.info("=== 메모리 상태 확인 ===")
    summary = chat_manager.get_conversation_summary()
    log.info(f"전체 메시지: {summary.get('total_messages', 0)}개")
    log.info(f"메모리 메시지: {summary.get('memory_messages', 0)}개")
    
    # 6. 실제 메모리 내용 확인
    log.info("\n=== 실제 메모리 내용 ===")
    try:
        # LangChain 메모리에서 실제 대화 내용 확인
        memory_messages = chat_manager.memory.chat_memory.messages
        log.info(f"메모리에 저장된 메시지 수: {len(memory_messages)}")
        
        for i, msg in enumerate(memory_messages):
            if hasattr(msg, 'content'):
                content = msg.content
                role = "사용자" if hasattr(msg, 'type') and msg.type == 'human' else "AI"
                log.info(f"{i+1}. {role}: {content[:100]}{'...' if len(content) > 100 else ''}")
            else:
                log.info(f"{i+1}. 알 수 없는 메시지 타입: {type(msg)}")
                
    except Exception as e:
        log.error(f"메모리 내용 확인 중 오류: {str(e)}")
    
    # 7. 세 번째 질문으로 메모리 기능 추가 테스트
    log.info("\n" + "="*50)
    log.info("=== 세 번째 질문 (메모리 연속성 테스트) ===")
    question3 = "첫 번째 질문에 대한 답변의 핵심 요점을 요약해줘"
    
    result3 = query_processor.unified_query(
        question=question3,
        chat_history_manager=chat_manager,  # 같은 인스턴스 사용
        auto_save=True  # 메모리에 저장
    )
    
    if result3["success"]:
        log.info("답변:", result3["response"])
        log.info("처리 타입:", result3["processing_type"])
    else:
        log.error("오류:", result3["error"])
    
    # 8. 최종 메모리 상태 확인
    log.info("\n" + "="*50)
    log.info("=== 최종 메모리 상태 ===")
    final_summary = chat_manager.get_conversation_summary()
    log.info(f"전체 메시지: {final_summary.get('total_messages', 0)}개")
    log.info(f"메모리 메시지: {final_summary.get('memory_messages', 0)}개")

if __name__ == "__main__": 
    main()