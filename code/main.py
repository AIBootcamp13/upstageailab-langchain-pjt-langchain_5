"""
Streamlit Chat WebUI

LangChain 기반 RAG 시스템을 위한 Streamlit 웹 인터페이스입니다.

주요 기능:
- 채팅 인터페이스
- 대화 히스토리 관리
- 설정 패널
- 문서 검색 및 응답 생성
"""

import os
import sys
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta
import pytz
from langchain_upstage import UpstageEmbeddings

# 현재 스크립트의 디렉토리를 sys.path에 추가
script_dir = Path(__file__).parent.absolute()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

# 모듈 import
from modules import (
    SQLManager, VectorStoreManager, 
    LLMManager, RetrieverManager, ChatHistoryManager,
    QueryRouter, RAGSystemInitializer, RAGQueryProcessor
)
from modules.config_loader import get_config_loader

# 환경변수 로드 (스크립트 디렉토리 기준)
load_dotenv(script_dir / '.env')

# 페이지 설정
st.set_page_config(
    page_title="제과제빵 AI 어시스턴트",
    page_icon="🍞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일링 (빵 테마)
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.8rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        background: linear-gradient(135deg, #f7f3e9 0%, #f0e6d6 100%);
        color: #8b4513;
    }
    .user-message {
        background: linear-gradient(135deg, #f7f3e9 0%, #f0e6d6 100%);
        margin-left: 20%;
        border-left: 4px solid #d4aa7d;
        color: #8b4513;
    }
    .assistant-message {
        background: linear-gradient(135deg, #f4e4bc 0%, #e8d5a3 100%);
        margin-right: 20%;
        border-left: 4px solid #b8860b;
        color: #8b4513;
    }
    .message-header {
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: #8b4513;
    }
    .stApp > header {
        background-color: transparent;
    }
    .stApp {
        margin-top: -80px;
        background: linear-gradient(180deg, #faf8f3 0%, #f5f0e8 100%);
    }
    .stSidebar {
        background: linear-gradient(180deg, #f9f6f0 0%, #f2e9d8 100%);
    }
    .stSidebar > div {
        background: linear-gradient(180deg, #f9f6f0 0%, #f2e9d8 100%);
    }
    .stSidebar > div > div {
        background: linear-gradient(180deg, #f9f6f0 0%, #f2e9d8 100%);
    }
    .stButton > button {
        background: linear-gradient(135deg, #e6b800 0%, #d4aa7d 100%);
        color: #ffffff;
        border: none;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #d4aa7d 0%, #b8860b 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stChatFloatingInputContainer {
        background-color: transparent !important;
        background: transparent !important;
    }
    .stChatFloatingInputContainer > * {
        background-color: transparent !important;
        background: transparent !important;
    }
    div[data-testid="stChatInput"] {
        border-radius: 1.5rem !important;
        overflow: visible !important;
        background: linear-gradient(135deg, #f7f3e9 0%, #f0e6d6 100%) !important;
        border: 1px solid #d4aa7d !important;
        min-height: 3rem !important;
        height: 3rem !important;
        max-height: 3rem !important;
        display: flex !important;
        align-items: center !important;
    }
    div[data-testid="stChatInput"] > div,
    div[data-testid="stChatInput"] > div > div,
    div[data-testid="stChatInput"] > div > div > div,
    div[data-testid="stChatInput"] > div > div > div > div {
        background: transparent !important;
        border: none !important;
        border-radius: inherit !important;
        height: 100% !important;
        display: flex !important;
        align-items: center !important;
    }
    div[data-testid="stChatInput"] form {
        background: transparent !important;
        border: none !important;
        display: flex !important;
        align-items: center !important;
        height: 100% !important;
        width: 100% !important;
        padding: 0.25rem !important;
        box-sizing: border-box !important;
        margin: 0 !important;
    }
    div[data-testid="stChatInput"] textarea {
        background: transparent !important;
        border: none !important;
        color: #8b4513 !important;
        padding: 0 1rem !important;
        flex: 1 !important;
        resize: none !important;
        height: 100% !important;
        line-height: 2.5rem !important;
        overflow: hidden !important;
        box-sizing: border-box !important;
        vertical-align: middle !important;
        display: flex !important;
        align-items: center !important;
        font-size: 1rem !important;
    }
    div[data-testid="stChatInput"] button {
        background: linear-gradient(135deg, #e6b800 0%, #d4aa7d 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 50% !important;
        width: 2.2rem !important;
        height: 2.2rem !important;
        margin: auto 0.4rem !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        flex-shrink: 0 !important;
        align-self: center !important;
        min-width: 2.2rem !important;
        min-height: 2.2rem !important;
        max-width: 2.2rem !important;
        max-height: 2.2rem !important;
    }
    div[data-testid="stChatInput"] button:hover {
        background: linear-gradient(135deg, #d4aa7d 0%, #b8860b 100%) !important;
    }
    div[data-testid="stChatInput"] button svg {
        fill: white !important;
    }
    .main > div {
        background-color: transparent !important;
    }
    [data-testid="stAppViewBlockContainer"] {
        background-color: transparent !important;
    }
    section[data-testid="stChatFlowMessage"] {
        background-color: transparent !important;
    }
    .stBottom {
        background-color: transparent !important;
        background: transparent !important;
    }
    .stBottom > div {
        background-color: transparent !important;
        background: transparent !important;
    }

    /* 다크모드 감지 및 적용 */
    @media (prefers-color-scheme: dark) {
        .chat-message {
            background: linear-gradient(135deg, #232323 0%, #2c2c2c 100%) !important;
            color: #f0e6d6 !important;
        }
        .user-message {
            background: linear-gradient(135deg, #232323 0%, #2c2c2c 100%) !important;
            border-left: 4px solid #b8860b !important;
            color: #f0e6d6 !important;
        }
        .assistant-message {
            background: linear-gradient(135deg, #4d392a 0%, #3b301e 100%) !important;
            border-left: 4px solid #e6b800 !important;
            color: #f0e6d6 !important;
        }
        .message-header {
            color: #ffe082 !important;
        }
        .stApp > header {
            background-color: transparent;
        }
        .stApp {
            background: linear-gradient(180deg, #232323 0%, #2c2c2c 100%) !important;
        }
        .stSidebar,
        .stSidebar > div,
        .stSidebar > div > div {
            background: linear-gradient(180deg, #232323 0%, #2c2c2c 100%) !important;
        }
        .stButton > button {
            background: linear-gradient(135deg, #b8860b 0%, #e6b800 100%) !important;
            color: #232323 !important;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #e6b800 0%, #b8860b 100%) !important;
        }
        div[data-testid="stChatInput"] {
            background: linear-gradient(135deg, #232323 0%, #2c2c2c 100%) !important;
            border: 1px solid #b8860b !important;
        }
        div[data-testid="stChatInput"] textarea {
            color: #ffe082 !important;
        }
        div[data-testid="stChatInput"] button {
            background: linear-gradient(135deg, #b8860b 0%, #e6b800 100%) !important;
            color: #232323 !important;
        }
        div[data-testid="stChatInput"] button:hover {
            background: linear-gradient(135deg, #e6b800 0%, #b8860b 100%) !important;
        }
        div[data-testid="stChatInput"] button svg {
            fill: #232323 !important;
        }
        .main > div,
        [data-testid="stAppViewBlockContainer"],
        section[data-testid="stChatFlowMessage"],
        .stBottom,
        .stBottom > div {
            background-color: transparent !important;
            background: transparent !important;
        }
    }
</style>
""", unsafe_allow_html=True)


def format_timestamp_to_kst(timestamp_str):
    """UTC 타임스탬프를 Asia/Seoul 타임존으로 변환하여 포맷팅"""
    if not timestamp_str:
        return ""
    
    try:
        # SQLite timestamp 형식 파싱 (YYYY-MM-DD HH:MM:SS 형태)
        dt = datetime.fromisoformat(timestamp_str.replace(' ', 'T'))
        
        # UTC로 간주하고 Asia/Seoul로 변환
        utc_dt = dt.replace(tzinfo=pytz.UTC)
        seoul_tz = pytz.timezone('Asia/Seoul')
        seoul_dt = utc_dt.astimezone(seoul_tz)
        
        # 포맷팅 (월/일 시:분)
        return seoul_dt.strftime("%m/%d %H:%M")
    except:
        return timestamp_str


@st.cache_resource
def initialize_system():
    """시스템 초기화 (캐시됨) - 공통 모듈 사용 (config 기반)"""
    result = RAGSystemInitializer.initialize_system(
        current_file_path=script_dir,
        include_sql=True,
        use_config=True,  # config.yaml 사용
        logger_name="StreamlitRAG",
        enable_db_memory=False  # WebUI에서는 개별 세션별로 ChatHistoryManager 관리
    )
    
    if result is None:
        st.error("시스템 초기화에 실패했습니다.")
        return None, None, None, None, None, None
    
    # 라우터는 RAGQueryProcessor 내부에 포함됨
    return result


def initialize_session_state():
    """세션 상태 초기화"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None
    
    if "chat_history_manager" not in st.session_state:
        st.session_state.chat_history_manager = None
    
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = True


def create_new_conversation(sql_manager):
    """새 대화 생성"""
    try:
        # 새 세션 생성
        session_id = sql_manager.create_conversation()
        
        # 채팅 히스토리 관리자 초기화 (설정의 memory_window 적용)
        try:
            config_loader = get_config_loader()
            db_conf = config_loader.get_database_config()
            memory_k = db_conf.get("memory_window", 3)
        except Exception:
            memory_k = 3

        chat_manager = ChatHistoryManager(
            session_id=session_id,
            sql_manager=sql_manager,
            memory_k=memory_k
        )
        
        # 세션 상태 업데이트
        st.session_state.current_session_id = session_id
        st.session_state.chat_history_manager = chat_manager
        st.session_state.messages = []
        
        st.success("새로운 제빵 상담이 시작되었습니다!")
        st.rerun()
        
    except Exception as e:
        st.error(f"새 대화 생성 오류: {str(e)}")


def load_conversation(session_id, sql_manager):
    """기존 대화 로드"""
    try:
        # 채팅 히스토리 관리자 초기화 (설정의 memory_window 적용)
        try:
            config_loader = get_config_loader()
            db_conf = config_loader.get_database_config()
            memory_k = db_conf.get("memory_window", 3)
        except Exception:
            memory_k = 3

        chat_manager = ChatHistoryManager(
            session_id=session_id,
            sql_manager=sql_manager,
            memory_k=memory_k
        )
        
        # 전체 대화 기록 가져오기
        messages = chat_manager.get_full_conversation_history()
        
        # 세션 상태 업데이트
        st.session_state.current_session_id = session_id
        st.session_state.chat_history_manager = chat_manager
        st.session_state.messages = messages
        
        st.success(f"제빵 상담 기록을 불러왔습니다! ({len(messages)}개 메시지)")
        st.rerun()
        
    except Exception as e:
        st.error(f"대화 로드 오류: {str(e)}")


def delete_conversation_by_id(session_id, sql_manager):
    """대화 삭제"""
    try:
        # 현재 활성 대화인지 확인
        is_current = (st.session_state.current_session_id == session_id)
        
        # 데이터베이스에서 대화 삭제
        success = sql_manager.delete_conversation(session_id)
        
        if success:
            # 현재 활성 대화였다면 세션 상태 초기화
            if is_current:
                st.session_state.current_session_id = None
                st.session_state.chat_history_manager = None
                st.session_state.messages = []
            
            st.success("상담 기록이 삭제되었습니다!")
            st.rerun()
        else:
            st.error("상담 기록 삭제에 실패했습니다.")
            
    except Exception as e:
        st.error(f"상담 기록 삭제 오류: {str(e)}")


def render_sidebar(sql_manager):
    """사이드바 렌더링"""
    with st.sidebar:
        st.header("🥖 제빵 상담 관리")
        
        # 새 대화 버튼
        if st.button("🆕 새 제빵 상담 시작", use_container_width=True):
            create_new_conversation(sql_manager)
        
        st.divider()
        
        # 대화 목록
        st.subheader("📝 상담 기록")
        
        try:
            conversations = sql_manager.get_conversations(limit=20)
            
            if conversations:
                for conv in conversations:
                    session_id = conv["session_id"]
                    title = conv["title"]
                    updated_at = conv["updated_at"]
                    message_count = conv["message_count"]
                    
                    # 현재 선택된 대화 표시
                    is_current = (st.session_state.current_session_id == session_id)
                    button_label = f"{'🔵' if is_current else '⚪'} {title}"
                    
                    # 대화 제목과 삭제 버튼을 나란히 배치
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        if st.button(
                            button_label, 
                            key=f"conv_{session_id}",
                            help=f"메시지: {message_count}개, 업데이트: {format_timestamp_to_kst(updated_at)}",
                            use_container_width=True
                        ):
                            if not is_current:
                                load_conversation(session_id, sql_manager)
                    
                    with col2:
                        # 삭제 확인 상태 관리
                        confirm_key = f"confirm_delete_{session_id}"
                        if confirm_key not in st.session_state:
                            st.session_state[confirm_key] = False
                        
                        # 삭제 버튼 또는 확인 버튼 표시
                        if not st.session_state[confirm_key]:
                            if st.button(
                                "🗑️", 
                                key=f"delete_{session_id}",
                                help="상담 기록 삭제",
                                use_container_width=True
                            ):
                                st.session_state[confirm_key] = True
                                st.rerun()
                        else:
                            # 확인/취소 버튼을 위한 하위 컬럼
                            subcol1, subcol2 = st.columns(2)
                            
                            with subcol1:
                                if st.button(
                                    "✓", 
                                    key=f"confirm_{session_id}",
                                    help="삭제 확인",
                                    use_container_width=True,
                                    type="primary"
                                ):
                                    st.session_state[confirm_key] = False
                                    delete_conversation_by_id(session_id, sql_manager)
                            
                            with subcol2:
                                if st.button(
                                    "✕", 
                                    key=f"cancel_{session_id}",
                                    help="삭제 취소",
                                    use_container_width=True
                                ):
                                    st.session_state[confirm_key] = False
                                    st.rerun()
            else:
                st.info("저장된 상담 기록이 없습니다.")
                
        except Exception as e:
            st.error(f"상담 기록 목록 로드 오류: {str(e)}")
        
        st.divider()
        
        # 설정 패널
        st.subheader("⚙️ 설정")
        
        # 소스 표시 토글
        st.checkbox(
            "제빵 자료 출처 표시", 
            key="show_sources"
        )
        
        # 시스템 정보
        with st.expander("ℹ️ 시스템 정보"):
            if st.session_state.current_session_id:
                st.write(f"**현재 상담 세션:** {st.session_state.current_session_id[:8]}...")
            
            if st.session_state.chat_history_manager:
                summary = st.session_state.chat_history_manager.get_conversation_summary()
                st.write(f"**전체 메시지:** {summary.get('total_messages', 0)}개")
                st.write(f"**메모리 메시지:** {summary.get('memory_messages', 0)}개")


def render_chat_interface(query_processor):
    """채팅 인터페이스 렌더링"""
    st.header("🍞 제과제빵 상담 어시스턴트")
    
    # 세션이 없으면 안내 메시지
    if not st.session_state.current_session_id:
        st.info("👈 사이드바에서 '새 제빵 상담 시작'을 클릭하여 상담을 시작하세요.")
        return
    
    # 메시지 표시 영역
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            role = message["role"]
            content = message["content"]
            timestamp = message.get("timestamp", "")
            formatted_timestamp = format_timestamp_to_kst(timestamp)
            metadata = message.get("metadata", {})
            
            if role == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <div class="message-header">👨‍🍳 제빵사 {formatted_timestamp}</div>
                    <div>{content}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <div class="message-header">🍞 빵지니 {formatted_timestamp}</div>
                    <div>{content}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # 소스 정보 표시 (설정이 켜져 있고 소스가 있는 경우)
                if st.session_state.show_sources and metadata and metadata.get("sources"):
                    sources = metadata["sources"]
                    with st.expander(f"📚 참조 제빵 자료 ({len(sources)}개)", expanded=False):
                        for source in sources:
                            st.write(f"• {source}")
    
    # 사용자 입력
    user_input = st.chat_input("제빵에 관한 질문을 입력하세요...")
    
    if user_input and st.session_state.chat_history_manager:
        # 사용자 메시지 추가
        st.session_state.chat_history_manager.add_user_message(user_input)
        
        # 통합 질의 처리 (라우팅 + 분기 + 메모리)
        with st.spinner("질문을 분석하고 답변을 준비하는 중..."):
            try:
                result = query_processor.unified_query(
                    question=user_input,
                    chat_history_manager=st.session_state.chat_history_manager,
                    auto_save=True,
                    return_sources=True
                )
                
                if result["success"]:
                    # UI 메시지 리스트 업데이트
                    st.session_state.messages = st.session_state.chat_history_manager.get_full_conversation_history()
                    st.rerun()
                else:
                    st.error(f"답변 생성 오류: {result.get('error', '알 수 없는 오류')}")
                
            except Exception as e:
                st.error(f"답변 생성 오류: {str(e)}")


def main():
    """메인 함수"""
    # 세션 상태 초기화
    initialize_session_state()
    
    # 시스템 초기화
    result = initialize_system()
    if not result:
        return
    
    vector_manager, llm_manager, retriever_manager, sql_manager, query_processor = result
    
    # UI 렌더링
    render_sidebar(sql_manager)
    render_chat_interface(query_processor)


if __name__ == "__main__":
    main()