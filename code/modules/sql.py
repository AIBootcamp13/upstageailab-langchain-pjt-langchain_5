"""
SQLite 데이터베이스 관리 모듈

이 모듈은 채팅 대화 내용을 SQLite 데이터베이스에 저장하고 관리하는 기능을 제공합니다.

주요 기능:
1. 데이터베이스 연결 및 테이블 생성
2. 대화 세션 관리 (conversations 테이블)
3. 메시지 저장 및 조회 (messages 테이블)
4. CRUD 기능 제공

테이블 구조:
- conversations: id, session_id, created_at, updated_at, title
- messages: id, conversation_id, role, content, timestamp, metadata
"""

import sqlite3
import json
import uuid
import pytz
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class SQLManager:
    """SQLite 데이터베이스 관리 클래스"""
    
    def __init__(self, db_path: str = "../data/chat.db"):
        """
        SQLManager 초기화
        
        Args:
            db_path (str): 데이터베이스 파일 경로
        """
        self.db_path = db_path
        
        # 데이터베이스 디렉토리 생성
        db_dir = Path(db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        
        # 데이터베이스 초기화
        self._init_database()
    
    def _init_database(self):
        """데이터베이스 테이블 생성"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # conversations 테이블 생성
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # messages 테이블 생성
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (id)
                )
            """)
            
            # 인덱스 생성
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_conversation_id 
                ON messages (conversation_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_timestamp 
                ON messages (timestamp)
            """)
            
            conn.commit()
    
    def create_conversation(self, title: str = None) -> str:
        """
        새로운 대화 세션 생성
        
        Args:
            title (str, optional): 대화 제목. None이면 자동 생성
            
        Returns:
            str: 생성된 session_id
        """
        session_id = str(uuid.uuid4())
        
        if title is None:
            title = "새 질문"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO conversations (session_id, title)
                VALUES (?, ?)
            """, (session_id, title))
            conn.commit()
        
        return session_id
    
    def get_conversation_id(self, session_id: str) -> Optional[int]:
        """
        session_id로 conversation_id 조회
        
        Args:
            session_id (str): 세션 ID
            
        Returns:
            Optional[int]: conversation_id 또는 None
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id FROM conversations WHERE session_id = ?
            """, (session_id,))
            result = cursor.fetchone()
            return result[0] if result else None
    
    def add_message(self, session_id: str, role: str, content: str, metadata: Dict = None) -> int:
        """
        메시지 추가
        
        Args:
            session_id (str): 세션 ID
            role (str): 메시지 역할 ('user' 또는 'assistant')
            content (str): 메시지 내용
            metadata (Dict, optional): 추가 메타데이터
            
        Returns:
            int: 추가된 메시지 ID
        """
        conversation_id = self.get_conversation_id(session_id)
        if conversation_id is None:
            raise ValueError(f"세션을 찾을 수 없습니다: {session_id}")
        
        metadata_json = json.dumps(metadata) if metadata else None
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO messages (conversation_id, role, content, metadata)
                VALUES (?, ?, ?, ?)
            """, (conversation_id, role, content, metadata_json))
            
            # conversations 테이블의 updated_at 업데이트
            cursor.execute("""
                UPDATE conversations 
                SET updated_at = CURRENT_TIMESTAMP 
                WHERE id = ?
            """, (conversation_id,))
            
            conn.commit()
            return cursor.lastrowid
    
    def get_messages(self, session_id: str, limit: int = None) -> List[Dict]:
        """
        세션의 메시지 목록 조회
        
        Args:
            session_id (str): 세션 ID
            limit (int, optional): 조회할 메시지 수 제한
            
        Returns:
            List[Dict]: 메시지 목록
        """
        conversation_id = self.get_conversation_id(session_id)
        if conversation_id is None:
            return []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT id, role, content, timestamp, metadata
                FROM messages 
                WHERE conversation_id = ?
                ORDER BY timestamp ASC
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query, (conversation_id,))
            rows = cursor.fetchall()
            
            messages = []
            for row in rows:
                message = {
                    "id": row[0],
                    "role": row[1],
                    "content": row[2],
                    "timestamp": row[3],
                    "metadata": json.loads(row[4]) if row[4] else None
                }
                messages.append(message)
            
            return messages
    
    def get_conversations(self, limit: int = 50) -> List[Dict]:
        """
        대화 목록 조회 (최신순)
        
        Args:
            limit (int): 조회할 대화 수 제한
            
        Returns:
            List[Dict]: 대화 목록
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    c.session_id,
                    c.title,
                    c.created_at,
                    c.updated_at,
                    (SELECT COUNT(*) FROM messages WHERE conversation_id = c.id) as message_count,
                    (
                        SELECT content
                        FROM messages
                        WHERE conversation_id = c.id AND role = 'user'
                        ORDER BY timestamp ASC
                        LIMIT 1
                    ) as first_user_message
                FROM conversations c
                ORDER BY c.updated_at DESC
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            conversations = []
            for row in rows:
                conversation = {
                    "session_id": row[0],
                    "title": row[1],
                    "created_at": row[2],
                    "updated_at": row[3],
                    "message_count": row[4],
                    "first_user_message": row[5]
                }
                conversations.append(conversation)
            
            return conversations
    
    def update_conversation_title(self, session_id: str, title: str) -> bool:
        """
        대화 제목 업데이트
        
        Args:
            session_id (str): 세션 ID
            title (str): 새 제목
            
        Returns:
            bool: 성공 여부
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE conversations 
                SET title = ?, updated_at = CURRENT_TIMESTAMP
                WHERE session_id = ?
            """, (title, session_id))
            conn.commit()
            return cursor.rowcount > 0
    
    def delete_conversation(self, session_id: str) -> bool:
        """
        대화 삭제 (메시지 포함)
        
        Args:
            session_id (str): 세션 ID
            
        Returns:
            bool: 성공 여부
        """
        conversation_id = self.get_conversation_id(session_id)
        if conversation_id is None:
            return False
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 메시지 먼저 삭제
            cursor.execute("""
                DELETE FROM messages WHERE conversation_id = ?
            """, (conversation_id,))
            
            # 대화 삭제
            cursor.execute("""
                DELETE FROM conversations WHERE id = ?
            """, (conversation_id,))
            
            conn.commit()
            return cursor.rowcount > 0
    
    def get_recent_messages(self, session_id: str, count: int = 10) -> List[Tuple[str, str]]:
        """
        최근 메시지를 (role, content) 튜플 리스트로 반환
        
        Args:
            session_id (str): 세션 ID
            count (int): 조회할 메시지 수
            
        Returns:
            List[Tuple[str, str]]: (role, content) 튜플 리스트
        """
        messages = self.get_messages(session_id)
        recent_messages = messages[-count:] if len(messages) > count else messages
        return [(msg["role"], msg["content"]) for msg in recent_messages]
    
    def close(self):
        """데이터베이스 연결 종료 (현재는 자동 관리되므로 필요 없음)"""
        pass