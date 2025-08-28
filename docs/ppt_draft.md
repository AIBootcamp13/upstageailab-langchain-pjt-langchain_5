### Slide 0

- 제과제빵 문서 기반 RAG 시스템 (LangChain + Upstage) | 2025. 08. 29(금)

### Slide 1

- 1
- 목차
  - 01. 팀 소개
  - 02. 프로젝트 개요
  - 03. 프로젝트 수행 절차 및 방법
  - 04. 회고
  - 목표 수립 / 기술 스택 & 아키텍처 설계
  - 결과 / 인사이트 도출 / 향후 계획 / 느낀점
  - 작품 소개 / 구현 전략 / 기능 리뷰 / 트러블 슈팅

### Slide 2

- 2
- 팀 소개
  - 팀장/팀원 소개
  - 협업 방식

### Slide 3

- 팀 각오: 제과제빵 지식의 검색·생성을 결합한 실무형 RAG 완성
- 팀장
  - 류지헌 (https://github.com/mahomi)
  - 역할: RAG 아키텍처 설계, LangChain 파이프라인 구현
- 팀원
  - 김태현 (https://github.com/huefilm)
  - 역할: 문서 전처리/분할, PDF 로더 최적화
- 팀원
  - 박진섭 (https://github.com/seob1504)
  - 역할: 임베딩/벡터 저장소, FAISS 성능 튜닝
- 팀원
  - 문진숙 (https://github.com/June3723)
  - 역할: 프롬프트 엔지니어링, 답변 품질 개선
- 팀원
  - 김재덕 (https://github.com/ttcoaster)
  - 역할: API 통합/배포, 환경 설정 관리
- 팀원 소개

### Slide 4

- 프로젝트 협업 방식
  - GitHub 이슈/PR 중심 태스크 관리, Slack 알림 연동
  - UV 기반 통일 실행환경으로 온보딩/재현성 확보
  - 주 2회 정기 싱크 + 데일리 비동기 업데이트
  - 모듈 단위 코드 소유권 + 상호 코드리뷰
  - 문제 발생 시 로그/테스트 우선 분석, 재현 스크립트 공유
- 협업 방식

### Slide 5

- 02
- 프로젝트 개요
  - 목표 수립
  - 작품 소개 / 기술 스택

### Slide 6

- 프로젝트 목표 수립
- 제과제빵 문서 기반 한국어 RAG 어시스턴트 구축
- 기간
  - 2025. 08. 19 ~ 2025. 08. 29
- 개요
  - PDF 로드 → 텍스트 분할 → 임베딩 → 벡터 검색 → 답변 생성
- 목표
  - 제빵 전문 주제(레시피/공정/위생) 질의응답과 근거 문서 제공
- 주요 작업
  - Upstage Chat/Embeddings + FAISS 파이프라인 구현 및 튜닝
  - Streamlit WebUI 제작(제빵 테마), 설정 패널 연동
  - RAGAS 기반 자동 평가 및 결과 저장

### Slide 7

- 기술 스택 & 아키텍처 설계
- 기술 스택
  - Python 3.10+, UV, Streamlit
  - LangChain (Core/Community/OpenAI/Upstage), FAISS, PyMuPDF
  - RAGAS, pytest, SQLite, python-dotenv
  - API: Upstage AI (Chat, Embeddings)
- 아키텍처 요약
  - QueryRouter로 제빵 관련/일반 지식 라우팅
  - 제빵 관련: RAG(FAISS Retriever + 프롬프트 + Upstage Chat)
  - 일반 지식: LLM 단독 응답

### Slide 8

- 프로젝트 수행 절차 및 방법
  - 작품 소개 / 구현 전략 / 기능 리뷰 / 트러블 슈팅

### Slide 9

- 작품 소개
- 제과제빵 분야 문서(RAG)로 정확하고 근거 있는 Q&A 제공
- Streamlit WebUI에서 대화형 인터페이스와 출처 표시 지원
- 운영을 위한 로깅/테스트/설정관리 포함

### Slide 10

- 구현 전략
- 데이터 처리: PyMuPDF 추출 → RecursiveCharacterTextSplitter (chunk_size 1000, overlap 50)
- 검색/생성: Upstage Embeddings, FAISS 검색, 제빵 전문가 프롬프트 + Upstage Chat(solar-pro2)
- 라우팅: QueryRouter가 RAG/일반응답 분기
- 품질관리: RAGAS로 faithfulness/relevancy/recall/correctness 측정

### Slide 11

- 기능 리뷰
  - WebUI: 채팅, 대화 히스토리, 설정 패널, 출처 표시
  - VectorStoreManager: FAISS 인덱스 관리/증분 업데이트
  - LLMManager/PromptLoader: 모델·프롬프트 일원화 관리
  - QueryRouter: 제빵/일반 라우팅으로 비용·품질 최적화
  - Evaluate: RAGAS 자동 평가 및 JSON 결과 저장

### Slide 12

- 트러블 슈팅
  - PDF 포맷 다양성으로 추출/분할 품질 편차 → 전처리 규칙·파라미터 기준 수립
  - FAISS 검색 품질 편차 → 임베딩/인덱스 튜닝, 증분 업데이트 플로우 도입
  - 프롬프트에 따른 사실성 편차 → 근거 강조 템플릿, 컨텍스트 구성 규칙 정리
  - 환경 차이로 실행 오류 → UV/환경변수 템플릿, README 온보딩 강화

### Slide 13

- 회고
  - 결과 / 인사이트 도출 / 향후 계획
  - 느낀점

### Slide 14

- 결과 및 향후 계획
- RAGAS 결과 요약 (모델 조합별 ragas_score)
  - upstage embedding-query + solar-pro2(reasoning): 0.469
  - gemini-embedding-001 + solar-pro2(reasoning): 0.625
  - gemini-embedding-001 + gemini-2.5-flash: 0.640
- 인사이트: 임베딩/LLM 조합과 분할·컨텍스트 규칙이 품질에 큰 영향
- 향후: 하이브리드 검색 검토, 임베딩/LLM 고도화, 평가 루프 자동화

### Slide 15

- 프로젝트 진행 소감
  - 류지헌: 검색·생성 균형과 운영 관점 로깅/테스트 중요성 확인
  - 김태현: 로더/분할 전략이 검색 성능 핵심, 기준 수립의 의미
  - 박진섭: 임베딩·FAISS 파라미터 영향도 정리, 인덱스 운영 플로우 확립
  - 문진숙: 프롬프트로 일관성·사실성 개선, 실패 사례의 빠른 학습 효과
  - 김재덕: 환경 차이 버그 최소화, 재현 가능한 실행 환경·온보딩 문서화
