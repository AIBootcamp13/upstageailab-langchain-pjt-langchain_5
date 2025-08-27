@main.py
webui 사용자가 질문을 하면 제빵 관련 질문일 경우, 자료에서 RAG를 하고,
그렇지 않으면 일반 llm답변을 하도록 라우팅 기능을 추가해줘.
라우팅은 model="upstage/solar-1-mini-chat" 모델을 사용하고
RAG 시에는 model="solar-pro2" 를 사용하도록 해줘.
