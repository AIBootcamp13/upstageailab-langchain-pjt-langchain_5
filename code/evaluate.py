"""
RAG 시스템 품질 평가 도구 (evaluate.py)

WebUI와 동일한 라우터 및 메모리 방식을 사용하여
RAGAS 프레임워크로 답변 품질을 평가하는 도구입니다.

주요 기능:
- WebUI와 동일한 시스템 초기화 (VectorStoreManager, LLMManager, RetrieverManager, ChatHistoryManager, QueryRouter)
- 라우터를 통한 질문 분류 (일반답변 vs RAG답변)
- 메모리 기능을 포함한 대화 처리 (DB 저장 없음)
- 사전 정의된 질문-정답 데이터셋 사용 (data/eval/question_dataset.json)
- RAGAS 메트릭을 통한 품질 평가 (faithfulness, answer_relevancy, context_recall, answer_correctness)
- 평가 결과 저장 및 리포트 생성 (data/eval/evaluation_results/)
- Upstage API 직접 사용으로 RAGAS 호환성 확보

사용법:
    uv run python code/evaluate.py

"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

# 현재 스크립트의 디렉토리를 sys.path에 추가
script_dir = Path(__file__).parent.absolute()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

# 환경변수 로드
from dotenv import load_dotenv
load_dotenv(script_dir / '.env')

# LangChain 및 Upstage imports
from langchain_upstage import UpstageEmbeddings

# RAGAS imports (Upstage API 사용)
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy, 
    context_recall,
    answer_correctness
)
from datasets import Dataset

# 모듈 imports
from modules import (
    VectorStoreManager, LLMManager, RetrieverManager, 
    ChatHistoryManager, LoggerManager, RAGSystemInitializer
)
from modules.config_loader import get_config_loader


# 전역 설정
project_root = script_dir.parent
dataset_path = project_root / "data" / "eval" / "question_dataset.json"
results_dir = project_root / "data" / "eval" / "evaluation_results"

def setup_upstage_for_ragas():
    """RAGAS에서 Upstage API를 사용하도록 환경변수 설정"""
    upstage_api_key = os.getenv("UPSTAGE_API_KEY")
    if upstage_api_key:
        os.environ["OPENAI_API_KEY"] = upstage_api_key
        os.environ["OPENAI_BASE_URL"] = "https://api.upstage.ai/v1"
        os.environ["OPENAI_MODEL_NAME"] = "solar-pro2"




class RAGEvaluator:
    """RAG 시스템 품질 평가를 위한 클래스"""
    
    def __init__(self):
        self.logger = LoggerManager("RAGEvaluator")
        self.results_dir = results_dir
        
        # 결과 디렉토리 생성
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def load_evaluation_dataset(self) -> Dict[str, Any]:
        """평가 데이터셋 로드"""
        with open(dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def initialize_system(self):
        """RAG 시스템 초기화 (라우터 포함)"""
        result = RAGSystemInitializer.initialize_system(
            current_file_path=script_dir,
            include_sql=False,
            use_config=True,  # config.yaml 사용
            logger_name="EvaluationRAG",
            enable_db_memory=False  # 평가용은 메모리만 사용
        )
        
        if result is None:
            return False
        
        self.vector_manager, self.llm_manager, self.retriever_manager, self.query_processor = result
        
        # 라우터는 query_processor 내부에 포함됨
        return True
    
    def process_questions(self, dataset: Dict[str, Any]) -> List[Dict[str, Any]]:
        """질문들을 라우터를 통해 처리하고 답변 생성"""
        results = []
        questions = dataset["questions"]
        
        # 메모리 관리자 초기화 (DB 저장 없음, 메모리만 사용) - 모든 질문에 대해 동일한 인스턴스 사용
        try:
            config_loader = get_config_loader()
            db_conf = config_loader.get_database_config()
            memory_k = db_conf.get("memory_window", 3)
        except Exception:
            memory_k = 3

        # 모든 질문에 대해 동일한 ChatHistoryManager 인스턴스 사용 (메모리 연속성 확보)
        chat_manager = ChatHistoryManager(
            session_id="evaluation_session",  # 고정된 세션 ID 사용
            memory_k=memory_k,
            sql_manager=None,  # DB 저장 안함
            auto_save=True     # 메모리에 저장 활성화
        )
        
        for i, question_data in enumerate(questions, 1):
            start_time = time.time()
            
            try:
                question = question_data["question"]
                
                # 통합 질의 처리 (라우팅 + 분기 + 메모리) - 같은 chat_manager 사용
                result = self.query_processor.unified_query(
                    question=question,
                    chat_history_manager=chat_manager,  # 같은 인스턴스 사용
                    auto_save=True,  # 메모리에 저장 활성화
                    return_sources=True
                )
                
                if result["success"]:
                    response = result["response"]
                    processing_type = result["processing_type"]
                    routing_result = result["routing_info"]
                    sources = result.get("sources", [])
                    documents = result.get("documents", [])
                    
                    # 처리 시간 계산
                    processing_time = (time.time() - start_time) * 1000  # ms
                    
                    # 간결한 로그 출력으로 질문과 답변 내용 확인
                    self.logger.info(f"\n{'='*60}")
                    self.logger.info(f"📝 질문 {i}: {question}")
                    self.logger.info(f"🏷️  카테고리: {question_data['category']} | 처리타입: {processing_type}")
                    
                    # 답변 내용 출력 (긴 답변은 요약)
                    if len(response) > 300:
                        response_preview = response[:300] + "..."
                        self.logger.info(f"💬 답변 (요약): {response_preview}")
                        self.logger.info(f"📄 전체 답변 길이: {len(response)}자")
                    else:
                        self.logger.info(f"💬 답변: {response}")
                    
                    # RAG 관련 정보 출력
                    if routing_result["use_rag"] and documents:
                        self.logger.info(f"🔍 검색된 문서: {len(documents)}개")
                        if sources:
                            self.logger.info(f"📚 소스: {', '.join(sources[:3])}")
                    else:
                        self.logger.info("💭 일반 답변 (문서 검색 없음)")
                    
                    self.logger.info(f"⏱️  처리시간: {round(processing_time, 0)}ms")
                    self.logger.info(f"{'='*60}")
                    
                else:
                    raise Exception(result["error"])
                
                # 결과 저장
                results.append({
                    "question_id": question_data["id"],
                    "question": question,
                    "generated_answer": response,
                    "ground_truth": question_data["ground_truth"],
                    "category": question_data["category"],
                    "difficulty": question_data["difficulty"],
                    "processing_type": processing_type,
                    "routing_confidence": routing_result.get("confidence", 0.0),
                    "retrieved_contexts": [doc.page_content for doc in documents] if routing_result["use_rag"] and documents else [],
                    "source_documents": sources if routing_result["use_rag"] else [],
                    "expected_sources": question_data.get("expected_sources", []),
                    "processing_time_ms": round(processing_time, 2),
                    "depends_on": question_data.get("depends_on"),
                    "keywords": question_data.get("keywords", [])
                })
                
            except Exception as e:
                # 실패한 경우에도 빈 결과 추가
                results.append({
                    "question_id": question_data["id"],
                    "question": question_data["question"],
                    "generated_answer": f"오류 발생: {str(e)}",
                    "ground_truth": question_data["ground_truth"],
                    "category": question_data["category"],
                    "difficulty": question_data["difficulty"],
                    "processing_type": "ERROR",
                    "routing_confidence": 0.0,
                    "retrieved_contexts": [],
                    "source_documents": [],
                    "expected_sources": question_data.get("expected_sources", []),
                    "processing_time_ms": 0,
                    "error": str(e)
                })
        
        # 메모리 상태 확인 및 로깅
        try:
            summary = chat_manager.get_conversation_summary()
            self.logger.log_step("메모리 상태 확인", 
                               f"전체 메시지: {summary.get('total_messages', 0)}개, "
                               f"메모리 메시지: {summary.get('memory_messages', 0)}개")
            
            # 실제 메모리 내용 확인
            memory_messages = chat_manager.memory.chat_memory.messages
            self.logger.log_step("메모리 내용 확인", 
                               f"메모리에 저장된 메시지 수: {len(memory_messages)}개")
            
        except Exception as e:
            self.logger.log_error("메모리 상태 확인", e)
        
        return results
    
    def run_ragas_evaluation(self, results: List[Dict[str, Any]]) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """RAGAS 메트릭을 사용한 평가 실행 (Upstage API 사용) - 전체 점수와 개별 질문별 점수 반환"""
        self.logger.log_function_start("run_ragas_evaluation")
        
        try:
            # RAGAS 평가를 위한 데이터셋 구성
            dataset_dict = {
                "question": [r["question"] for r in results],
                "answer": [r["generated_answer"] for r in results],
                "contexts": [r["retrieved_contexts"] for r in results],
                "ground_truth": [r["ground_truth"] for r in results]
            }
            
            # Dataset 객체 생성
            dataset = Dataset.from_dict(dataset_dict)
            
            self.logger.log_step("RAGAS 평가 실행", "메트릭: faithfulness, answer_relevancy, context_recall, answer_correctness (LangChain rate limit 설정 활용)")
            
            # baseline.py 방식으로 Upstage 모델 직접 사용
            from langchain_upstage import ChatUpstage, UpstageEmbeddings
            
            # LangChain rate limit 방지 설정을 포함한 모델 초기화
            upstage_llm = ChatUpstage(
                api_key=os.getenv("UPSTAGE_API_KEY"),
                model="solar-pro2",
                reasoning_effort="low",
                # Rate limit 방지 설정
                request_timeout=300,  # 요청 타임아웃 120초 (증가)
                max_retries=10       # 최대 재시도 10회 (증가)
            )
            
            # RAGAS 평가 실행 (전체 데이터셋을 한 번에 처리, LangChain rate limit 설정 활용)
            evaluation_result = evaluate(
                dataset=dataset,
                metrics=[faithfulness, answer_relevancy, context_recall, answer_correctness],
                llm=upstage_llm,
                embeddings=RAGSystemInitializer.initialize_embeddings()
            )
            
            # 결과를 딕셔너리로 변환
            scores = {}
            scores_dict = evaluation_result._scores_dict
            
            # 개별 질문별 점수 추출
            individual_scores = []
            for i in range(len(results)):
                question_scores = {}
                for metric_name in ["faithfulness", "answer_relevancy", "context_recall", "answer_correctness"]:
                    if metric_name in scores_dict and len(scores_dict[metric_name]) > i:
                        score_value = scores_dict[metric_name][i]
                        # numpy scalar이나 float 처리
                        if hasattr(score_value, 'item'):
                            question_scores[metric_name] = float(score_value.item())
                        else:
                            question_scores[metric_name] = float(score_value)
                    else:
                        question_scores[metric_name] = 0.0
                
                # 개별 질문의 RAGAS 점수 계산
                question_scores["ragas_score"] = sum(question_scores.values()) / len(question_scores)
                individual_scores.append(question_scores)
            
            # 전체 평균 점수 계산
            for metric_name in ["faithfulness", "answer_relevancy", "context_recall", "answer_correctness"]:
                if metric_name in scores_dict and len(scores_dict[metric_name]) > 0:
                    # 리스트의 평균값 계산
                    score_values = scores_dict[metric_name]
                    if isinstance(score_values, list):
                        # numpy scalar이나 float 처리
                        avg_score = sum(float(v.item()) if hasattr(v, 'item') else float(v) for v in score_values) / len(score_values)
                        scores[metric_name] = avg_score
                    else:
                        scores[metric_name] = float(score_values.item()) if hasattr(score_values, 'item') else float(score_values)
                else:
                    scores[metric_name] = 0.0
            
            # RAGAS 종합 점수 계산 (평균)
            scores["ragas_score"] = sum(scores.values()) / len(scores)
            
            self.logger.log_function_end("run_ragas_evaluation", f"평가 완료: 전체 {scores['ragas_score']:.3f}, 개별 {len(individual_scores)}개 질문")
            return scores, individual_scores
            
        except Exception as e:
            import traceback
            self.logger.error(f"RAGAS 평가 중 심각한 오류 발생: {e}")
            self.logger.error(f"오류 타입: {type(e)}")
            self.logger.error(f"상세 트레이스백: {traceback.format_exc()}")
            # 평가 실패 시 기본값 반환
            default_scores = {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_recall": 0.0,
                "answer_correctness": 0.0,
                "ragas_score": 0.0,
                "error": str(e)
            }
            
            # 개별 점수도 기본값으로 생성
            default_individual_scores = []
            for _ in results:
                default_individual_scores.append(default_scores.copy())
            
            return default_scores, default_individual_scores
    
    def calculate_category_scores(self, results: List[Dict[str, Any]], 
                                overall_scores: Dict[str, float]) -> Dict[str, Any]:
        """카테고리별 점수 계산"""
        self.logger.log_function_start("calculate_category_scores")
        
        categories = {}
        
        for result in results:
            category = result["category"]
            if category not in categories:
                categories[category] = {
                    "questions": [],
                    "question_count": 0,
                    "rag_count": 0,
                    "general_count": 0
                }
            
            categories[category]["questions"].append(result)
            categories[category]["question_count"] += 1
            
            # 라우팅 타입별 카운트
            if result.get("processing_type") == "RAG":
                categories[category]["rag_count"] += 1
            elif result.get("processing_type") == "GENERAL":
                categories[category]["general_count"] += 1
        
        # 각 카테고리별 점수 (전체 점수 기반 추정)
        category_scores = {}
        for category, data in categories.items():
            category_scores[category] = {
                "question_count": data["question_count"],
                "rag_count": data["rag_count"],
                "general_count": data["general_count"],
                "avg_processing_time_ms": sum(q.get("processing_time_ms", 0) 
                                            for q in data["questions"]) / data["question_count"],
                "success_rate": len([q for q in data["questions"] 
                                   if not q.get("error")]) / data["question_count"]
            }
        
        self.logger.log_function_end("calculate_category_scores", 
                                   f"{len(categories)}개 카테고리 분석")
        return category_scores
    
    def save_evaluation_results(self, dataset: Dict[str, Any], results: List[Dict[str, Any]], 
                              overall_scores: Dict[str, float], category_scores: Dict[str, Any],
                              individual_scores: List[Dict[str, float]]) -> str:
        """평가 결과 저장"""
        self.logger.log_function_start("save_evaluation_results")
        
        try:
            # 타임스탬프 생성
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"{timestamp}.json"
            filepath = self.results_dir / filename
            
            # 라우팅 통계 계산
            total_questions = len(results)
            rag_questions = len([r for r in results if r.get("processing_type") == "RAG"])
            general_questions = len([r for r in results if r.get("processing_type") == "GENERAL"])
            error_questions = len([r for r in results if r.get("processing_type") == "ERROR"])
            
            # 실제 설정값 가져오기
            config_loader = get_config_loader()
            llm_config = config_loader.get_llm_config()
            embeddings_config = config_loader.get_embeddings_config()
            vectorstore_config = config_loader.get_vectorstore_config()
            retriever_config = config_loader.get_retriever_config()
            
            model_config = {
                "router_model": llm_config.get("router", {}).get("model", "solar-pro2"),
                "rag_llm_model": llm_config.get("rag", {}).get("model", "solar-pro2"),
                "general_llm_model": llm_config.get("general", {}).get("model", "solar-pro2"),
                "embedding_model": embeddings_config.get("model", "embedding-query"),
                "chunk_size": vectorstore_config.get("chunk_size", 1000),
                "chunk_overlap": vectorstore_config.get("chunk_overlap", 50),
                "retrieval_k": retriever_config.get("top_k", 5)
            }
            
            # 개별 점수를 각 결과에 추가
            enhanced_results = []
            for i, result in enumerate(results):
                enhanced_result = result.copy()
                if i < len(individual_scores):
                    enhanced_result["ragas_scores"] = individual_scores[i]
                else:
                    enhanced_result["ragas_scores"] = {
                        "faithfulness": 0.0,
                        "answer_relevancy": 0.0,
                        "context_recall": 0.0,
                        "answer_correctness": 0.0,
                        "ragas_score": 0.0
                    }
                enhanced_results.append(enhanced_result)
            
            # 결과 데이터 구성
            evaluation_report = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "dataset_version": dataset["metadata"]["version"],
                    "model_config": model_config,
                    "evaluation_framework": "RAGAS 0.3.2",
                    "routing_enabled": True
                },
                "routing_statistics": {
                    "total_questions": total_questions,
                    "rag_questions": rag_questions,
                    "general_questions": general_questions,
                    "error_questions": error_questions,
                    "rag_percentage": (rag_questions / total_questions * 100) if total_questions > 0 else 0,
                    "general_percentage": (general_questions / total_questions * 100) if total_questions > 0 else 0
                },
                "overall_scores": overall_scores,
                "category_scores": category_scores,
                "detailed_results": enhanced_results,
                "summary": {
                    "total_questions": total_questions,
                    "total_processing_time_ms": sum(r.get("processing_time_ms", 0) for r in results),
                    "avg_processing_time_ms": sum(r.get("processing_time_ms", 0) for r in results) / len(results) if results else 0,
                    "memory_test_count": len([r for r in results if r["category"] == "memory"]),
                    "error_count": error_questions
                }
            }
            
            # 결과 저장
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(evaluation_report, f, ensure_ascii=False, indent=2)
            
            # latest.json 심볼릭 링크 업데이트
            latest_path = self.results_dir / "latest.json"
            if latest_path.exists() or latest_path.is_symlink():
                latest_path.unlink()
            
            # 상대 경로로 심볼릭 링크 생성
            latest_path.symlink_to(filename)
            
            self.logger.log_function_end("save_evaluation_results", f"결과 저장: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.log_error("save_evaluation_results", e)
            raise
    
    def print_evaluation_summary(self, overall_scores: Dict[str, float], 
                               category_scores: Dict[str, Any], 
                               results: List[Dict[str, Any]],
                               individual_scores: List[Dict[str, float]]):
        """평가 결과 요약 출력"""
        self.logger.info("\n" + "="*60)
        self.logger.info("🎯 RAG 시스템 품질 평가 결과 (라우터 기반)")
        self.logger.info("="*60)
        
        # 라우팅 통계
        total_questions = len(results)
        rag_questions = len([r for r in results if r.get("processing_type") == "RAG"])
        general_questions = len([r for r in results if r.get("processing_type") == "GENERAL"])
        error_questions = len([r for r in results if r.get("processing_type") == "ERROR"])
        
        self.logger.info("\n🔄 라우팅 통계:")
        self.logger.info(f"  • 총 질문 수: {total_questions}")
        self.logger.info(f"  • RAG 처리: {rag_questions}개 ({rag_questions/total_questions*100:.1f}%)")
        self.logger.info(f"  • 일반답변: {general_questions}개 ({general_questions/total_questions*100:.1f}%)")
        self.logger.info(f"  • 오류: {error_questions}개")
        
        # 전체 점수
        self.logger.info("\n📊 전체 RAGAS 점수:")
        self.logger.info(f"  • Faithfulness (사실 정확성):     {overall_scores.get('faithfulness', 0):.3f}")
        self.logger.info(f"  • Answer Relevancy (답변 관련성):  {overall_scores.get('answer_relevancy', 0):.3f}")
        self.logger.info(f"  • Context Recall (컨텍스트 회상률): {overall_scores.get('context_recall', 0):.3f}")
        self.logger.info(f"  • Answer Correctness (답변 정확성): {overall_scores.get('answer_correctness', 0):.3f}")
        self.logger.info(f"  • 📈 RAGAS 종합 점수:            {overall_scores.get('ragas_score', 0):.3f}")
        
        # 카테고리별 점수
        self.logger.info("\n📂 카테고리별 분석:")
        for category, scores in category_scores.items():
            self.logger.info(f"  • {category.upper()}: {scores['question_count']}개 질문")
            self.logger.info(f"    - RAG: {scores['rag_count']}개, 일반: {scores['general_count']}개")
            self.logger.info(f"    - 성공률: {scores['success_rate']:.1%}")
            self.logger.info(f"    - 평균 처리시간: {scores['avg_processing_time_ms']:.0f}ms")
        
        # 메모리 테스트 결과
        memory_questions = [r for r in results if r["category"] == "memory"]
        if memory_questions:
            memory_success = len([r for r in memory_questions if not r.get("error")])
            self.logger.info(f"\n🧠 메모리 기능 테스트: {memory_success}/{len(memory_questions)} 성공")
        
        # 처리 통계
        total_time = sum(r.get("processing_time_ms", 0) for r in results)
        avg_time = total_time / len(results) if results else 0
        self.logger.info(f"\n⏱️  처리 시간 통계:")
        self.logger.info(f"  • 총 처리시간: {total_time:.0f}ms")
        self.logger.info(f"  • 평균 처리시간: {avg_time:.0f}ms")
        
        # 개별 질문별 RAGAS 점수 요약
        self.logger.info(f"\n📊 개별 질문별 RAGAS 점수 요약:")
        if individual_scores:
            # 점수 분포 계산
            score_ranges = {
                "excellent (0.8-1.0)": 0,
                "good (0.6-0.8)": 0,
                "fair (0.4-0.6)": 0,
                "poor (0.2-0.4)": 0,
                "very_poor (0.0-0.2)": 0
            }
            
            for scores in individual_scores:
                ragas_score = scores.get("ragas_score", 0.0)
                if ragas_score >= 0.8:
                    score_ranges["excellent (0.8-1.0)"] += 1
                elif ragas_score >= 0.6:
                    score_ranges["good (0.6-0.8)"] += 1
                elif ragas_score >= 0.4:
                    score_ranges["fair (0.4-0.6)"] += 1
                elif ragas_score >= 0.2:
                    score_ranges["poor (0.2-0.4)"] += 1
                else:
                    score_ranges["very_poor (0.0-0.2)"] += 1
            
            for range_name, count in score_ranges.items():
                if count > 0:
                    percentage = (count / len(individual_scores)) * 100
                    self.logger.info(f"  • {range_name}: {count}개 ({percentage:.1f}%)")
        
        self.logger.info("\n" + "="*60)
    
    def run_evaluation(self):
        """전체 평가 프로세스 실행"""
        self.logger.log_success("=== RAG 품질 평가 시작 ===")
        
        try:
            # 1. 데이터셋 로드
            dataset = self.load_evaluation_dataset()
            
            # 2. 시스템 초기화
            if not self.initialize_system():
                self.logger.log_error_with_icon("시스템 초기화 실패")
                return False
            
            # 3. 질문 처리 및 답변 생성
            results = self.process_questions(dataset)
            
            # 4. RAGAS 평가 실행
            overall_scores, individual_scores = self.run_ragas_evaluation(results)
            
            # 5. 카테고리별 분석
            category_scores = self.calculate_category_scores(results, overall_scores)
            
            # 6. 결과 저장
            result_file = self.save_evaluation_results(dataset, results, overall_scores, category_scores, individual_scores)
            
            # 7. 결과 출력
            self.print_evaluation_summary(overall_scores, category_scores, results, individual_scores)
            
            self.logger.info(f"\n💾 상세 결과가 저장되었습니다: {result_file}")
            self.logger.info(f"📋 최신 결과 확인: {self.results_dir}/latest.json")
            
            self.logger.log_success("=== RAG 품질 평가 완료 ===")
            return True
            
        except Exception as e:
            self.logger.log_error("run_evaluation", e)
            self.logger.error(f"\n❌ 평가 중 오류 발생: {str(e)}")
            return False


def main():
    """메인 함수"""
    # 로거 초기화
    log = LoggerManager("Evaluate")
    
    log.info("🚀 RAG 시스템 품질 평가 CLI")
    log.info("WebUI와 동일한 RAG 방식으로 평가를 수행합니다.\n")
    
    try:
        # RAGAS 설정
        # setup_upstage_for_ragas()
        
        # 평가기 생성 및 실행
        evaluator = RAGEvaluator()
        success = evaluator.run_evaluation()
        
        if success:
            log.info("\n✅ 평가가 성공적으로 완료되었습니다!")
            return 0
        else:
            log.error("\n❌ 평가 중 오류가 발생했습니다.")
            return 1
            
    except Exception as e:
        log.error(f"\n❌ 평가 중 오류 발생: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())