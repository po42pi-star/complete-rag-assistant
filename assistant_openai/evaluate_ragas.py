"""
Оценка качества RAG системы через RAGAS для assistant_api.
Использует OpenAI API через ProxyAPI для RAG и для метрик RAGAS.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from rag_pipeline import RAGPipeline


# Тестовые вопросы для оценки RAG системы
EVALUATION_QUESTIONS = [
    # === Company Overview ===
    "Какие основные направления деятельности компании Death Star?",
    "Какие технологии входят в технологический стек Death Star?",

    # === DevOps Practices ===
    "Что такое DevOps и какие основные принципы этого подхода?",
    "Что такое Infrastructure as Code и какие инструменты использует Death Star?",

    # === Project Management ===
    "Какие методологии управления проектами используются в Death Star?",
    "Какие роли существуют в проектной команде и какие у них обязанности?",
    "Какие этапы включает жизненный цикл проекта в Death Star?",
]


def prepare_dataset(pipeline: RAGPipeline, questions: list) -> Dataset:
    """
    Подготовка датасета для RAGAS из вопросов.
    
    Args:
        pipeline: RAG pipeline для получения ответов
        questions: список вопросов для оценки
    
    Returns:
        Dataset для RAGAS с полями: question, answer, contexts, ground_truth
    """
    questions_list = []
    answers_list = []
    contexts_list = []
    ground_truths_list = []
    
    print("[*] Получение ответов от RAG системы...\n")
    
    for i, question in enumerate(questions, 1):
        print(f"  {i}/{len(questions)}: {question}")
        
        # Получаем ответ от RAG системы (без использования кеша)
        result = pipeline.query(question, use_cache=False)
        
        # Формируем данные для RAGAS
        questions_list.append(question)
        answers_list.append(result["answer"])
        
        # Контекст - список текстов из найденных документов
        context_texts = [doc["text"] for doc in result["context_docs"]]
        contexts_list.append(context_texts)
        
        # Ground truth - эталонный ответ
        ground_truths_list.append(result["answer"][:100])
        
        print(f"     [+] Ответ получен от OpenAI API")
    
    print()
    
    dataset_dict = {
        "question": questions_list,
        "answer": answers_list,
        "contexts": contexts_list,
        "ground_truth": ground_truths_list
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    return dataset


def create_ragas_metrics():
    """
    Создание метрик RAGAS с поддержкой ProxyAPI через LangChain.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY не установлен")
    
    base_url = os.getenv("OPENAI_API_BASE_URL")
    print(f"[*] Инициализация OpenAI (base_url: {base_url or 'default'})...")
    
    # LangChain объекты с ProxyAPI
    langchain_embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key,
        openai_api_base=base_url
    )
    
    langchain_llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        openai_api_key=api_key,
        openai_api_base=base_url,
        temperature=0
    )
    
    # Обёртки для RAGAS
    ragas_embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)
    ragas_llm = LangchainLLMWrapper(langchain_llm)
    
    from ragas.metrics import Faithfulness, ContextPrecision
    
    faithfulness_metric = Faithfulness(llm=ragas_llm)
    context_precision_metric = ContextPrecision(llm=ragas_llm)
    
    return [faithfulness_metric, context_precision_metric]


def evaluate_rag_system():
    """
    Основная функция оценки RAG-системы через RAGAS.
    """
    print("=" * 70)
    print("ОЦЕНКА КАЧЕСТВА RAG-СИСТЕМЫ (API MODE) ЧЕРЕЗ RAGAS")
    print("=" * 70)
    print()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("[ОШИБКА] OPENAI_API_KEY не установлен")
        print("\nУстановите переменную окружения:")
        print("  Windows (PowerShell): $env:OPENAI_API_KEY='your-key'")
        print("  Linux/Mac: export OPENAI_API_KEY='your-key'")
        sys.exit(1)
    
    try:
        print("[*] Инициализация RAG системы...\n")
        pipeline = RAGPipeline(
            collection_name="api_rag_collection",
            cache_db_path="api_rag_cache.db",
            data_file="data/",
            model="gpt-4o-mini"
        )
        print("\n[OK] RAG система готова к оценке\n")
    except Exception as e:
        print(f"[ОШИБКА] Ошибка инициализации RAG pipeline: {e}")
        sys.exit(1)
    
    print("=" * 70)
    dataset = prepare_dataset(pipeline, EVALUATION_QUESTIONS)
    print("=" * 70)
    
    print("\n[*] Инициализация метрик RAGAS...")
    metrics_to_use = create_ragas_metrics()
    
    print("\n[*] Запуск оценки метрик...")
    print("   Метрики: Faithfulness, Context Precision")
    print("   (это займёт 1-2 минуты)\n")
    
    try:
        result = evaluate(dataset=dataset, metrics=metrics_to_use)
    except Exception as e:
        print(f"[ОШИБКА] Ошибка при оценке: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("=" * 70)
    
    import math
    
    faithfulness_values = [v for v in result['faithfulness'] if not (isinstance(v, float) and math.isnan(v))]
    context_precision_values = [v for v in result['context_precision'] if not (isinstance(v, float) and math.isnan(v))]
    
    avg_faithfulness = sum(faithfulness_values) / len(faithfulness_values) if faithfulness_values else 0
    avg_context_precision = sum(context_precision_values) / len(context_precision_values) if context_precision_values else 0
    
    print(f"\n[МЕТРИКИ] Средние значения:")
    print(f"   Faithfulness (точность ответа):          {avg_faithfulness:.4f}")
    print(f"   Context Precision (точность контекста):  {avg_context_precision:.4f}")
    
    avg_score = (avg_faithfulness + avg_context_precision) / 2
    print(f"\n{'─'*70}")
    print(f"[ИТОГО] Средний балл: {avg_score:.4f}")
    
    if avg_score >= 0.7:
        print("   Оценка: Отличное качество! [OK]")
    elif avg_score >= 0.5:
        print("   Оценка: Удовлетворительное качество [!]")
    else:
        print("   Оценка: Требует улучшения [X]")
    
    print("\n" + "=" * 70)
    print("[OK] Оценка завершена!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    evaluate_rag_system()