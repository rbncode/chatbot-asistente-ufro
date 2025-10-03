import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from loguru import logger
from sentence_transformers import SentenceTransformer, util

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from rag.retrieve import Retriever
from providers.chatgpt import ChatGPTProvider
from providers.deepseek import DeepSeekProvider

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).parent.parent
EVAL_DIR = BASE_DIR / "eval"
GOLD_SET_PATH = EVAL_DIR / "gold_set.jsonl"
RESULTS_PATH = EVAL_DIR / "evaluation_results.csv"

# --- Evaluation Parameters ---
K_FOR_RETRIEVAL = 5  # Number of chunks to retrieve for evaluation
SEMANTIC_SIMILARITY_THRESHOLD = 0.7  # Threshold for semantic accuracy

# Load semantic similarity model
SIMILARITY_MODEL = SentenceTransformer('all-MiniLM-L6-v2')


def load_gold_set() -> List[Dict[str, Any]]:
    """Loads the gold set from the .jsonl file."""
    if not GOLD_SET_PATH.exists():
        raise FileNotFoundError(f"Gold set not found at {GOLD_SET_PATH}")
    
    with open(GOLD_SET_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def calculate_retriever_precision_at_k(retrieved_chunks: List[Dict], expected_refs: List[Any], k: int) -> float:
    if not retrieved_chunks:
        return 0.0
    
    retrieved_doc_ids = {str(chunk['doc_id']) for chunk in retrieved_chunks}
    
    expected_refs_set = {str(ref) for ref in expected_refs}
    
    correct_retrievals = len(retrieved_doc_ids.intersection(expected_refs_set))
    return correct_retrievals / k


def calculate_semantic_accuracy(generated_answer: str, expected_answer: str) -> float:
    if not generated_answer or not expected_answer:
        return 0.0
    
    embedding1 = SIMILARITY_MODEL.encode(generated_answer, convert_to_tensor=True)
    embedding2 = SIMILARITY_MODEL.encode(expected_answer, convert_to_tensor=True)

    similarity = util.cos_sim(embedding1, embedding2).item()
    
    return max(0.0, similarity)


def calculate_citation_coverage(llm_response: str, retrieved_chunks: List[Dict]) -> float:
    if not retrieved_chunks:
        return 0.0
    
    cited_count = 0
    
    for i, chunk in enumerate(retrieved_chunks, 1):
        if f"[{i}]" in llm_response:
            cited_count += 1
        elif str(chunk.get('doc_id', '')) in llm_response:
            cited_count += 1
    
    return cited_count / len(retrieved_chunks)


def build_rag_messages(question: str, retrieved_chunks: List[Dict]) -> List[Dict[str, str]]:
    context_parts = []
    
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_parts.append(f"[{i}] {chunk.get('text', '')}")
    
    context = "\n\n".join(context_parts)
    
    system_message = """You are a helpful assistant. Answer questions based on the provided context. 
If you use information from the context, cite the source using [1], [2], etc."""
    
    user_message = f"""Context:
{context}

Question: {question}

Answer:"""
    
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]


def estimate_cost(provider_name: str, input_tokens: int, output_tokens: int) -> float:
    pricing = {
        "gpt-4o-mini": (0.150, 0.600),  # Precios
        "deepseek-chat": (0.14, 0.28),  # Precios
    }
    
    provider_key = provider_name.lower()
    if provider_key not in pricing:
        return 0.0
    
    input_price, output_price = pricing[provider_key]
    
    cost = (input_tokens / 1_000_000 * input_price) + (output_tokens / 1_000_000 * output_price)
    
    return cost


def estimate_tokens(messages: List[Dict[str, str]]) -> int:
    total_text = " ".join([msg["content"] for msg in messages])
    return len(total_text) // 4


def main():

    logger.info("Starting evaluation process...")
    
    gold_set = load_gold_set()
    retriever = Retriever()
    
    results = []

    providers = {
        "gpt-4o-mini": ChatGPTProvider(),
        "deepseek-chat": DeepSeekProvider(), 
    }
    
    logger.info(f"Evaluating with providers: {list(providers.keys())}")
    logger.info(f"Note: RAG chunks are already processed and stored in the vector database\n")
    
    for item in gold_set:
        question = item["question"]
        expected_answer = item.get("answer", "")
        expected_refs = item["references"]
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Question {gold_set.index(item) + 1}/{len(gold_set)}: '{question}'")
        logger.info(f"{'='*80}")
        
        start_time = time.time()
        retrieved_chunks = retriever.retrieve(question, k=K_FOR_RETRIEVAL)
        retrieval_latency = time.time() - start_time
        
        precision_at_k = calculate_retriever_precision_at_k(
            retrieved_chunks, expected_refs, K_FOR_RETRIEVAL
        )
        
        logger.info(f"Retriever Precision@{K_FOR_RETRIEVAL}: {precision_at_k:.2f}")
        logger.info(f"Retrieval Latency: {retrieval_latency:.4f} seconds")
        
        rag_messages = build_rag_messages(question, retrieved_chunks)
        input_tokens = estimate_tokens(rag_messages)
        
        for provider_name, provider in providers.items():
            logger.info(f"\n--- Evaluating with provider: {provider_name.upper()} ---")
            
            try:
                start_time = time.time()
                llm_response = provider.chat(rag_messages)
                llm_latency = time.time() - start_time
                
                output_tokens = len(llm_response) // 4
                
                logger.info(f"LLM Response: {llm_response[:200]}...")
                logger.info(f"LLM Latency: {llm_latency:.4f} seconds")
                
                semantic_accuracy = calculate_semantic_accuracy(llm_response, expected_answer)
                logger.info(f"Semantic Accuracy: {semantic_accuracy:.2f}")
                
                citation_coverage = calculate_citation_coverage(llm_response, retrieved_chunks)
                logger.info(f"Citation Coverage: {citation_coverage:.2f}")
                
                estimated_cost = estimate_cost(provider_name, input_tokens, output_tokens)
                logger.info(f"Estimated Cost: ${estimated_cost:.6f}")
                
                results.append({
                    "question": question,
                    "provider": provider_name,
                    "retriever_precision_at_k": precision_at_k,
                    "retrieval_latency_sec": retrieval_latency,
                    "llm_answer": llm_response,
                    "llm_latency_sec": llm_latency,
                    "semantic_accuracy": semantic_accuracy,
                    "citation_coverage": citation_coverage,
                    "estimated_cost_usd": estimated_cost,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                })
                
            except Exception as e:
                logger.error(f"Error evaluating {provider_name}: {str(e)}")
                results.append({
                    "question": question,
                    "provider": provider_name,
                    "retriever_precision_at_k": precision_at_k,
                    "retrieval_latency_sec": retrieval_latency,
                    "llm_answer": f"ERROR: {str(e)}",
                    "llm_latency_sec": 0,
                    "semantic_accuracy": 0,
                    "citation_coverage": 0,
                    "estimated_cost_usd": 0,
                    "input_tokens": input_tokens,
                    "output_tokens": 0,
                })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Evaluation complete. Results saved to {RESULTS_PATH}")
    logger.info(f"{'='*80}")
    
    print("\nObtained Results:")
    print(results_df[['question', 'provider', 'semantic_accuracy', 'citation_coverage', 'estimated_cost_usd']])

if __name__ == "__main__":
    main()

