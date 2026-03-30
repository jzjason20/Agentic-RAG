import sys
from pathlib import Path
import asyncio
import json

root = Path(__file__).parent.parent
sys.path.append(str(root))

from episodic_rag import EpisodicRAG
from utils.helper import count_tokens


def test_episodic_rag():
    rag = EpisodicRAG()
    chunks = asyncio.run(rag.custom_text_splitters("2023-10-0T00:00:00"))
    print(f"Retrieved {len(chunks)} chunks.")

    rag.index_creation(chunks)

    query = "What is RAG setup am i building"
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}\n")

    # Test WITHOUT reranker
    print("--- WITHOUT Reranker (using only vector similarity) ---")
    result_no_rerank = rag.retrieve_chunks(query, top_k=3, use_reranker=False)
    if result_no_rerank:
        for i, res in enumerate(result_no_rerank, 1):
            print(f"\n{i}. Score: {res['score']:.4f} | Type: {res['type']}")
            print(f"   Context preview: {res['context'][:200]}...")
            print(f"   Token count: {count_tokens(res['context'])}")
    else:
        print("No results found.")

    # Test WITH reranker
    print(f"\n{'='*60}")
    print("--- WITH Reranker (cross-encoder reranking) ---")
    result_rerank = rag.retrieve_chunks(query, top_k=3, use_reranker=True, initial_k=20)
    if result_rerank:
        for i, res in enumerate(result_rerank, 1):
            print(f"\n{i}. Rerank Score: {res.get('rerank_score', 0):.4f} | Original Score: {res.get('original_score', res['score']):.4f} | Type: {res['type']}")
            print(f"   Context preview: {res['context'][:200]}...")
            print(f"   Token count: {count_tokens(res['context'])}")
    else:
        print("No results found.")


if __name__ == "__main__":
    test_episodic_rag()
