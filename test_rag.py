import sys
from pathlib import Path
import asyncio
import json

root = Path(__file__).parent.parent
sys.path.append(str(root))

from rag.episodic_rag import EpisodicRAG
from utils.helper import count_tokens


def test_episodic_rag():
    rag = EpisodicRAG()
    chunks = asyncio.run(rag.custom_text_splitters("2023-10-0T00:00:00"))
    print(f"Retrieved {len(chunks)} chunks.")

    rag.index_creation(chunks)
    result = rag.retrieve_chunks("What is RAG setup am i building", top_k=3)
    print(result)
    if result:
        for i in result:
            print(f"Chunk: {i['context']}, score: {i['score']}")
            print(f"Token count: {count_tokens(i['context'])}")
    else:
        print("No results found.")


if __name__ == "__main__":
    test_episodic_rag()
