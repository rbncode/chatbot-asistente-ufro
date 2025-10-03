from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from loguru import logger

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_PATH = DATA_DIR / "index.faiss"
CHUNKS_PATH = DATA_DIR / "processed" / "chunks.parquet"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class Retriever:
    def __init__(self):
        logger.info("Initializing Retriever...")
        self.index = self._load_index()
        self.chunks_df = self._load_chunks()
        self.model = self._load_model()
        logger.info("Retriever initialized successfully.")

    def _load_index(self) -> faiss.Index:
        if not INDEX_PATH.exists():
            raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}. Please run embed.py.")
        logger.info(f"Loading FAISS index from {INDEX_PATH}...")
        return faiss.read_index(str(INDEX_PATH))

    def _load_chunks(self) -> pd.DataFrame:
        if not CHUNKS_PATH.exists():
            raise FileNotFoundError(f"Chunks data not found at {CHUNKS_PATH}. Please run ingest.py.")
        logger.info(f"Loading chunks metadata from {CHUNKS_PATH}...")
        return pd.read_parquet(CHUNKS_PATH)

    def _load_model(self) -> SentenceTransformer:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}...")
        return SentenceTransformer(EMBEDDING_MODEL)

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:

        logger.debug(f"Retrieving top {k} chunks for query: '{query}'")

        query_embedding = self.model.encode([query], convert_to_numpy=True)

        distances, indices = self.index.search(np.array(query_embedding, dtype="float32"), k)

        retrieved_chunks = []
        for i in indices[0]:
            chunk = self.chunks_df.iloc[i].to_dict()
            retrieved_chunks.append(chunk)

        logger.debug(f"Retrieved {len(retrieved_chunks)} chunks.")
        return retrieved_chunks


# Example usage
if __name__ == "__main__":
    retriever = Retriever()
    test_query = "Cuales son los requisitos para la titulacion?"
    results = retriever.retrieve(test_query, k=3)

    print(f"Results for query: '{test_query}'")
    for res in results:
        print(f"\n--- Chunk from {res['doc_title']} (Page {res['page']}) ---")
        print(res['text'])