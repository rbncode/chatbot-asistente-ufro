import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from loguru import logger
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INDEX_PATH = DATA_DIR / "index.faiss"
CHUNKS_PATH = PROCESSED_DATA_DIR / "chunks.parquet"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def main():
    logger.info("Starting embedding generation and FAISS indexing...")

    if not CHUNKS_PATH.exists():
        logger.error(f"Chunks file not found at {CHUNKS_PATH}. Please run the ingest script first.")
        return

    chunks_df = pd.read_parquet(CHUNKS_PATH)
    texts = chunks_df["text"].tolist()
    logger.info(f"Loaded {len(texts)} chunks from {CHUNKS_PATH}.")

    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    logger.info("Generating embeddings for all chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)
    embedding_dim = embeddings.shape[1]
    logger.info(f"Embeddings generated successfully. Shape: {embeddings.shape}")

    logger.info("Creating FAISS index (IndexFlatL2)...")
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.array(embeddings, dtype="float32"))

    faiss.write_index(index, str(INDEX_PATH))
    logger.info(f"FAISS index saved to: {INDEX_PATH}")
    logger.info(f"Total vectors in index: {index.ntotal}")

    logger.info("Embedding and indexing process completed successfully.")

if __name__ == "__main__":
    main()