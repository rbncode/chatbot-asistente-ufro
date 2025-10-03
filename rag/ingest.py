import re
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger
from pypdf import PdfReader

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 120


def clean_text(text: str) -> str:
    text = re.sub(r"Página \d+ de \d+", "", text)
    text = re.sub(r"\s*\n\s*", "\n", text.strip())
    return text


def extract_text_from_pdf(pdf_path: Path) -> List[Tuple[int, str]]:
    logger.info(f"Processing PDF: {pdf_path.name}")
    reader = PdfReader(pdf_path)
    page_texts = []
    for i, page in enumerate(reader.pages):
        page_texts.append((i + 1, page.extract_text() or ""))
    return page_texts


def extract_text_from_txt(file_path: Path) -> List[Tuple[int, str]]:
    logger.info(f"Processing TXT: {file_path.name}")
    text = file_path.read_text(encoding="utf-8")
    return [(1, text)]


def create_chunks(
    doc_id: str,
    doc_title: str,
    doc_url: str,
    page_texts: List[Tuple[int, str]],
) -> List[Dict]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )

    chunks = []
    for page_num, page_text in page_texts:
        cleaned_page_text = clean_text(page_text)
        if not cleaned_page_text:
            continue

        page_chunks = text_splitter.create_documents([cleaned_page_text])
        for i, chunk in enumerate(page_chunks):
            chunks.append(
                {
                    "doc_id": doc_id,
                    "doc_title": doc_title,
                    "page": page_num,
                    "url": doc_url,
                    "chunk_id": f"{doc_id}_page_{page_num}_chunk_{i}",
                    "text": chunk.page_content,
                }
            )
    return chunks


def main():
    PROCESSED_DATA_DIR.mkdir(exist_ok=True)
    sources_df = pd.read_csv(DATA_DIR / "sources.csv", skipinitialspace=True)
    all_chunks = []

    for _, row in sources_df.iterrows():
        doc_id = row["doc_id"]
        file_path = RAW_DATA_DIR / row["path"]

        if not file_path.exists():
            logger.warning(f"File not found for doc_id: {doc_id} at {file_path}. Skipping.")
            continue

        page_texts = []
        if file_path.suffix == ".pdf":
            page_texts = extract_text_from_pdf(file_path)
        elif file_path.suffix == ".txt":
            page_texts = extract_text_from_txt(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_path.suffix} for doc_id: {doc_id}. Skipping.")
            continue

        if page_texts:
            doc_chunks = create_chunks(
                doc_id, row["title"], row["url"], page_texts
            )
            all_chunks.extend(doc_chunks)

    chunks_df = pd.DataFrame(all_chunks)
    output_path = PROCESSED_DATA_DIR / "chunks.parquet"
    chunks_df.to_parquet(output_path)
    logger.info(f"Successfully created {len(all_chunks)} chunks.")
    logger.info(f"Processed chunks saved to: {output_path}")

if __name__ == "__main__":
    main()