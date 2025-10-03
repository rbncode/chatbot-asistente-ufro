import argparse
import os
from typing import List, Dict

from dotenv import load_dotenv
from loguru import logger

from providers.base import Provider
from providers.chatgpt import ChatGPTProvider
from providers.deepseek import DeepSeekProvider
from rag.retrieve import Retriever
from rag.prompts import get_system_prompt

load_dotenv()

PROVIDER_MAP = {
    "chatgpt": ChatGPTProvider,
    "deepseek": DeepSeekProvider,
}


def format_context(retrieved_chunks: List[Dict]) -> str:
    context_str = ""
    for i, chunk in enumerate(retrieved_chunks):
        context_str += f"--- Documento de Referencia {i+1} ---\n"
        context_str += f"Título: {chunk['doc_title']}\n"
        context_str += f"Página: {chunk['page']}\n"
        context_str += f"Contenido: {chunk['text']}\n\n"
    return context_str


def main():
    parser = argparse.ArgumentParser(description="Asistente de normativas UFRO")
    parser.add_argument("question", type=str, help="La pregunta a realizar.")
    parser.add_argument(
        "--provider",
        type=str,
        choices=PROVIDER_MAP.keys(),
        default="chatgpt",
        help="El proveedor LLM a utilizar.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="El número de chunks a recuperar.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    if not args.debug:
        logger.remove()
        logger.add(lambda msg: print(msg, end=""), level="INFO")

    logger.info("Recuperando documentos relevantes...")
    retriever = Retriever()
    retrieved_chunks = retriever.retrieve(args.question, k=args.k)
    context = format_context(retrieved_chunks)

    logger.info("Generando prompt para el LLM...")
    prompt = get_system_prompt(context, args.question)
    messages = [{"role": "system", "content": prompt}]

    if args.debug:
        logger.debug(f"--- CONTEXT ---\n{context}")
        logger.debug(f"--- PROMPT ---\n{prompt}")

    provider_class = PROVIDER_MAP[args.provider]

    if args.provider == "chatgpt" and not os.getenv("OPENAI_API_KEY"):
        logger.error("Error: OPENAI_API_KEY no está configurada en el archivo .env.")
        return
    if args.provider == "deepseek" and not os.getenv("DEEPSEEK_API_KEY"):
        logger.error("Error: DEEPSEEK_API_KEY no está configurada en el archivo .env.")
        return

    provider: Provider = provider_class()
    logger.info(f"Enviando consulta al proveedor: {provider.name}...")

    response = provider.chat(messages=messages, temperature=0.0)

    print("\n--- Respuesta del Asistente ---")
    print(response)


if __name__ == "__main__":
    main()