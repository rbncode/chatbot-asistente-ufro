import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, render_template_string, request, jsonify
from loguru import logger

BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR))

from providers.base import Provider
from providers.chatgpt import ChatGPTProvider
from providers.deepseek import DeepSeekProvider
from rag.retrieve import Retriever
from rag.prompts import get_system_prompt

load_dotenv()

app = Flask(__name__)

PROVIDER_MAP = {
    "chatgpt": ChatGPTProvider,
    "deepseek": DeepSeekProvider,
}
retriever = None

def get_retriever():
    """Initializes and returns the retriever, loading it only once."""
    global retriever
    if retriever is None:
        logger.info("Initializing retriever for the first time...")
        retriever = Retriever()
        logger.info("Retriever initialized.")
    return retriever

SIMPLE_HTML = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Asistente Normativa UFRO</title>
    <style>
        body { font-family: sans-serif; margin: 2em; }
        .container { max-width: 800px; margin: 0 auto; }
        textarea { width: 100%; padding: 0.5em; }
        .response { margin-top: 1em; padding: 1em; border: 1px solid #ccc; white-space: pre-wrap; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Asistente de Normativas UFRO</h1>
        <form method="post">
            <label for="question">Tu pregunta:</label><br>
            <textarea name="question" id="question" rows="4">{{ question or '' }}</textarea><br><br>

            <label for="provider">Proveedor LLM:</label>
            <select name="provider" id="provider">
                <option value="chatgpt" {% if provider == 'chatgpt' %}selected{% endif %}>ChatGPT</option>
                <option value="deepseek" {% if provider == 'deepseek' %}selected{% endif %}>DeepSeek</option>
            </select>
            <br><br>
            <button type="submit">Enviar</button>
        </form>

        {% if answer %}
        <h2>Respuesta:</h2>
        <div class="response">{{ answer }}</div>
        {% endif %}
    </div>
</body>
</html>
"""

def format_context(retrieved_chunks):
    context_str = ""
    for i, chunk in enumerate(retrieved_chunks):
        context_str += f"--- Documento de Referencia {i+1} ---\n"
        context_str += f"Título: {chunk['doc_title']}\n"
        context_str += f"Página: {chunk['page']}\n"
        context_str += f"Contenido: {chunk['text']}\n\n"
    return context_str

@app.route("/", methods=["GET", "POST"])
def index():
    question = ""
    provider_name = "chatgpt"
    answer = ""

    if request.method == "POST":
        question = request.form.get("question", "").strip()
        provider_name = request.form.get("provider", "chatgpt")

        if not question:
            answer = "Por favor, ingresa una pregunta."
        else:
            logger.info(f"Received question: '{question}' for provider: {provider_name}")

            # Check for API keys
            api_key_env = "OPENAI_API_KEY" if provider_name == "chatgpt" else "DEEPSEEK_API_KEY"
            if not os.getenv(api_key_env):
                answer = f"Error: La variable de entorno {api_key_env} no está configurada."
                logger.error(answer)
            else:
                # RAG Pipeline
                current_retriever = get_retriever()
                retrieved_chunks = current_retriever.retrieve(question, k=5)
                context = format_context(retrieved_chunks)
                prompt = get_system_prompt(context, question)
                messages = [{"role": "system", "content": prompt}]

                provider_class = PROVIDER_MAP[provider_name]
                provider: Provider = provider_class()

                try:
                    answer = provider.chat(messages=messages, temperature=0.0)
                except Exception as e:
                    logger.error(f"Error from provider {provider.name}: {e}")
                    answer = "Error al contactar al proveedor de LLM. Revisa los logs del servidor."

    return render_template_string(SIMPLE_HTML, question=question, provider=provider_name, answer=answer)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
