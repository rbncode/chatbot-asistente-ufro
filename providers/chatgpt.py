import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI
from loguru import logger

from .base import Provider

load_dotenv()


class ChatGPTProvider:
    def __init__(self, api_key: str = None, model: str = "gpt-4.1-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY must be set in .env or passed to the constructor.")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1")
        self.model = model

    @property
    def name(self) -> str:
        return "ChatGPT"

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:

        logger.debug(f"Sending request to ChatGPT with model {self.model}...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs,
            )
            content = response.choices[0].message.content
            logger.debug(f"Received response from ChatGPT: {content[:100]}...")
            return content or ""
        except Exception as e:
            logger.error(f"Error calling ChatGPT API: {e}")
            return "Error: Could not get a response from the provider."
pass