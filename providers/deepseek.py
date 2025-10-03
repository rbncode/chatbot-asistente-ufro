import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI
from loguru import logger

from .base import Provider

load_dotenv()


class DeepSeekProvider:
    def __init__(self, api_key: str = None, model: str = "deepseek-chat"):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY must be set in .env or passed to the constructor.")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com",
        )
        self.model = model

    @property
    def name(self) -> str:
        return "DeepSeek"

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        logger.debug(f"Sending request to DeepSeek with model {self.model}...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs,
            )
            content = response.choices[0].message.content
            logger.debug(f"Received response from DeepSeek: {content[:100]}...")
            return content or ""
        except Exception as e:
            logger.error(f"Error calling DeepSeek API: {e}")
            return "Error: Could not get a response from the provider."
pass