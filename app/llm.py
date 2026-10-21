"""Streaming chat with any OpenAI-compatible local server (Ollama, vLLM, LM Studio)."""
from typing import AsyncIterator

from openai import AsyncOpenAI

from .config import settings


class ChatModel:
    def __init__(self):
        self.client = AsyncOpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        """Yield raw text tokens. The server strips any <think> reasoning."""
        response = await self.client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            stream=True,
            max_tokens=settings.llm_max_tokens,
            temperature=0.8,
        )
        async for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta
