import os
from typing import Callable
from .openai_provider import OpenAIWrapper
from .openrouter_provider import OpenRouterWrapper
from .anthropic_provider import AnthropicWrapper

class LLMRouter:
    def __init__(self, provider: str, model: str, api_key: str = None, base_url: str = None):
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key or os.getenv("API_KEY")
        self.base_url = base_url or os.getenv("API_BASE")

        if self.provider not in self._provider_registry:
            raise ValueError(f"Unsupported provider: {self.provider}")

        self.client = self._provider_registry[self.provider](self.api_key, self.base_url)

    def ask(self, prompt: str) -> str:
        return self.client.ask(prompt, self.model)

    @property
    def _provider_registry(self) -> dict[str, Callable]:
        return {
            "openai": OpenAIWrapper,
            "anthropic": AnthropicWrapper,
            "openrouter": OpenRouterWrapper,
        }


