import os
import yaml
from typing import Callable, Dict, Any, Tuple
from .openai_provider import OpenAIWrapper
from .openrouter_provider import OpenRouterWrapper
from .anthropic_provider import AnthropicWrapper

class LLMRouter:
    def __init__(self, provider: str = None, model: str = None, api_key: str = None, base_url: str = None, config_path: str = None):
        self.config = self._load_config(config_path or "config/model_config.yaml")
        
        if provider and model:
            # Direct specification
            self.provider = provider.lower()
            self.model = model
        elif model:
            # Auto-detect from model name
            self.provider, self.model = self._detect_provider_from_model(model)
        else:
            raise ValueError("Either provider+model or just model must be specified")
        
        # Get provider-specific configuration
        provider_config = self.config.get(self.provider, {})
        self.api_key = api_key or provider_config.get("api_key") or os.getenv("API_KEY")
        self.base_url = base_url or provider_config.get("base_url") or os.getenv("API_BASE")

        if self.provider not in self._provider_registry:
            raise ValueError(f"Unsupported provider: {self.provider}")

        self.client = self._provider_registry[self.provider](self.api_key, self.base_url)

    def ask(self, prompt: str) -> str:
        return self.client.ask(prompt, self.model)

    def switch_provider(self, provider: str, model: str = None):
        """Switch to a different provider at runtime"""
        self.provider = provider.lower()
        if model:
            self.model = model
        
        provider_config = self.config.get(self.provider, {})
        self.api_key = provider_config.get("api_key") or os.getenv("API_KEY")
        self.base_url = provider_config.get("base_url") or os.getenv("API_BASE")
        self.client = self._provider_registry[self.provider](self.api_key, self.base_url)

    def get_available_models(self, provider: str = None) -> list:
        """Get list of available models for a provider"""
        target_provider = provider or self.provider
        return self.config.get(target_provider, {}).get("default_models", [])

    def ask_with_model(self, prompt: str, model: str = None) -> str:
        """Ask with a specific model, auto-switching provider if needed"""
        if model and model != self.model:
            old_provider = self.provider
            old_model = self.model
            try:
                new_provider, _ = self._detect_provider_from_model(model)
                if new_provider != self.provider:
                    self.switch_provider(new_provider, model)
                else:
                    self.model = model
                return self.ask(prompt)
            finally:
                # Restore original provider/model if changed
                if old_provider != self.provider or old_model != self.model:
                    self.switch_provider(old_provider, old_model)
        else:
            return self.ask(prompt)

    def _detect_provider_from_model(self, model: str) -> Tuple[str, str]:
        """Auto-detect provider based on model name patterns"""
        if model.startswith(("gpt-", "text-", "code-")):
            return "openai", model
        elif model.startswith("claude-"):
            return "anthropic", model
        elif "/" in model:  # OpenRouter format: provider/model
            return "openrouter", model
        else:
            raise ValueError(f"Cannot detect provider for model: {model}")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
                # Expand environment variables
                return self._expand_env_vars(config)
        except FileNotFoundError:
            return {}
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
            return {}

    def _expand_env_vars(self, obj):
        """Recursively expand environment variables in config"""
        if isinstance(obj, dict):
            return {k: self._expand_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._expand_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            env_var = obj[2:-1]
            return os.getenv(env_var, obj)
        return obj

    @property
    def _provider_registry(self) -> dict[str, Callable]:
        return {
            "openai": OpenAIWrapper,
            "anthropic": AnthropicWrapper,
            "openrouter": OpenRouterWrapper,
        }


