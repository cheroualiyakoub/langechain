import os
import yaml
from typing import Callable, Dict, Any, Tuple
from .openai_provider import OpenAIWrapper
from .openrouter_provider import OpenRouterWrapper
from .anthropic_provider import AnthropicWrapper

class LLMRouter:
    def __init__(self, provider: str = None, model: str = None, api_key: str = None, base_url: str = None, config_path: str = None):
        self.config = self._load_config(config_path or "config/model_config.yaml")
        
        # Resolve model alias if provided
        resolved_model = self._resolve_model_alias(model) if model else model
        
        if provider and resolved_model:
            # Direct specification
            self.provider = provider.lower()
            self.model = resolved_model
        elif resolved_model:
            # Auto-detect from model name
            self.provider, self.model = self._detect_provider_from_model(resolved_model)
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
            # Resolve alias for the requested model
            resolved_model = self._resolve_model_alias(model)
            old_provider = self.provider
            old_model = self.model
            try:
                new_provider, _ = self._detect_provider_from_model(resolved_model)
                if new_provider != self.provider:
                    self.switch_provider(new_provider, resolved_model)
                else:
                    self.model = resolved_model
                return self.ask(prompt)
            finally:
                # Restore original provider/model if changed
                if old_provider != self.provider or old_model != self.model:
                    self.switch_provider(old_provider, old_model)
        else:
            return self.ask(prompt)

    def _detect_provider_from_model(self, model: str) -> Tuple[str, str]:
        """Auto-detect provider based on model name patterns"""
        # OpenAI models - updated pattern to be more specific
        if model.startswith(("gpt-", "text-", "code-", "davinci", "curie", "babbage", "ada")) or model in ["gpt-4.1-nano", "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]:
            return "openai", model
        # Anthropic models  
        elif model.startswith(("claude-", "claude")):
            return "anthropic", model
        # OpenRouter format: provider/model
        elif "/" in model:
            return "openrouter", model
        else:
            raise ValueError(f"Cannot detect provider for model: {model}")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        # Try multiple possible config paths
        possible_paths = [
            config_path,
            "/app/config/model_config.yaml",  # Docker container path
            "config/model_config.yaml",       # Relative path
            os.path.join(os.path.dirname(__file__), "../../config/model_config.yaml")  # Relative to router.py
        ]
        
        for path in possible_paths:
            try:
                print(f"DEBUG: Trying to load config from: {path}")
                if os.path.exists(path):
                    print(f"DEBUG: Config file exists at: {path}")
                    with open(path, 'r') as f:
                        config = yaml.safe_load(f) or {}
                        print(f"DEBUG: Loaded config: {config}")
                        # Expand environment variables
                        expanded_config = self._expand_env_vars(config)
                        print(f"DEBUG: Expanded config: {expanded_config}")
                        return expanded_config
                else:
                    print(f"DEBUG: Config file not found at: {path}")
            except Exception as e:
                print(f"DEBUG: Error loading config from {path}: {e}")
                continue
        
        print("DEBUG: No config file found, using empty config")
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

    def _resolve_model_alias(self, model: str) -> str:
        """Resolve model alias to actual model name"""
        aliases = self.config.get("model_aliases", {})
        resolved = aliases.get(model, model)
        print(f"DEBUG: Resolving alias '{model}' -> '{resolved}'")
        print(f"DEBUG: Available aliases: {aliases}")
        return resolved

    def get_available_aliases(self) -> Dict[str, str]:
        """Get all available model aliases"""
        return self.config.get("model_aliases", {})


