# Source Code - Main Application Logic

This folder contains the core application code organized by functionality. All production code should be placed in appropriate subfolders based on their purpose.

## Folder Structure

### `api/`
**Purpose**: REST API endpoints and FastAPI application
**What to do here**:
- Define API routes and endpoints
- Handle HTTP requests and responses
- Implement authentication and middleware
- API documentation and validation

### `llm/`
**Purpose**: LLM provider implementations and routing logic
**What to do here**:
- Create new LLM provider wrappers
- Implement the unified router interface
- Handle provider-specific configurations
- Manage model switching and auto-detection

### `handlers/`
**Purpose**: Business logic handlers for different operations
**What to do here**:
- Implement request processing logic
- Handle complex business operations
- Coordinate between different components
- Process and transform data

### `models/`
**Purpose**: Data models and schemas
**What to do here**:
- Define Pydantic models for API requests/responses
- Create data validation schemas
- Implement database models (if using databases)
- Define type definitions and interfaces

### `utils/`
**Purpose**: Utility functions and helper modules
**What to do here**:
- Create reusable utility functions
- Implement configuration loaders
- Add logging and monitoring helpers
- Develop common functionality used across the app

### `prompt_engineering/`
**Purpose**: Prompt templates and engineering utilities
**What to do here**:
- Create prompt template managers
- Implement prompt optimization tools
- Build prompt chaining logic
- Develop prompt testing utilities

### `langchain_wrappers/`
**Purpose**: LangChain-specific integrations and wrappers
**What to do here**:
- Wrap LangChain components for your use cases
- Create custom LangChain tools and agents
- Implement vector store integrations
- Build document processing pipelines

## Coding Standards

### 1. **File Organization**
- One class per file (generally)
- Clear, descriptive file names
- Group related functionality
- Use `__init__.py` for package exports

### 2. **Import Standards**
```python
# Standard library imports first
import os
import sys
from typing import Dict, List, Optional

# Third-party imports
import yaml
from fastapi import FastAPI
from pydantic import BaseModel

# Local imports last
from .models import ChatRequest
from ..utils import load_config
```

### 3. **Code Structure**
```python
"""Module docstring explaining purpose and usage."""

import statements...

# Constants
DEFAULT_MODEL = "gpt-3.5-turbo"
MAX_RETRIES = 3

# Type definitions
ConfigDict = Dict[str, Any]

class MyClass:
    """Clear class docstring."""
    
    def __init__(self, param: str):
        """Constructor with type hints."""
        self.param = param
    
    def method_name(self, arg: str) -> str:
        """Method with clear docstring and type hints."""
        return f"processed: {arg}"
```

### 4. **Error Handling**
```python
# Use specific exceptions
class LLMProviderError(Exception):
    """Raised when LLM provider fails."""
    pass

# Implement graceful degradation
def safe_llm_call(prompt: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            return llm.ask(prompt)
        except LLMProviderError as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
```

## Development Workflow

### 1. **Adding New Features**
1. Create appropriate folder structure
2. Write tests first (TDD approach)
3. Implement the feature
4. Update documentation
5. Add examples if relevant

### 2. **Modifying Existing Code**
1. Understand the current interface
2. Write tests for new behavior
3. Implement changes
4. Ensure backward compatibility
5. Update related documentation

### 3. **Code Review Checklist**
- [ ] Clear, descriptive names
- [ ] Proper type hints
- [ ] Adequate error handling
- [ ] Documentation/docstrings
- [ ] Tests included
- [ ] No hardcoded values
- [ ] Follows project patterns

## Architecture Patterns

### 1. **Dependency Injection**
```python
# Good: Inject dependencies
class ChatHandler:
    def __init__(self, llm_router: LLMRouter):
        self.llm_router = llm_router

# Avoid: Hard dependencies
class ChatHandler:
    def __init__(self):
        self.llm_router = LLMRouter()  # Hard to test
```

### 2. **Factory Pattern**
```python
# For creating provider instances
class LLMProviderFactory:
    @staticmethod
    def create_provider(provider_type: str) -> LLMProvider:
        providers = {
            "openai": OpenAIWrapper,
            "anthropic": AnthropicWrapper,
        }
        return providers[provider_type]()
```

### 3. **Strategy Pattern**
```python
# For different processing strategies
class PromptStrategy:
    def process(self, prompt: str) -> str:
        raise NotImplementedError

class SimplePromptStrategy(PromptStrategy):
    def process(self, prompt: str) -> str:
        return prompt

class TemplatePromptStrategy(PromptStrategy):
    def process(self, prompt: str) -> str:
        return template.format(prompt=prompt)
```

## Testing Integration

Each source folder should have corresponding test files:

```
src/
├── api/
│   ├── main.py
│   └── test_main.py
├── llm/
│   ├── router.py
│   └── test_router.py
└── utils/
    ├── config.py
    └── test_config.py
```

## Performance Considerations

### 1. **Async/Await**
```python
# Use async for I/O operations
async def async_llm_call(prompt: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json={"prompt": prompt})
        return response.json()["response"]
```

### 2. **Caching**
```python
# Implement caching for expensive operations
from functools import lru_cache

@lru_cache(maxsize=128)
def get_provider_config(provider: str) -> dict:
    return load_config(f"config/{provider}.yaml")
```

### 3. **Resource Management**
```python
# Use context managers for resources
class LLMConnection:
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
```

## Security Best Practices

1. **Never hardcode secrets**
2. **Validate all inputs**
3. **Use environment variables for configuration**
4. **Implement proper authentication**
5. **Log security events**

## Monitoring and Logging

```python
import logging

# Configure logging
logger = logging.getLogger(__name__)

class LLMRouter:
    def ask(self, prompt: str) -> str:
        logger.info(f"Processing request with {self.provider}")
        start_time = time.time()
        
        try:
            response = self.client.ask(prompt)
            duration = time.time() - start_time
            logger.info(f"Request completed in {duration:.2f}s")
            return response
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise
```