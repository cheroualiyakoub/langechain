# Testing Suite

This folder contains all tests for the LangChain project, organized to ensure code quality and reliability.

## Purpose

Provide comprehensive testing coverage for all project components, including unit tests, integration tests, and end-to-end tests. This ensures your learning project maintains high quality as you add new features.

## Current Test Files

### `test_llm_clients.py`
**Purpose**: Test LLM provider implementations and router functionality
**What it should test**:
- Individual provider wrapper functionality
- Router auto-detection logic
- Model alias resolution
- Error handling and retries
- Provider switching capabilities

## Test Organization

### Recommended Structure
```
tests/
├── unit/              # Unit tests for individual components
│   ├── test_llm_router.py
│   ├── test_providers.py
│   ├── test_config_loader.py
│   └── test_utils.py
├── integration/       # Integration tests
│   ├── test_api_endpoints.py
│   ├── test_provider_integration.py
│   └── test_session_management.py
├── e2e/              # End-to-end tests
│   ├── test_complete_workflows.py
│   └── test_docker_setup.py
├── fixtures/         # Test data and fixtures
│   ├── sample_responses.json
│   ├── test_configs.yaml
│   └── mock_data.py
└── conftest.py       # Pytest configuration and fixtures
```

## Testing Framework Setup

### Core Dependencies
```python
# requirements-test.txt
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-mock>=3.10.0
pytest-cov>=4.0.0
httpx>=0.24.0  # For async HTTP testing
respx>=0.20.0  # For HTTP mocking
```

### Configuration (`conftest.py`)
```python
import pytest
import os
from unittest.mock import Mock
from src.llm.router import LLMRouter

@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    os.environ["OPENAI_API_KEY"] = "test-key-openai"
    os.environ["ANTHROPIC_API_KEY"] = "test-key-anthropic"
    yield
    # Cleanup
    for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
        os.environ.pop(key, None)

@pytest.fixture
def llm_router(mock_env_vars):
    """Create LLM router for testing."""
    return LLMRouter(model="gpt-3.5-turbo")

@pytest.fixture
def sample_chat_data():
    """Sample chat request data."""
    return {
        "message": "Hello, test!",
        "model": "gpt-3.5-turbo"
    }
```

## Test Categories

### 1. Unit Tests
Test individual components in isolation:

```python
# tests/unit/test_llm_router.py
import pytest
from unittest.mock import Mock, patch
from src.llm.router import LLMRouter

class TestLLMRouter:
    def test_model_alias_resolution(self, mock_env_vars):
        """Test that model aliases are resolved correctly."""
        router = LLMRouter(model="gpt4")
        assert router.model == "gpt-4.1-nano"
    
    def test_provider_auto_detection(self, mock_env_vars):
        """Test automatic provider detection from model names."""
        router = LLMRouter(model="claude-3-sonnet")
        assert router.provider == "anthropic"
    
    @patch('src.llm.openai_provider.OpenAIWrapper.ask')
    def test_ask_method(self, mock_ask, llm_router):
        """Test the ask method returns response."""
        mock_ask.return_value = "Test response"
        
        response = llm_router.ask("Test prompt")
        
        assert response == "Test response"
        mock_ask.assert_called_once_with("Test prompt", llm_router.model)
```

### 2. Integration Tests
Test component interactions:

```python
# tests/integration/test_api_endpoints.py
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

class TestAPIEndpoints:
    def test_health_endpoint(self):
        """Test the health check endpoint."""
        response = client.get("/test-token")
        assert response.status_code == 200
        assert "response" in response.json()
    
    def test_chat_endpoint(self):
        """Test the chat endpoint with valid data."""
        chat_data = {
            "message": "Hello!",
            "model": "gpt-3.5-turbo"
        }
        response = client.post("/chat", json=chat_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "response" in data
        assert "provider" in data
        assert "model" in data
```

### 3. Mock Testing
Test with mocked external dependencies:

```python
# tests/unit/test_providers.py
import pytest
from unittest.mock import Mock, patch
from src.llm.openai_provider import OpenAIWrapper

class TestOpenAIWrapper:
    @patch('openai.OpenAI')
    def test_openai_wrapper_initialization(self, mock_openai):
        """Test OpenAI wrapper initializes correctly."""
        wrapper = OpenAIWrapper("test-key", "https://api.openai.com/v1")
        
        mock_openai.assert_called_once_with(
            api_key="test-key",
            base_url="https://api.openai.com/v1"
        )
    
    @patch('openai.OpenAI')
    def test_ask_method_success(self, mock_openai):
        """Test successful API call."""
        # Setup mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        wrapper = OpenAIWrapper("test-key")
        result = wrapper.ask("Test prompt", "gpt-3.5-turbo")
        
        assert result == "Test response"
```

## Testing Best Practices

### 1. **Test Structure (AAA Pattern)**
```python
def test_something():
    # Arrange - Set up test data
    router = LLMRouter(model="gpt-3.5-turbo")
    test_prompt = "Hello"
    
    # Act - Execute the functionality
    result = router.ask(test_prompt)
    
    # Assert - Verify the results
    assert isinstance(result, str)
    assert len(result) > 0
```

### 2. **Parametrized Tests**
```python
@pytest.mark.parametrize("model,expected_provider", [
    ("gpt-3.5-turbo", "openai"),
    ("claude-3-sonnet", "anthropic"),
    ("mistralai/mistral-7b", "openrouter"),
])
def test_provider_detection(model, expected_provider):
    router = LLMRouter(model=model)
    assert router.provider == expected_provider
```

### 3. **Async Testing**
```python
@pytest.mark.asyncio
async def test_async_llm_call():
    """Test asynchronous LLM calls."""
    router = AsyncLLMRouter(model="gpt-3.5-turbo")
    response = await router.ask_async("Test prompt")
    assert isinstance(response, str)
```

### 4. **Error Testing**
```python
def test_invalid_model_raises_error():
    """Test that invalid models raise appropriate errors."""
    with pytest.raises(ValueError, match="Cannot detect provider"):
        LLMRouter(model="invalid-model-name")

def test_missing_api_key_handling():
    """Test handling of missing API keys."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(Exception):  # Should fail without API key
            router = LLMRouter(model="gpt-3.5-turbo")
            router.ask("test")
```

## Running Tests

### Basic Test Execution
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_llm_router.py

# Run specific test
pytest tests/unit/test_llm_router.py::TestLLMRouter::test_model_alias_resolution

# Run tests with verbose output
pytest -v

# Run tests in parallel
pytest -n auto
```

### Test Categories
```bash
# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run tests by marker
pytest -m "not slow"  # Skip slow tests
pytest -m "integration"  # Run only integration tests
```

## Continuous Integration

### GitHub Actions Example (`.github/workflows/test.yml`)
```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-test.txt
      - name: Run tests
        run: pytest --cov=src --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Testing Utilities

### Mock Helpers
```python
# tests/fixtures/mock_data.py
def create_mock_llm_response(content: str = "Mock response"):
    """Create a mock LLM response."""
    return {
        "response": content,
        "provider": "mock",
        "model": "mock-model"
    }

def create_mock_openai_response(content: str = "Mock response"):
    """Create a mock OpenAI API response."""
    mock_choice = Mock()
    mock_choice.message.content = content
    
    mock_response = Mock()
    mock_response.choices = [mock_choice]
    
    return mock_response
```

### Test Data Management
```python
# tests/fixtures/sample_responses.json
{
  "openai_response": {
    "content": "This is a sample OpenAI response",
    "model": "gpt-3.5-turbo",
    "usage": {"total_tokens": 20}
  },
  "anthropic_response": {
    "content": "This is a sample Anthropic response",
    "model": "claude-3-sonnet"
  }
}
```

## Performance Testing

```python
# tests/performance/test_performance.py
import time
import pytest

def test_response_time():
    """Test that responses are returned within acceptable time."""
    start_time = time.time()
    
    router = LLMRouter(model="gpt-3.5-turbo")
    response = router.ask("Simple question")
    
    duration = time.time() - start_time
    assert duration < 30.0  # Should respond within 30 seconds
```

## Test Environment

### Environment Variables for Testing
```bash
# .env.test
OPENAI_API_KEY="test-key-openai"
ANTHROPIC_API_KEY="test-key-anthropic"
API_BASE="http://mock-api:8000"
LOG_LEVEL="DEBUG"
```

### Docker Testing
```bash
# Run tests in Docker
docker compose -f docker-compose.test.yml up --build

# Run specific test suites
docker compose exec api pytest tests/unit/
```

This comprehensive testing setup ensures your LangChain learning project maintains high quality as you experiment and add new features!