# Example Scripts and Use Cases

This folder contains practical examples demonstrating how to use the LangChain project components.

## Purpose

Provide ready-to-run examples that showcase different features and use cases of the project. These serve as learning materials and starting points for your own implementations.

## Current Examples

### `chat_session.py`
**Purpose**: Demonstrate conversation management with session persistence
**What it shows**:
- How to start and manage chat sessions
- Conversation history handling
- Provider switching within sessions
- Session persistence and retrieval

**Usage**:
```bash
python examples/chat_session.py
```

## Examples to Add

### 1. **Basic Provider Usage**
Create `examples/basic_provider_usage.py`:
```python
# Show how to use each provider individually
from src.llm.router import LLMRouter

# OpenAI example
openai_client = LLMRouter(provider="openai", model="gpt-4")
response = openai_client.ask("What is machine learning?")

# Anthropic example  
claude_client = LLMRouter(provider="anthropic", model="claude-3-sonnet")
response = claude_client.ask("Explain quantum computing")
```

### 2. **Model Comparison**
Create `examples/compare_models.py`:
```python
# Compare responses across different models
def compare_across_providers(prompt):
    providers = ["openai", "anthropic", "openrouter"]
    results = {}
    
    for provider in providers:
        client = LLMRouter(provider=provider, model="default")
        results[provider] = client.ask(prompt)
    
    return results
```

### 3. **Prompt Engineering**
Create `examples/prompt_engineering.py`:
```python
# Demonstrate different prompting techniques
def few_shot_example():
    prompt = """
    Examples:
    Q: What is 2+2?
    A: 4
    
    Q: What is 3+3?
    A: 6
    
    Q: What is 5+7?
    A: """
    
    return client.ask(prompt)
```

### 4. **Streaming Responses**
Create `examples/streaming_chat.py`:
```python
# Show streaming responses (if implemented)
def stream_conversation():
    for chunk in client.stream("Tell me a story"):
        print(chunk, end="", flush=True)
```

### 5. **Error Handling**
Create `examples/error_handling.py`:
```python
# Demonstrate robust error handling
def safe_llm_call(prompt, retries=3):
    for attempt in range(retries):
        try:
            return client.ask(prompt)
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
```

### 6. **Configuration Examples**
Create `examples/configuration_examples.py`:
```python
# Show different configuration patterns
def load_custom_config():
    custom_config = {
        "provider": "openai",
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 1000
    }
    return LLMRouter(**custom_config)
```

### 7. **Batch Processing**
Create `examples/batch_processing.py`:
```python
# Process multiple prompts efficiently
def process_batch(prompts, provider="openai"):
    client = LLMRouter(provider=provider)
    results = []
    
    for prompt in prompts:
        result = client.ask(prompt)
        results.append(result)
    
    return results
```

### 8. **API Integration**
Create `examples/api_integration.py`:
```python
# Show how to integrate with the FastAPI endpoints
import requests

def test_api_endpoints():
    base_url = "http://localhost:8000"
    
    # Test basic chat
    response = requests.post(f"{base_url}/chat", 
                           json={"message": "Hello!"})
    
    # Test provider switching
    response = requests.post(f"{base_url}/switch-provider",
                           params={"provider": "anthropic"})
```

## File Organization

```
examples/
├── basic/              # Simple usage examples
│   ├── hello_world.py
│   ├── basic_chat.py
│   └── provider_setup.py
├── advanced/           # Complex use cases
│   ├── conversation_management.py
│   ├── model_comparison.py
│   └── prompt_chaining.py
├── integrations/       # External service integrations
│   ├── api_usage.py
│   ├── database_integration.py
│   └── web_scraping.py
└── utilities/          # Helper scripts
    ├── benchmark_models.py
    ├── test_connections.py
    └── data_export.py
```

## Best Practices for Examples

### 1. **Documentation**
- Include docstrings explaining what each example does
- Add comments for complex logic
- Provide expected output samples

### 2. **Error Handling**
- Show proper exception handling
- Demonstrate graceful degradation
- Include retry mechanisms

### 3. **Configuration**
- Use environment variables for API keys
- Show different configuration methods
- Include validation steps

### 4. **Performance**
- Include timing measurements
- Show caching strategies
- Demonstrate efficient patterns

## Running Examples

### Prerequisites
```bash
# Ensure environment is set up
cp .env.example .env
# Add your API keys to .env

# Install dependencies
pip install -r requirements.txt
```

### Individual Examples
```bash
# Run specific example
python examples/chat_session.py

# Run with custom parameters
python examples/compare_models.py --prompt "Explain AI" --providers openai,anthropic
```

### All Examples
```bash
# Run all examples (create this script)
python examples/run_all_examples.py
```

## Creating New Examples

1. **Choose a clear focus**: Each example should demonstrate one main concept
2. **Include comments**: Explain what's happening and why
3. **Handle errors**: Show robust error handling patterns
4. **Test thoroughly**: Ensure examples work in different environments
5. **Document**: Add clear docstrings and usage instructions

## Learning Path

**Beginner**:
1. `basic/hello_world.py`
2. `basic/basic_chat.py`
3. `basic/provider_setup.py`

**Intermediate**:
1. `chat_session.py`
2. `advanced/model_comparison.py`
3. `integrations/api_usage.py`

**Advanced**:
1. `advanced/prompt_chaining.py`
2. `utilities/benchmark_models.py`
3. `integrations/database_integration.py`