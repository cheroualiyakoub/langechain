# Jupyter Notebooks for Interactive Learning

This folder contains Jupyter notebooks designed for hands-on learning and experimentation with the LangChain project.

## Purpose

Provide interactive learning experiences where you can experiment with code, test different approaches, and understand concepts through direct manipulation and immediate feedback.

## Current Notebooks

### `expiriment.ipynb`
**Purpose**: Main experimentation notebook for testing project features
**What it contains**:
- Environment setup and configuration
- LLM provider testing and comparison
- API endpoint testing with different methods
- Interactive examples of curl requests and Python requests

**How to use**:
1. Start Jupyter from the project root: `docker compose up`
2. Access at `http://localhost:8888`
3. Open `notebooks/expiriment.ipynb`
4. Run cells sequentially to learn different features

## Notebooks to Create

### 1. **Getting Started** (`01_getting_started.ipynb`)
**Purpose**: Complete beginner introduction
**Topics**:
- Project overview and setup
- First LLM interaction
- Understanding providers and models
- Basic configuration

### 2. **Provider Deep Dive** (`02_provider_comparison.ipynb`)
**Purpose**: Explore different LLM providers
**Topics**:
- OpenAI models and capabilities
- Anthropic Claude variations
- OpenRouter model ecosystem
- Performance and cost comparison

### 3. **Prompt Engineering** (`03_prompt_engineering.ipynb`)
**Purpose**: Learn effective prompting techniques
**Topics**:
- System prompts vs user prompts
- Few-shot learning examples
- Chain-of-thought prompting
- Prompt optimization strategies

### 4. **Session Management** (`04_conversation_sessions.ipynb`)
**Purpose**: Advanced conversation handling
**Topics**:
- Creating and managing chat sessions
- Conversation history and context
- Session persistence and retrieval
- Multi-turn conversations

### 5. **API Integration** (`05_api_integration.ipynb`)
**Purpose**: Working with the REST API
**Topics**:
- FastAPI endpoint exploration
- Different request methods (curl, requests, httpx)
- Error handling and debugging
- Authentication and configuration

### 6. **Data Analysis** (`06_data_analysis.ipynb`)
**Purpose**: Analyze LLM responses and performance
**Topics**:
- Response quality metrics
- Token usage analysis
- Cost comparison across providers
- Performance benchmarking

### 7. **Advanced Features** (`07_advanced_features.ipynb`)
**Purpose**: Explore advanced capabilities
**Topics**:
- Custom provider implementation
- Streaming responses
- Batch processing
- Rate limiting and optimization

### 8. **Real-World Applications** (`08_real_world_apps.ipynb`)
**Purpose**: Practical use case implementations
**Topics**:
- Document summarization
- Code review automation
- Content generation
- Data extraction and analysis

## Notebook Structure Template

Each notebook should follow this structure:

```python
# Cell 1: Setup and imports
import sys, os
from dotenv import load_dotenv

# Bootstrap environment
src_path = os.path.abspath("../src")
if src_path not in sys.path:
    sys.path.append(src_path)
load_dotenv("../.env")

print("Environment ready for experiments")

# Cell 2: Import project modules
from llm.router import LLMRouter
import requests

# Cell 3: Learning objectives
"""
# Learning Objectives
By the end of this notebook, you will:
1. Understand [concept 1]
2. Be able to [skill 1]
3. Know how to [application 1]
"""

# Cell 4+: Interactive content with explanations
```

## Best Practices for Notebooks

### 1. **Clear Documentation**
- Use markdown cells to explain concepts
- Include learning objectives at the start
- Add section headers for organization
- Provide context for each code example

### 2. **Interactive Elements**
```python
# Include interactive widgets where helpful
from ipywidgets import interact, widgets

@interact(provider=['openai', 'anthropic', 'openrouter'])
def test_provider(provider):
    client = LLMRouter(provider=provider, model="default")
    return client.ask("Hello, introduce yourself!")
```

### 3. **Error Handling**
```python
# Show both successful and error cases
try:
    response = client.ask("Test prompt")
    print(f"Success: {response}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")
    print("This shows how to handle API errors gracefully")
```

### 4. **Visual Outputs**
```python
# Include visualizations where relevant
import matplotlib.pyplot as plt
import pandas as pd

# Example: Compare response times
def plot_response_times(results):
    df = pd.DataFrame(results)
    df.plot(kind='bar', x='provider', y='response_time')
    plt.title('Provider Response Time Comparison')
    plt.show()
```

## Development Environment

### Jupyter Extensions
Recommended extensions for better learning experience:
```bash
# Install useful extensions
pip install jupyter-contrib-nbextensions
pip install ipywidgets
pip install matplotlib seaborn pandas
```

### Docker Integration
The notebooks run in Docker with:
- Full access to project source code
- Environment variables loaded
- All dependencies pre-installed
- Hot-reload for development

### Notebook Environment Variables
```python
# Check environment setup
import os

required_vars = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY']
for var in required_vars:
    status = "✅" if os.getenv(var) else "❌"
    print(f"{status} {var}")
```

## Learning Path

### Beginner Track
1. `01_getting_started.ipynb` - Basic setup and first interactions
2. `expiriment.ipynb` - Hands-on experimentation
3. `02_provider_comparison.ipynb` - Understanding different providers

### Intermediate Track
1. `03_prompt_engineering.ipynb` - Effective prompting
2. `04_conversation_sessions.ipynb` - Session management
3. `05_api_integration.ipynb` - API usage patterns

### Advanced Track
1. `06_data_analysis.ipynb` - Performance analysis
2. `07_advanced_features.ipynb` - Complex implementations
3. `08_real_world_apps.ipynb` - Practical applications

## Tips for Effective Learning

1. **Run cells incrementally**: Don't just read, execute each cell
2. **Modify examples**: Change parameters and see what happens
3. **Add your own cells**: Experiment with variations
4. **Take notes**: Use markdown cells to document your insights
5. **Save your work**: Export interesting results to the `data/outputs/` folder

## Troubleshooting

### Common Issues
- **Import errors**: Ensure environment setup cell ran successfully
- **API errors**: Check API keys in `.env` file
- **Connection errors**: Use `langchain_api:8000` for internal Docker requests
- **Kernel crashes**: Restart kernel and rerun setup cells

### Getting Help
- Check the main project README for setup instructions
- Review example scripts in `examples/` folder
- Look at configuration files in `config/` folder
- Use the troubleshooting section in each notebook