# Data Storage and Management

This folder manages all data-related operations for the LangChain project including caching, outputs, and input data.

## Folder Structure

### `cache/`
**Purpose**: Store cached responses and temporary data
**What to do here**:
- Cache LLM responses to avoid redundant API calls
- Store vector embeddings for faster retrieval
- Cache preprocessed data
- Implement cache expiration strategies

**Example usage**:
```python
# Cache LLM responses
cache_key = f"{provider}_{model}_{hash(prompt)}"
cached_response = load_from_cache(f"cache/{cache_key}.json")

# Cache vector embeddings
embedding_cache = "cache/embeddings/document_vectors.pkl"
```

**File organization**:
```
cache/
├── llm_responses/     # Cached LLM API responses
├── embeddings/        # Vector embeddings
├── preprocessed/      # Preprocessed data
└── temp/             # Temporary files
```

### `outputs/`
**Purpose**: Store generated content and results
**What to do here**:
- Save chat session transcripts
- Export conversation histories
- Store generated reports and summaries
- Archive experiment results

**Example outputs**:
```
outputs/
├── chat_sessions/     # Saved conversations
├── reports/          # Generated summaries/reports  
├── experiments/      # Experiment results
└── exports/         # Data exports
```

**File naming conventions**:
```
chat_sessions/session_2025-06-10_14-30-45.json
reports/weekly_summary_2025-06-10.md
experiments/provider_comparison_2025-06-10.csv
```

### `prompts/`
**Purpose**: Store prompt files and templates
**What to do here**:
- Create reusable prompt files
- Store complex multi-part prompts
- Version control prompt iterations
- Organize prompts by use case

**Organization**:
```
prompts/
├── system/           # System prompts
├── templates/        # Prompt templates
├── chains/          # Multi-step prompt chains
└── examples/        # Example prompts for learning
```

**Example files**:
```python
# prompts/system/code_reviewer.txt
You are a senior software engineer with expertise in Python.
Review code for best practices, security, and efficiency.

# prompts/templates/summarize.txt
Summarize the following {content_type} in {word_count} words:
Focus on {key_aspects}.

Content: {content}
```

## Data Management Best Practices

### 1. **Cache Management**
```python
# Implement cache with TTL
from datetime import datetime, timedelta

def cache_with_expiry(key, data, hours=24):
    cache_data = {
        "data": data,
        "expires": datetime.now() + timedelta(hours=hours)
    }
    save_to_cache(key, cache_data)
```

### 2. **Output Organization**
```python
# Structured output saving
def save_conversation(session_id, messages, metadata):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"data/outputs/chat_sessions/{session_id}_{timestamp}.json"
    
    output = {
        "session_id": session_id,
        "timestamp": timestamp,
        "messages": messages,
        "metadata": metadata
    }
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
```

### 3. **Prompt Management**
```python
# Load and use prompts
def load_prompt_template(name):
    with open(f"data/prompts/templates/{name}.txt", 'r') as f:
        return f.read()

def format_prompt(template_name, **kwargs):
    template = load_prompt_template(template_name)
    return template.format(**kwargs)
```

## File Formats

- **Cache**: JSON, Pickle for complex objects
- **Outputs**: JSON for structured data, Markdown for reports
- **Prompts**: Plain text (.txt) or Markdown (.md)

## Cleanup and Maintenance

### Automated Cleanup Script
```python
# data/cleanup.py
def cleanup_old_cache(days=7):
    """Remove cache files older than specified days"""
    
def archive_old_outputs(days=30):
    """Archive outputs older than 30 days"""
    
def validate_prompt_templates():
    """Validate all prompt templates for syntax"""
```

## Environment Variables

```env
# Data paths (optional overrides)
DATA_CACHE_PATH="data/cache"
DATA_OUTPUT_PATH="data/outputs"
DATA_PROMPTS_PATH="data/prompts"

# Cache settings
CACHE_TTL_HOURS=24
MAX_CACHE_SIZE_MB=1000
```

## Usage Examples

```python
# Save experiment results
from utils.data_manager import save_experiment_result

result = compare_providers(prompt="What is AI?")
save_experiment_result("provider_comparison", result)

# Load cached embeddings
from utils.cache_manager import get_cached_embedding

embedding = get_cached_embedding(text="document content")

# Use prompt template
from utils.prompt_manager import format_prompt

prompt = format_prompt("summarize", 
                      content_type="article",
                      word_count="100",
                      content=article_text)
```