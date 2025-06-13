# Configuration Files

This folder contains all configuration files for the LangChain project in YAML format.

## Files

### `model_config.yaml`
**Purpose**: Configure LLM providers, models, and aliases
**What to do here**:
- Add new LLM provider configurations
- Define model aliases for easier usage
- Set provider-specific settings (API endpoints, default models)
- Configure authentication methods

**Example additions**:
```yaml
# Add new providers
cohere:
  api_key: "${COHERE_API_KEY}"
  base_url: "https://api.cohere.ai/v1"
  default_models:
    - "command-r-plus"
    - "command-r"

# Add more aliases
model_aliases:
  fast: "gpt-3.5-turbo"
  smart: "gpt-4"
  creative: "claude-3-opus-20240229"
```

### `prompt_templates.yaml`
**Purpose**: Store reusable prompt templates
**What to do here**:
- Create prompt templates for common use cases
- Define system prompts for different scenarios
- Store prompt chains and workflows
- Version control your prompts

**Example usage**:
```yaml
templates:
  code_review:
    system: "You are a senior software engineer reviewing code."
    template: "Review this code and provide feedback:\n\n{code}"
  
  summarization:
    system: "You are a helpful assistant that summarizes content."
    template: "Summarize the following text in {length} words:\n\n{text}"
```

### `logging_config.yaml`
**Purpose**: Configure logging levels and output formats
**What to do here**:
- Set logging levels for different components
- Configure log file locations
- Define log formats and handlers
- Set up structured logging

**Example configuration**:
```yaml
version: 1
formatters:
  default:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: default
loggers:
  llm.router:
    level: DEBUG
  api.main:
    level: INFO
```

## Best Practices

1. **Environment Variables**: Use `${VAR_NAME}` for sensitive data
2. **Version Control**: Keep configs in git, exclude secrets
3. **Documentation**: Comment complex configurations
4. **Validation**: Test configs after changes
5. **Backup**: Keep backup copies of working configurations

## Usage in Code

```python
# Load configurations
from utils.config_loader import load_config

model_config = load_config('config/model_config.yaml')
prompts = load_config('config/prompt_templates.yaml')
```

## Adding New Configurations

1. Create new YAML file in this folder
2. Follow existing naming conventions
3. Add corresponding loader in `src/utils/`
4. Update main README with new config info
5. Test configuration loading