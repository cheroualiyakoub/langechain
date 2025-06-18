# LangChain Learning Project

A clean, modular LangChain project designed for learning and experimentation with multiple LLM providers.

## Project Overview

This project provides a flexible framework for working with different Language Model providers (OpenAI, Anthropic, OpenRouter) through a unified interface. It's built with Docker for easy deployment and Jupyter notebooks for interactive learning.

## Quick Start

```bash
# Clone and navigate to project
cd langechain

# Start all services
docker compose up --build

# Access services
- API: http://localhost:8000
- Jupyter: http://localhost:8888
- Vector DB: http://localhost:8001
```

## Project Structure

```
langechain/
├── config/          # Configuration files (YAML)
├── data/            # Data storage and cache
├── examples/        # Example usage scripts
├── notebooks/       # Jupyter notebooks for learning
├── src/             # Main source code
├── tests/           # Unit and integration tests
└── docker-compose.yaml  # Container orchestration
```

## Key Features

- 🔄 **Multi-Provider Support**: OpenAI, Anthropic, OpenRouter
- 🎯 **Model Auto-Detection**: Automatic provider routing by model name
- 📝 **Model Aliases**: Use simple names like `gpt4`, `claude`, `mistral`
- 🔧 **Configuration-Driven**: YAML-based setup
- 🐳 **Docker Ready**: Containerized development environment
- 📚 **Learning Focused**: Jupyter notebooks and examples
- 🔗 **Session Management**: Conversation history support
- 🌐 **REST API**: FastAPI-based endpoints

## Environment Setup

Copy `.env.example` to `.env` and add your API keys:

```env
OPENAI_API_KEY="your-openai-key"
ANTHROPIC_API_KEY="your-anthropic-key"
OPENROUTER_API_KEY="your-openrouter-key"
```

## Documentation

- See individual folder README files for detailed information
- Check `notebooks/` for interactive tutorials
- Review `examples/` for usage patterns

## Learning Path

1. **Start with**: `notebooks/expiriment.ipynb`
2. **Explore**: `examples/chat_session.py`
3. **Configure**: `config/model_config.yaml`
4. **Build**: Create your own LLM applications

## Contributing

This is a learning project. Feel free to experiment, break things, and learn from the process!

## License

MIT License - See LICENSE file for details
