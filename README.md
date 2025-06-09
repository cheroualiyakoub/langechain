generative_ai_project/
│
├── config/
│   ├── __init__.py
│   ├── model_config.yaml
│   ├── prompt_templates.yaml
│   └── logging_config.yaml
│
├── src/
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── claude_client.py
│   │   ├── gpt_client.py
│   │   └── utils.py
│   │
│   ├── prompt_engineering/
│   │   ├── __init__.py
│   │   ├── templates.py
│   │   ├── few_shot.py
│   │   └── chainer.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── rate_limiter.py
│   │   ├── token_counter.py
│   │   ├── cache.py
│   │   └── logger.py
│   │
│   └── handlers/
│       ├── __init__.py
│       └── error_handler.py
│
├── data/
│   ├── cache/
│   ├── prompts/
│   ├── outputs/
│   └── embeddings/
│
├── examples/
│   ├── basic_completion.py
│   ├── chat_session.py
│   └── chain_prompts.py
│
├── notebooks/
│   ├── prompt_testing.ipynb
│   ├── response_analysis.ipynb
│   └── model_experimentation.ipynb
│
├── requirements.txt
├── setup.py
├── README.md
└── Dockerfile


## Project Overview

A structured generative AI project template for building robust AI applications, following best practices for maintainability and scalability.

---

## Key Components

- `config/` – Configuration files separate from code
- `src/` – Core source code with modular organization
- `data/` – Organized storage for different data types
- `examples/` – Implementation references
- `notebooks/` – Experimentation and analysis

---

## Best Practices

- Use YAML for configuration files
- Implement proper error handling
- Use rate limiting for APIs
- Separate model clients
- Cache results appropriately
- Maintain documentation
- Use notebooks for testing

---

## Getting Started

1. Clone the repository
2. Install dependencies from `requirements.txt`
3. Configure model settings
4. Review the example code
5. Start with the notebooks

---

## Development Tips

- Follow modular design
- Write component tests
- Use version control
- Keep documentation updated
- Monitor API usage

---

## Core Files

- `requirements.txt` – Package dependencies
- `README.md` – Project documentation
- `Dockerfile` – Containerization setup
