#!/bin/bash

PROJECT_NAME=""

echo "Creating project: $PROJECT_NAME"
mkdir -p $PROJECT_NAME/{config,src/{llm,prompt_engineering,langchain_wrappers,models,utils,handlers},data/{prompts,outputs,cache},examples,notebooks,tests}

# Create empty Python module files
touch $PROJECT_NAME/src/llm/__init__.py
touch $PROJECT_NAME/src/prompt_engineering/__init__.py
touch $PROJECT_NAME/src/langchain_wrappers/__init__.py
touch $PROJECT_NAME/src/models/__init__.py
touch $PROJECT_NAME/src/utils/__init__.py
touch $PROJECT_NAME/src/handlers/__init__.py

# Create config files
touch $PROJECT_NAME/config/model_config.yaml
touch $PROJECT_NAME/config/prompt_templates.yaml
touch $PROJECT_NAME/config/logging_config.yaml

# Create example script
touch $PROJECT_NAME/examples/chat_session.py

# Create a placeholder test
touch $PROJECT_NAME/tests/test_llm_clients.py

# Create root files
touch $PROJECT_NAME/.env
touch $PROJECT_NAME/.env.example
touch $PROJECT_NAME/requirements.txt
touch $PROJECT_NAME/setup.py
touch $PROJECT_NAME/README.md
touch $PROJECT_NAME/Dockerfile

echo "✅ Project structure created successfully!"
