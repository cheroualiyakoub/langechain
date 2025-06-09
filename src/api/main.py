from src.llm.router import LLMRouter
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Initialize with auto-detection - now you can just specify a model!
client = LLMRouter(model="gpt-4.1-nano")  # Auto-detects OpenAI provider

class ChatRequest(BaseModel):
    message: str
    model: str = None
    provider: str = None

@app.get("/health")
async def test_token():
    try:
        response = client.ask("Hello! Can you confirm you're working?")
        return {"response": response, "provider": client.provider, "model": client.model}
    except Exception as e:
        return {"error": str(e)}

@app.post("/chat")
async def chat(request: ChatRequest):
    """Chat with any LLM - specify model or provider+model"""
    try:
        if request.model:
            # Use specific model (auto-detects provider)
            response = client.ask_with_model(request.message, request.model)
        else:
            # Use current model
            response = client.ask(request.message)
        return {
            "response": response, 
            "provider": client.provider, 
            "model": client.model
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/providers")
async def get_providers():
    """Get information about available providers and models"""
    return {
        "current_provider": client.provider,
        "current_model": client.model,
        "available_providers": list(client._provider_registry.keys()),
        "available_models": {
            provider: client.get_available_models(provider) 
            for provider in client._provider_registry.keys()
        }
    }

@app.post("/switch-provider")
async def switch_provider(provider: str, model: str = None):
    """Switch to a different provider at runtime"""
    try:
        client.switch_provider(provider, model)
        return {
            "message": f"Switched to {provider}",
            "provider": client.provider,
            "model": client.model
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
