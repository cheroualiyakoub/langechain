from langchain.chat_models import ChatOpenAI

class OpenRouterWrapper:
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        self.api_key = api_key
        self.base_url = base_url

    def ask(self, prompt: str, model: str) -> str:
        llm = ChatOpenAI(
            openai_api_key=self.api_key,
            model_name=model,
            temperature=0.7,
            openai_api_base=self.base_url,
        )
        return str(llm.invoke(prompt).content)