from openai import OpenAI as OpenAIClient

class OpenAIWrapper:
    def __init__(self, api_key, base_url=None):
        self.client = OpenAIClient(api_key=api_key, base_url=base_url or "https://api.openai.com/v1")

    def ask(self, prompt: str, model: str) -> str:
        res = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content