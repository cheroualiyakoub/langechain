from anthropic import Anthropic

class AnthropicWrapper:
    def __init__(self, api_key, base_url=None):
        self.client = Anthropic(api_key=api_key)

    def ask(self, prompt: str, model: str) -> str:
        res = self.client.messages.create(
            model=model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        return res.content[0].text