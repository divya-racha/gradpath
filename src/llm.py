"""LLM clients over REST APIs (free-tier friendly).

The user supplies their own API key — it is never stored anywhere except
the app's session / Streamlit secrets. Supported providers:
  - Gemini  (free key from https://aistudio.google.com/)
  - OpenAI  (key from https://platform.openai.com/api-keys)
"""
import requests

GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

OPENAI_MODEL = "gpt-4o-mini"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"


class GeminiClient:
    def __init__(self, api_key: str):
        self.api_key = api_key.strip()

    def generate(self, system: str, user: str, temperature: float = 0.4,
                 max_tokens: int = 1200) -> str:
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"parts": [{"text": user}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        r = requests.post(GEMINI_URL, params={"key": self.api_key}, json=payload, timeout=90)
        r.raise_for_status()
        data = r.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError):
            raise RuntimeError(f"Unexpected Gemini response: {str(data)[:300]}")


class OpenAIClient:
    def __init__(self, api_key: str, model: str = OPENAI_MODEL):
        self.api_key = api_key.strip()
        self.model = model

    def generate(self, system: str, user: str, temperature: float = 0.4,
                 max_tokens: int = 1200) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        r = requests.post(OPENAI_URL,
                          headers={"Authorization": f"Bearer {self.api_key}"},
                          json=payload, timeout=90)
        r.raise_for_status()
        data = r.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError):
            raise RuntimeError(f"Unexpected OpenAI response: {str(data)[:300]}")


def make_client(provider: str, api_key: str):
    if provider == "OpenAI":
        return OpenAIClient(api_key)
    return GeminiClient(api_key)
