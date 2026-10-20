from abc import ABC, abstractmethod
from typing import Optional
import requests


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        pass


class OllamaLLM(BaseLLM):
    def __init__(self, model: str = "qwen2.5-coder:3b", base_url: str = "http://localhost:11434", temperature: float = 0.2):
        self.model = model
        self.endpoint = f"{base_url.rstrip('/')}/api/generate"
        self.temperature = temperature

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = requests.post(self.endpoint, json=payload, timeout=120)
            response.raise_for_status()
        except requests.exceptions.ConnectionError as err:
            raise ConnectionError(
                f"Could not connect to Ollama at '{self.endpoint}'. "
                f"Is Ollama running? (Start it with `ollama serve` or use `--mock`)"
            ) from err
        data = response.json()
        return data.get("response", "").strip()


class MockLLM(BaseLLM):
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        return f"[MockLLM Output] Processed query based on provided context."
