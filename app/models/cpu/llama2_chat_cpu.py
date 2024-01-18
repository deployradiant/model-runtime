from typing import Any, Dict, List, Optional

from app.models.text_generation_lm import TextGenerationLM
from ..lm import LM
from ctransformers import AutoModelForCausalLM

MODEL_NAME = "meta-llama-ggml-4bit"


class LLama2ChatCpuLM(TextGenerationLM):
    def __init__(self, ignored: str = None):
        super().__init__(MODEL_NAME)

    @classmethod
    def get_model_names(cls) -> List[str]:
        return ["llama2_cpu"]

    def setup(self) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_cache,
            model_type="llama",  # Has to be in https://github.com/marella/ctransformers#supported-models
        )

    async def answer(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 100,
        verbose: bool = False,
        stream: bool = False,
    ) -> str:
        return self.model(
            prompt, temperature=temperature, max_new_tokens=max_tokens, threads=8
        )
