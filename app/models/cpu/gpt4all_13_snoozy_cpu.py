import multiprocessing
from typing import List
from app.models.text_generation_lm import TextGenerationLM
from ctransformers import AutoModelForCausalLM

MODEL_NAME = "nomic-ai/gpt4all-13b-snoozy"


class Gpt4AllSnoozyCpuLM(TextGenerationLM):
    def __init__(self, ignored: str = None):
        super().__init__(MODEL_NAME)

    @classmethod
    def get_model_names(cls) -> List[str]:
        return ["gpt4all-13b-snoozy_cpu"]

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
        if stream:

            def generate():
                for response in self.model(
                    prompt,
                    temperature=temperature,
                    max_new_tokens=max_tokens,
                    stream=True,
                ):
                    yield response

            return generate()

        return self.model(
            prompt,
            temperature=temperature,
            max_new_tokens=max_tokens,
        )
