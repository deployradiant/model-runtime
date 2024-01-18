from abc import abstractmethod
from app.models.lm import LM
from typing import Any, Dict, List, TypeVar

Answer = TypeVar("Answer", str, List[str])
TEXT_GENERATION_CATEGORY = "text_generation"


class TextGenerationLM(LM):
    @classmethod
    def get_category(cls):
        return TEXT_GENERATION_CATEGORY

    @classmethod
    def total_max_tokens(cls):
        return None

    @abstractmethod
    async def answer(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        verbose: bool,
        stream: bool,
    ) -> Answer:
        pass

    async def extract(
        self, prompt: str, schema: Dict[str, Any], temperature: float, max_tokens: int
    ) -> Dict[str, Any]:
        raise NotImplementedError()
