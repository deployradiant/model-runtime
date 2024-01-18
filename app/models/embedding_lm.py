from abc import abstractmethod
from app.models.lm import LM
from typing import List, Optional

EMBEDDING_CATEGORY = "embedding"


class EmbeddingLM(LM):
    @classmethod
    def get_category(cls):
        return EMBEDDING_CATEGORY

    @abstractmethod
    def embedding(self, texts: List[str]) -> List[List[float]]:
        pass
