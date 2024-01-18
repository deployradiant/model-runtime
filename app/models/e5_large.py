from typing import List

from app.models.e5_utils import average_pool
from app.models.embedding_lm import EmbeddingLM
from app.utils import get_device_to_use
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "intfloat/e5-large"


class E5LargeLm(EmbeddingLM):
    def __init__(self, ignored: str = None):
        super().__init__(MODEL_NAME)

    def setup(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(
            self.model_name, torch_dtype=torch.float32
        )
        self.model.to(get_device_to_use())

    def embedding(self, texts: List[str]) -> List[List[float]]:
        batch_dict = self.tokenizer(
            texts, max_length=512, padding=True, truncation=True, return_tensors="pt"
        ).to(get_device_to_use())

        with torch.no_grad():
            outputs = self.model(**batch_dict)
            embeddings = average_pool(
                outputs.last_hidden_state, batch_dict["attention_mask"]
            )

            return embeddings.tolist()
