from typing import List

from app.models.e5_utils import average_pool
from app.models.embedding_lm import EmbeddingLM
from app.utils import get_device_to_use
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "intfloat/e5-large-v2"
MODEL_REVISION = "3bd6250601667dc5c96d22b3dbc40fc43db42bf6"


class E5LargeV2(EmbeddingLM):
    def __init__(self, ignored: str = None):
        super().__init__(MODEL_NAME)

    @classmethod
    def get_model_names(cls) -> List[str]:
        return ["e5_v2"]

    def setup(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            revision=MODEL_REVISION,
        )

        self.model = AutoModel.from_pretrained(
            self.model_name,
            revision=MODEL_REVISION,
            torch_dtype=torch.float32,
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
            # embeddings = F.normalize(embeddings, p=2, dim=1)
            # scores = (embeddings[:2] @ embeddings[2:].T) * 100

            return embeddings.tolist()
