from typing import List
from app.models.dollyv2_lm import LanguageGeneratorLmDolly
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from .instruct_pipeline import InstructionTextGenerationPipeline
import torch


MODEL_NAME = "databricks/dolly-v2-12b"
MEDIUM_MODEL_VERSION = "7dde7ac26ddcb679e0429fed1b6f751c1ffab1fb"


class LanguageGeneratorLmDollyLarge(LanguageGeneratorLmDolly):
    def __init__(
        self, model_name: str = MODEL_NAME, model_version: str = MEDIUM_MODEL_VERSION
    ):
        super().__init__(model_name, model_version)

    @classmethod
    def get_model_names(cls) -> List[str]:
        return ["dollyv2_large"]

    def setup(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, revision=self.model_version, padding_side="left"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            revision=self.model_version,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            load_in_8bit=True,
            device_map="auto",
        )
        self.model.tie_weights()

        self.pipe = InstructionTextGenerationPipeline(
            do_sample=False,
            model=self.model,
            tokenizer=self.tokenizer,
        )
