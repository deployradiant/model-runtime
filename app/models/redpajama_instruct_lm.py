from typing import Any, Dict, List
import os
from app.models.lm import LM
from app.models.text_generation_lm import TextGenerationLM
import torch
from jsonformer import Jsonformer
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "togethercomputer/RedPajama-INCITE-7B-Instruct"
MODEL_VERSION = "95667a602ff2646bf67fe3a57c4eb9a1edec87fe"
MIN_TRANSFORMERS_VERSION = "4.25.1"


class LanguageGeneratorLmRedpajama(TextGenerationLM):
    def __init__(self, ignored: str = None):
        # check transformers version
        assert (
            transformers.__version__ >= MIN_TRANSFORMERS_VERSION
        ), f"Please upgrade transformers to version {MIN_TRANSFORMERS_VERSION} or higher."

        self.model_version = MODEL_VERSION
        super().__init__(MODEL_NAME)

    @classmethod
    def get_model_names(cls) -> List[str]:
        return ["redpajama"]

    def setup(self):
        if self.model_name != MODEL_NAME:
            raise ValueError(f"{self.model_name} is not a valid model name")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, revision=self.model_version
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, revision=self.model_version, torch_dtype=torch.float16
        )
        self.model.to("cuda:0")

    async def answer(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        verbose: bool = False,
        stream: bool = False,
    ) -> str:
        return self._create_lm_answer(prompt, temperature, max_tokens)

    def _create_lm_answer(
        self, prompt: str, temperature: float, max_tokens: int
    ) -> str:
        in_text = prompt
        inputs = self.tokenizer(in_text, return_tensors="pt").to(self.model.device)
        input_length = inputs.input_ids.shape[1]
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            return_dict_in_generate=True,
        )
        token = outputs.sequences[0, input_length:]
        return self.tokenizer.decode(token)

    async def extract(
        self, prompt: str, schema: Dict[str, Any], temperature: float, max_tokens: int
    ) -> str:
        jsonformer = Jsonformer(
            self.model, self.tokenizer, schema, prompt, temperature=temperature
        )
        return jsonformer()
