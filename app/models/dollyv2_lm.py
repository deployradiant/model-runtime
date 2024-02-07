from typing import Any, Dict, List

from app.models.text_generation_lm import TextGenerationLM
from .lm import LM
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from .instruct_pipeline import InstructionTextGenerationPipeline
import torch
import os
from jsonformer import Jsonformer


def get_max_new_tokens() -> int:
    try:
        return int(os.getenv("MAX_NEW_TOKENS", 128))
    except Exception:
        return 128


MODEL_NAME = "databricks/dolly-v2-7b"
MEDIUM_MODEL_VERSION = "9fd22254be6c5ffb4a2a0f7333e6fbb5a3ebdd93"


class LanguageGeneratorLmDolly(TextGenerationLM):
    def __init__(
        self, model_name: str = MODEL_NAME, model_version: str = MEDIUM_MODEL_VERSION
    ):
        self.model_version = model_version
        super().__init__(model_name)

    @classmethod
    def get_model_names(cls) -> List[str]:
        return ["dollyv2"]

    def setup(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, revision=self.model_version, padding_side="left"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            revision=self.model_version,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.model.to("cuda:0")

        self.pipe = InstructionTextGenerationPipeline(
            do_sample=False,
            # max_new_tokens=512,
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.model.device,
        )

    def _create_prompt(self, prompt: str):
        return prompt

    def ensure_max_tokens(self, prompt: str, max_tokens: int) -> str:
        prompt = prompt.strip()
        prompt_lines = prompt.split("\n")
        total_tokens = 0

        if len(prompt_lines) > 2:
            cleaned_prompt_lines = []
            # the last two lines are instructions so we want to keep them

            total_tokens += len(self.tokenizer.encode(prompt_lines[-1])) + len(
                self.tokenizer.encode(prompt_lines[-2])
            )

            for line in prompt_lines[:-2]:
                tokens = len(self.tokenizer.encode(line))
                # print("Tokens in sentence ", index, " is ", tokens)
                if total_tokens + tokens < max_tokens:  # arbitrary border
                    cleaned_prompt_lines.append(line)
                    total_tokens += tokens
                else:
                    print("Skipping line as it does not fit")

            cleaned_prompt_lines.append(prompt_lines[-2])
            cleaned_prompt_lines.append(prompt_lines[-1])

            prompt = "\n".join(cleaned_prompt_lines)
        return prompt

    def _create_lm_answer(
        self, prompt: str, temperature: float, max_tokens: int
    ) -> str:
        prompt = self.ensure_max_tokens(prompt, 1624)
        in_text = self._create_prompt(prompt).strip()

        with torch.no_grad():
            out_text = self.pipe(
                in_text,
                temperature=temperature,
                max_new_tokens=get_max_new_tokens(),
            )
            out_text = out_text[0]["generated_text"]

        if out_text.startswith(in_text):
            out_text = out_text[len(in_text) :].strip()

        if out_text.startswith(prompt):
            out_text = out_text[len(prompt) :].strip()
        return out_text

    async def answer(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        verbose: bool,
        stream: bool = False,
    ) -> str:
        return self._create_lm_answer(prompt, temperature, max_tokens)

    async def extract(
        self, prompt: str, schema: Dict[str, Any], temperature: float, max_tokens: int
    ) -> Dict[str, Any]:
        prompt = self.ensure_max_tokens(prompt, 1624)
        jsonformer = Jsonformer(
            self.model, self.tokenizer, schema, prompt, temperature=temperature
        )
        response = jsonformer()
        return response
