from typing import Any, Dict
from transformers import (
    AutoConfig,
    AutoTokenizer,
    pipeline,
    AutoModelForCausalLM,
)
from .dollyv2_lm import LanguageGeneratorLmDolly
import torch


MODEL_NAME = "mosaicml/mpt-7b-instruct"
MODEL_VERSION = "bbe7a55d70215e16c00c1825805b81e4badb57d7"


class LanguageGeneratorLmMosaic(LanguageGeneratorLmDolly):
    def __init__(self, ignored: str = None):
        super().__init__(MODEL_NAME, MODEL_VERSION)
        return

    def setup(self):
        if self.model_name != MODEL_NAME:
            raise ValueError(f"{self.model_name} is not a valid model name")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            # load_in_8bit=True,
            torch_dtype=torch.float16,  # Load model weights in bfloat16
            trust_remote_code=True,
            revision=self.model_version,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, revision=self.model_version
        )
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

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
        prompt = prompt.strip()
        in_text = self._create_prompt(prompt).strip()

        with torch.autocast("cuda", dtype=torch.float16):
            out_text = self.pipeline(
                in_text, max_new_tokens=248, do_sample=False, use_cache=True
            )
            out_text = out_text[0]["generated_text"]

        if out_text.startswith(in_text):
            out_text = out_text[len(in_text) :].strip()

        out_text = out_text.split("<|endoftext|>")[0]

        print(out_text)

        if out_text.startswith(prompt):
            out_text = out_text[len(prompt) :].strip()
        return out_text
