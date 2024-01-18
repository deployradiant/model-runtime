from typing import Any, Dict, List
from ..dollyv2_lm import LanguageGeneratorLmDolly
import torch
from jsonformer import Jsonformer
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

MODEL_NAME = "tiiuae/falcon-7b-instruct"
MODEL_VERSION = "9f16e66a0235c4ba24e321e3be86dd347a7911a0"


class LanguageGeneratorLmFalcon(LanguageGeneratorLmDolly):
    def __init__(self, ignored: str = None):
        super().__init__(MODEL_NAME, MODEL_VERSION)
        return

    def setup(self):
        if self.model_name != MODEL_NAME:
            raise ValueError(f"{self.model_name} is not a valid model name")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, revision=self.model_version
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            revision=self.model_version,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.model.to("cuda:0")

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.model.device,
        )
        self.pipeline.model.config.pad_token_id = (
            self.pipeline.model.config.eos_token_id
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

        in_length = len(self.tokenizer.encode(in_text))
        start_t = time.time()
        print("starting generation", start_t)
        sequences = self.pipeline(
            in_text,
            max_length=in_length + 62,
            do_sample=False,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        answer = sequences[0]["generated_text"]
        end_t = time.time()

        print("done with generation", end_t)
        print(answer)
        print("time taken", end_t - start_t)

        if answer.startswith(in_text):
            answer = answer[len(in_text) :].strip()

        if "<<QUESTION>>" in answer:
            answer = answer.split("<<QUESTION>>")[0].strip()

        return answer

    async def extract(
        self, prompt: str, schema: Dict[str, Any], temperature: float, max_tokens: int
    ) -> str:
        jsonformer = Jsonformer(
            self.model, self.tokenizer, schema, prompt, temperature=temperature
        )
        return jsonformer()

    def __del__(self):
        del self.model
        del self.tokenizer
        del self.pipeline
