from threading import Thread
from typing import Any, Dict
from app.models.text_generation_lm import Answer, TextGenerationLM
from app.config import config
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)
import torch
from jsonformer import Jsonformer


MODEL_CONFIG = {
    "llama2": {
        "name": "meta-llama/Llama-2-7b-chat-hf",
        "version": "40c5e2b32261834431f89850c8d5359631ffa764",
    }
}


class LLama2ChatLM(TextGenerationLM):
    def __init__(self, model_name: str):
        if model_name not in MODEL_CONFIG.keys():
            raise ValueError(
                f"{model_name} is not a valid model name. Valid names are: {MODEL_CONFIG.keys()}"
            )

        self.model_version = MODEL_CONFIG[model_name]["version"]
        super().__init__(MODEL_CONFIG[model_name]["name"])

    @classmethod
    def get_model_names(cls):
        return ["llama2"]

    def setup(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            revision=self.model_version,
            use_fast=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            revision=self.model_version,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            rope_scaling={
                "type": "dynamic",
                "factor": 2.0,
            },
        )
        self.model.to("cuda:0")

    async def answer(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        verbose: bool = False,
        stream: bool = False,
    ) -> Answer:
        if verbose or config.is_debug_mode():
            print("prompt: ", prompt)
            print("temperature: ", temperature)
            print("max_tokens: ", max_tokens)

        with torch.no_grad():
            tokenized_inputs = self.tokenizer(prompt, return_tensors="pt").to(
                self.model.device
            )
            if stream:
                streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
                kwargs = dict(
                    input_ids=tokenized_inputs.input_ids,
                    attention_mask=tokenized_inputs.attention_mask,
                    max_new_tokens=max_tokens,
                    do_sample=temperature == 0,
                    temperature=temperature,
                    streamer=streamer,
                )
                t = Thread(target=self.model.generate, kwargs=kwargs)
                t.start()
                return streamer
            else:
                tokenized_output = self.model.generate(
                    input_ids=tokenized_inputs.input_ids,
                    attention_mask=tokenized_inputs.attention_mask,
                    max_new_tokens=max_tokens,
                    do_sample=temperature == 0,
                    temperature=temperature,
                )
                out_text = self.tokenizer.decode(
                    tokenized_output[0], skip_special_tokens=True
                )

                if verbose or config.is_debug_mode():
                    print("out_text: ", out_text)

                if out_text.startswith(prompt):
                    out_text = out_text[len(prompt) :].strip()

                return out_text

    async def extract(
        self, prompt: str, schema: Dict[str, Any], temperature: float, max_tokens: int
    ) -> Dict[str, Any]:
        former = Jsonformer(
            self.model,
            self.tokenizer,
            schema,
            prompt,
            temperature=temperature,
            max_string_token_length=max_tokens,
        )
        return former()
