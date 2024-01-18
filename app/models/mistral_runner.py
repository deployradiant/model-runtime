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


MODEL_CONFIS = {
    "Mistral-7B-Instruct-v0.1": {
        "name": "mistralai/Mistral-7B-Instruct-v0.1",
        "version": "7ad5799710574ba1c1d953eba3077af582f3a773",
    },
    "MistralLite": {
        "amazon/MistralLite" "version": "4ececff5e47771677c8a900453020ea795f4b6cd"
    },
}


class MistralRunner(TextGenerationLM):
    def __init__(self, model_name: str):
        if model_name not in MODEL_CONFIS.keys():
            raise ValueError(
                f"Invalid model name {model_name}. Valid names are {MODEL_CONFIS.keys()}"
            )

        self.model_version = MODEL_CONFIS[model_name]["version"]
        super().__init__(MODEL_CONFIS[model_name]["name"])

    @classmethod
    def get_model_names(cls):
        return ["MistralLite", "Mistral-7B-Instruct-v0.1"]

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
                    do_sample=False if temperature == 0.0 else True,
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
                    do_sample=False if temperature == 0.0 else True,
                    temperature=temperature,
                )
                out_text = self.tokenizer.decode(
                    tokenized_output[0], skip_special_tokens=True
                )

                if verbose or config.is_debug_mode():
                    print("Model response: ", out_text)

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
