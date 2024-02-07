from app.models.lm import LM
from app.models.text_generation_lm import Answer, TextGenerationLM
from app.config import config
from peft import PeftModel
import torch
import os


class FinetunedTextGenerationLm(TextGenerationLM):
    def __init__(self, model_name: str, base_model: LM):
        self.base_model = base_model
        super().__init__(
            model_name=os.path.join("finetuned", model_name), has_tokenizer=False
        )

    def setup(self):
        # try loading the model back in
        self.model = PeftModel.from_pretrained(self.base_model.model, self.model_cache)

    async def answer(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        verbose: bool = False,
        stream: bool = False,
    ) -> Answer:
        with torch.no_grad():
            if config.is_debug_mode() or verbose:
                print("Running fine-tuned model...")
                print("Prompt: ", prompt)
            inputs = self.base_model.tokenizer(prompt, return_tensors="pt").to(
                self.model.device
            )
            raw_outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_tokens,
                do_sample=temperature == 0,
                temperature=temperature,
            )
            output = self.base_model.tokenizer.decode(
                raw_outputs[0], skip_special_tokens=True
            )
            if config.is_debug_mode() or verbose:
                print("Model response")
                print(output)
        return output
