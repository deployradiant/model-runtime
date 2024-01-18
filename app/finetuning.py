from typing import Dict, List, Tuple
from app.models.lm import LM
from app.s3 import list_finetuned_models_from_s3, upload_finetuned_model_to_s3
from app.types import FinetunedModelConfig, FinetuningTask
from app.config import config
from transformers import TrainingArguments
import tempfile
import os
import uuid
import time
from peft import LoraConfig
from trl import SFTTrainer
from datasets import load_dataset


TASK_STATE_PREPARING = "preparing"
TASK_STATE_TRAINING = "training"
TASK_STATE_SAVING = "saving"
TASK_STATE_UPLOADING = "uploading"
TASK_STATE_DONE = "done"
TASK_STATE_FAILED = "failed"


class FinetuningService:
    def __init__(self):
        # Tasks store information about the finetuning job and are uploaded as config to S3
        self.tasks: Dict[str, FinetuningTask] = {}
        self.models: List[FinetunedModelConfig] = None
        self.load_finetuned_models()

    def _create_task(self, base_model: str, model_name: str) -> str:
        task_id = str(uuid.uuid4())
        self.tasks[task_id] = FinetuningTask(
            task_id=task_id,
            base_model=base_model,
            model_name=model_name,
            state=TASK_STATE_PREPARING,
            start_ts=time.time(),
        )
        return task_id

    def load_finetuned_models(self):
        self.models = list(list_finetuned_models_from_s3())

    def get_models(self) -> List[FinetunedModelConfig]:
        if self.models is None:
            self.load_finetuned_models()
        return self.models

    def get_tasks(self) -> List[FinetuningTask]:
        return list(self.tasks.values())

    # This is inspired by the following blogpost
    # https://medium.com/@ud.chandra/instruction-fine-tuning-llama-2-with-pefts-qlora-method-d6a801ebb19
    # We take a base model, store the examples in temp dir and then run it
    async def autoregressive_ft(
        self, base_model: LM, new_model_name: str, steps: int, examples: List[str]
    ) -> Tuple[str, str]:
        base_model_names = base_model.get_model_names()
        if len(base_model_names) > 1:
            print(
                "Warning: More than one base model found: ",
                base_model_names,
                ". The first one will be used.",
            )

        task_id = self._create_task(
            base_model=base_model_names[0], model_name=new_model_name
        )

        dataset = None
        sample_file_name = None

        with tempfile.NamedTemporaryFile("w", delete=False) as file:
            for example in examples:
                file.write('{"text": "' + example + '" }\n')
            sample_file_name = file.name
        try:
            dataset = load_dataset("json", data_files=sample_file_name, split="train")
        finally:
            os.remove(sample_file_name)

        model, tokenizer = base_model.model, base_model.tokenizer
        model.config.use_cache = False
        # This is a change in API (Llama-2 (and also in the past Bloom) has introduced a new attribute in the config file pretraining)
        # The config is taken from the following blogpost https://github.com/huggingface/transformers/pull/24906
        # This will needed to be made configurable
        model.config.pretraining_tp = 1
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )

        temp_output_dir = tempfile.TemporaryDirectory()
        training_args = TrainingArguments(
            output_dir=temp_output_dir.name,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            max_steps=steps,
            # TODO: Figure out WANDB integration later
            report_to="none",
        )
        max_seq_length = 512

        # ensure tokenizer is loaded
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            args=training_args,
        )

        print("Running fine-tuning on model...")
        try:
            self.tasks[task_id].state = TASK_STATE_TRAINING
            trainer.train()
        except Exception as e:
            print("Error during training:", e)
            self.tasks[task_id].state = TASK_STATE_FAILED
            return task_id, None

        print("Saving final checkpoint...")
        self.tasks[task_id].state = TASK_STATE_SAVING
        final_folder = os.path.join(temp_output_dir.name, "final_checkpoint")
        trainer.model.save_pretrained(final_folder)
        self.tasks[task_id].end_ts = time.time()

        self.tasks[task_id].state = TASK_STATE_UPLOADING
        try:
            upload_finetuned_model_to_s3(
                task=self.tasks[task_id], model_folder=final_folder
            )
        except Exception as e:
            print("Error during upload:", e)
            self.tasks[task_id].state = TASK_STATE_FAILED
            return task_id, None

        self.load_finetuned_models()
        self.tasks[task_id].state = TASK_STATE_DONE
        return task_id, new_model_name


ft_service = None


def get_finetuning_service() -> FinetuningService:
    global ft_service
    if ft_service is None:
        ft_service = FinetuningService()
    return ft_service
