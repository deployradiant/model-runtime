from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar
from app.health import set_readiness
from app.s3 import get_s3_bucket, load_model_from_s3
from app.config import config
import gc

if not config.is_cpu_mode():
    import torch


class LM(ABC):
    def __init__(
        self,
        model_name: str,
        has_tokenizer=True,
        load_from_s3=get_s3_bucket() is not None,
    ):
        set_readiness(False)
        self._handle_init(model_name, has_tokenizer, load_from_s3)
        set_readiness(True)

    def _handle_init(
        self,
        model_name: str,
        has_tokenizer=True,
        load_from_s3=get_s3_bucket() is not None,
    ):
        self.model_name = model_name
        if load_from_s3:
            self.model_cache, self.tokenizer_cache = load_model_from_s3(
                self.model_name, has_tokenizer
            )
        self.model = None
        self.tokenizer = None
        if config.is_debug_mode():
            print(f"Setting up {self.model_name}...")
        self.setup()
        if config.is_debug_mode():
            print(f"Done setting up {self.model_name}")

    @abstractmethod
    def setup(self):
        pass

    @classmethod
    def get_category(cls) -> str:
        raise NotImplementedError()

    @classmethod
    def get_model_names(cls) -> List[str]:
        raise NotImplementedError()

    def reset(self):
        if hasattr(self, "model") and self.model is not None:
            del self.model
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            del self.tokenizer
        gc.collect()
        if not config.is_cpu_mode():
            torch.cuda.empty_cache()

    def __del__(self):
        self.reset()
