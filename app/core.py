from typing import List, Optional
from app.api_types import ModelConfig, ModelDetailsReponse
from app.config import config
from app.models.embedding_lm import EMBEDDING_CATEGORY, EmbeddingLM
from app.models.text_generation_lm import TEXT_GENERATION_CATEGORY, TextGenerationLM
from fastapi import HTTPException, status
from app.types import FinetunedModelConfig
from app.models.cpu.ollama import Ollama
from .models.lm import LM


if not config.is_cpu_mode():
    from .models.dollyv2_lm import LanguageGeneratorLmDolly
    from .models.redpajama_instruct_lm import LanguageGeneratorLmRedpajama
    from app.models.finetuned_lm import FinetunedTextGenerationLm
    from app.models.llama2_chat import LLama2ChatLM
    from app.models.e5_large_v2 import E5LargeV2
    from app.finetuning import get_finetuning_service
    from app.models.mistral_runner import MistralRunner

    AVAILABLE_MODELS = [
        E5LargeV2,
        LLama2ChatLM,
        LanguageGeneratorLmDolly,
        LanguageGeneratorLmRedpajama,
        MistralRunner,
        Ollama,
    ]
else:
    from app.models.cpu.cli_lm import CliLM
    from app.models.cpu.gpt4all_13_snoozy_cpu import Gpt4AllSnoozyCpuLM
    from app.models.cpu.llama2_chat_cpu import LLama2ChatCpuLM

    AVAILABLE_MODELS = [Ollama, Gpt4AllSnoozyCpuLM, LLama2ChatCpuLM, CliLM]


class ModelProvider:
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            raise Exception(f"{__class__} is not initialized")
        return cls._instance

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, text_generation_model: str = None, embedding_model: str = None):
        self.model_classes = {}
        # A handy dict to get the model class by name
        for model_class in AVAILABLE_MODELS:
            for name in model_class.get_model_names():
                self.model_classes[name] = model_class
        # A cache of models that have been loaded into the GPU
        self.loaded_models = {}

        # Initial loaded models
        self.selected_models = {
            TEXT_GENERATION_CATEGORY: None,
            EMBEDDING_CATEGORY: None,
        }

        if text_generation_model is not None:
            self.selected_models[TEXT_GENERATION_CATEGORY] = text_generation_model
            self.ensure_model_is_loaded(text_generation_model)
        if embedding_model is not None:
            self.selected_models[EMBEDDING_CATEGORY] = embedding_model
            self.ensure_model_is_loaded(embedding_model)

    def _init_model(self, model_name: str) -> LM:
        print("Initializing model:", model_name)
        ft_model = self._get_ft_model(model_name)
        if ft_model is not None:
            base_model = self.load_and_get_model(ft_model.base_model)
            return FinetunedTextGenerationLm(
                model_name=ft_model.model_name, base_model=base_model
            )
        elif model_name not in self.model_classes.keys():
            raise ValueError(f"Unknown model type: {model_name}")

        return self.model_classes[model_name](model_name)

    def _get_ft_model(self, model_name: str) -> Optional[FinetunedModelConfig]:
        if config.is_cpu_mode():
            return None

        ft_models = list(
            filter(
                lambda cfg: cfg.model_name == model_name,
                get_finetuning_service().get_models(),
            )
        )
        if len(ft_models) == 0:
            return None
        if len(ft_models) > 1:
            print(
                f"Warning: More than one finetuning model with name {model_name} found: {', '.join([f.model_name for f in ft_models])}. Only the first one will be used."
            )
        return ft_models[0]

    def _get_category(self, model_name: str) -> str:
        model = self._get_ft_model(model_name)
        if model is not None:
            return self.model_classes[model.base_model].get_category()
        else:
            return self.model_classes[model_name].get_category()

    def _get_model_names_to_operate(self, model_name: str) -> List[str]:
        ft_model = self._get_ft_model(model_name)
        if ft_model is not None:
            return [
                ft_model.base_model,
                ft_model.model_name,
            ]
        else:
            return [model_name]

    def load_and_get_model(self, model_name: str) -> LM:
        # Hack: has the side effect of loading the model
        self.ensure_model_is_loaded(model_name)
        return self.loaded_models[model_name]

    def ensure_model_is_loaded(self, model_name: str) -> None:
        if self.is_loaded(model_name):
            return

        self.loaded_models[model_name] = self._init_model(model_name)

    def is_loaded(self, model_name: str) -> bool:
        return model_name in self.loaded_models.keys()

    def clear_model(self, model_name: str) -> None:
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]

    def get_model_for_category(self, category: str) -> LM:
        return self.load_and_get_model(self.selected_models[category])

    def set_model_for_category(self, category: str, model_name: str):
        model_category = self._get_category(model_name)
        if category != model_category:
            raise ValueError(
                f"Model {model_name} is not of category {category}. It is of category {model_category}."
            )

        previous_model_name = self.selected_models[category]
        if model_name == previous_model_name:
            print(
                f"Model for category {category} is already {model_name}. Skipping change..."
            )
            return

        models_loaded = self._get_model_names_to_operate(previous_model_name)
        models_needed = self._get_model_names_to_operate(model_name)

        models_to_clear = list(set(models_loaded) - set(models_needed))
        for model in models_to_clear:
            self.clear_model(model)

        models_to_load = list(set(models_needed) - set(models_loaded))
        for model in models_to_load:
            self.ensure_model_is_loaded(model)

        self.selected_models[category] = model_name

        self.ensure_model_is_loaded(model_name)

    def get_model_details(self, model_name: str) -> ModelDetailsReponse:
        model_class = self.model_classes[model_name]
        return ModelDetailsReponse(
            total_max_tokens=model_class.get_max_tokens(),
        )

    def list_ft_models(self) -> List[ModelConfig]:
        if config.is_cpu_mode():
            return []
        ft_models = []
        for ft_config in get_finetuning_service().get_models():
            base_model_class = self.model_classes[ft_config.base_model]
            ft_models.append(
                ModelConfig(
                    model_type=ft_config.model_name,
                    category=base_model_class.get_category(),
                    is_available=ft_config.model_name in self.loaded_models.keys(),
                )
            )
        return ft_models

    # Returns models by name and category and if they are loaded
    def list_models(self) -> List[ModelConfig]:
        static_models = [
            ModelConfig(
                model_type=model_name,
                category=model_class.get_category(),
                is_available=model_name in self.loaded_models.keys(),
            )
            for model_name, model_class in self.model_classes.items()
        ]

        return static_models + self.list_ft_models()


model_provider: ModelProvider = None


def init_model_provider():
    global model_provider
    if model_provider is not None:
        return

    available_models: List[str] = []
    for cls in AVAILABLE_MODELS:
        for name in cls.get_model_names():
            available_models.append(name)

    print(available_models)
    text_generation_model, embedding_model = None, None
    if not config.is_text_generation_disabled():
        text_generation_model = config.get_text_generation_model()
    if not config.is_embedding_disabled():
        embedding_model = config.get_embedding_model()

    model_provider = ModelProvider(
        text_generation_model=(
            text_generation_model if text_generation_model in available_models else None
        ),
        embedding_model=(
            embedding_model if embedding_model in available_models else None
        ),
    )


def set_text_generation_model(model: str) -> bool:
    if config.is_text_generation_disabled():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text generation is disabled",
        )
    ModelProvider.get().set_model_for_category(TEXT_GENERATION_CATEGORY, model)


def get_text_generation_model() -> TextGenerationLM:
    if config.is_text_generation_disabled():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text generation is disabled",
        )
    return ModelProvider.get().get_model_for_category(TEXT_GENERATION_CATEGORY)


def get_embedding_model() -> EmbeddingLM:
    if config.is_embedding_disabled():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Embedding is disabled"
        )
    return ModelProvider.get().get_model_for_category(EMBEDDING_CATEGORY)
