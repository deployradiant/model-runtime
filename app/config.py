import os


class Config:
    # Environment variable configuration keys
    TEXT_GENERATION_MODEL_KEY = "TEXT_GENERATION_MODEL"
    TEXT_GENERATION_DISABLED_KEY = "TEXT_GENERATION_DISABLED"
    EMBEDDING_MODEL_KEY = "EMBEDDING_MODEL"
    EMBEDDING_DISABLED_KEY = "EMBEDDING_DISABLED"
    CPU_MODE_KEY = "RADIANT_CPU_MODE"
    DEBUG_MODE_KEY = "RADIANT_DEBUG"

    # Default configuration values
    DEFAULT_TEXT_GENERATION_MODEL = "llama2"
    DEFAULT_TEXT_GENERATION_DISABLED = "false"
    DEFAULT_TEXT_GENERATION_MODEL_CPU = "ollama-wrapper-zephyr"
    DEFAULT_EMBEDDING_MODEL = "e5_v2"
    DEFAULT_EMBEDDING_DISABLED = "false"
    DEFAULT_CPU_MODE = "false"
    DEFAULT_DEBUG_MODE = "false"

    def __init__(self) -> None:
        pass

    def get_text_generation_model(self) -> str:
        default_model = (
            self.DEFAULT_TEXT_GENERATION_MODEL_CPU
            if self.is_cpu_mode()
            else self.DEFAULT_TEXT_GENERATION_MODEL
        )
        return os.environ.get(self.TEXT_GENERATION_MODEL_KEY, default_model)

    def is_text_generation_disabled(self) -> bool:
        return (
            os.environ.get(
                self.TEXT_GENERATION_DISABLED_KEY, self.DEFAULT_TEXT_GENERATION_DISABLED
            ).lower()
            == "true"
        )

    def get_embedding_model(self) -> str:
        return os.environ.get(self.EMBEDDING_MODEL_KEY, self.DEFAULT_EMBEDDING_MODEL)

    def is_embedding_disabled(self) -> bool:
        return (
            os.environ.get(
                self.EMBEDDING_DISABLED_KEY, self.DEFAULT_EMBEDDING_DISABLED
            ).lower()
            == "true"
        )

    def is_cpu_mode(self) -> bool:
        return (
            os.environ.get(self.CPU_MODE_KEY, self.DEFAULT_CPU_MODE).lower() == "true"
        )

    def is_debug_mode(self) -> bool:
        return (
            os.environ.get(self.DEBUG_MODE_KEY, self.DEFAULT_DEBUG_MODE).lower()
            == "true"
        )


config = Config()
