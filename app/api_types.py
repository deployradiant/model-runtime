from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from app.types import FinetuningTask


class ModelRequest(BaseModel):
    prompt: str
    token_count: int = 100
    temperature: float = 0.0
    verbose: bool = False
    stream: bool = False


class ModelResponse(BaseModel):
    response: str


class ExtractRequest(BaseModel):
    prompt: str
    temperature: float = 0.0
    json_schema: Dict[str, Any]
    max_tokens: int = 100


class ExtractResponse(BaseModel):
    response: Dict[str, Any]


class EmbedMultipleRequest(BaseModel):
    prompts: List[str]


class EmbedMultipleResponse(BaseModel):
    embeddings: List[List[float]]


class SetModelTypeRequest(BaseModel):
    type: str


class ModelConfig(BaseModel):
    model_type: str
    is_available: bool
    category: str


class ListModelsRepsonse(BaseModel):
    models: List[ModelConfig]


class ModelDetailsReponse(BaseModel):
    total_max_tokens: Optional[int]


class FinetuneModelRequest(BaseModel):
    examples: List[str]
    steps: int = 100
    base_model: str
    name: Optional[str] = None


class FinetuneModelResponse(BaseModel):
    job_id: str
    finetuned_model_name: Optional[str]


class ListFinetuningTasksRepsonse(BaseModel):
    tasks: List[FinetuningTask]
