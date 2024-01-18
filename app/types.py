from typing import Optional
from pydantic import BaseModel


class FinetuningTask(BaseModel):
    task_id: str
    start_ts: float
    end_ts: Optional[float] = None
    state: str
    base_model: str
    model_name: str


class FinetunedModelConfig(BaseModel):
    model_name: str
    base_model: str
