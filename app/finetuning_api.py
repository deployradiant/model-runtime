from fastapi import Depends
from fastapi.routing import APIRouter
from app.api_types import (
    FinetuneModelRequest,
    FinetuneModelResponse,
    ListFinetuningTasksRepsonse,
)
from app.finetuning import FinetuningService, get_finetuning_service
from .core import ModelProvider
import time

router = APIRouter(prefix="/finetuning")


@router.post("/sft")
async def finetune_lm(
    request: FinetuneModelRequest,
    model_provider: ModelProvider = Depends(ModelProvider.get),
    ft_service: FinetuningService = Depends(get_finetuning_service),
) -> FinetuneModelResponse:
    new_name = (
        request.name
        if request.name is not None
        else f"{request.base_model}_finetuned_{str(time.time())}"
    )

    job_id, model_name = await ft_service.autoregressive_ft(
        base_model=model_provider.load_and_get_model(request.base_model),
        new_model_name=new_name,
        steps=request.steps,
        examples=request.examples,
    )

    return FinetuneModelResponse(job_id=job_id, finetuned_model_name=model_name)


@router.get("/task")
def list_finetuning_tasks(
    ft_service: FinetuningService = Depends(get_finetuning_service),
) -> ListFinetuningTasksRepsonse:
    return ListFinetuningTasksRepsonse(tasks=ft_service.get_tasks())
