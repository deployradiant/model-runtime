from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.encoders import jsonable_encoder
from starlette.responses import StreamingResponse
from app.models.embedding_lm import EmbeddingLM
from app.models.text_generation_lm import TextGenerationLM
from app.config import config
from app.api_types import (
    EmbedMultipleRequest,
    EmbedMultipleResponse,
    ExtractRequest,
    ExtractResponse,
    ListModelsRepsonse,
    ModelDetailsReponse,
    ModelRequest,
    ModelResponse,
    SetModelTypeRequest,
)
from .core import (
    ModelProvider,
    get_embedding_model,
    get_text_generation_model,
    set_text_generation_model,
)
import time
import json

OK_RESPONSE: str = "OK"

router = APIRouter(prefix="/api")


@router.post("/embedding")
def embedding(
    request: EmbedMultipleRequest,
    model: EmbeddingLM = Depends(get_embedding_model),
) -> EmbedMultipleResponse:
    vectors = model.embedding(request.prompts)
    return EmbedMultipleResponse(embeddings=vectors)


@router.post("/text_generation", response_model=None)
async def text_generation(
    request: ModelRequest,
    model: TextGenerationLM = Depends(get_text_generation_model),
) -> ModelResponse | StreamingResponse:
    start_time = time.time()
    model_response = await model.answer(
        prompt=request.prompt,
        temperature=request.temperature,
        max_tokens=request.token_count,
        verbose=request.verbose,
        stream=request.stream,
    )
    end_time = time.time()
    if config.is_debug_mode():
        print(f"Duration to lm response: {end_time - start_time} ")

    if request.stream:

        def content_stream():
            while True:
                for text in model_response:
                    yield json.dumps(
                        jsonable_encoder(ModelResponse(response=text))
                    ) + "\n"
                break

        return StreamingResponse(content_stream(), media_type="text/event-stream")
    else:
        return ModelResponse(response=model_response)


@router.post("/extract_json")
async def extract_json(
    model_request: ExtractRequest,
    model: TextGenerationLM = Depends(get_text_generation_model),
) -> ExtractResponse:
    model_response = await model.extract(
        prompt=model_request.prompt,
        schema=model_request.json_schema,
        temperature=model_request.temperature,
        max_tokens=model_request.max_tokens,
    )

    return ExtractResponse(response=model_response)


@router.post("/set_model_type")
def set_text_generation_model_type(
    request: SetModelTypeRequest,
):
    model_type = request.type
    if config.is_debug_mode():
        print("Changing model type to:", model_type)
    if not config.is_text_generation_disabled():
        set_text_generation_model(model_type)
    else:
        print(
            "Setting model type will be ignored due to text generation being disabled"
        )
    return OK_RESPONSE


@router.get("/connect")
async def connect():
    # There exists a bug where calling model.embedding at the same time will cause things to seg fault
    # Warming up the model makes this a non problem
    if not config.is_embedding_disabled():
        embedding_model = get_embedding_model()
        embedding_model.embedding(["Warm up..."])

    if not config.is_text_generation_disabled():
        text_generation_model = get_text_generation_model()
        try:
            await text_generation_model.answer(
                "Warm up...", temperature=0.1, max_tokens=1
            )
        except Exception as e:
            print("Error during warm up:", e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error during warm up",
            )

    return OK_RESPONSE


@router.get("/models")
def list_available_models(
    model_provider: ModelProvider = Depends(ModelProvider.get),
) -> ListModelsRepsonse:
    return ListModelsRepsonse(models=model_provider.list_models())


@router.get("/model/{model_id}")
def get_model_details(
    model_id: str, model_provider: ModelProvider = Depends(ModelProvider.get)
) -> ModelDetailsReponse:
    return model_provider.get_model_details(model_id=model_id)
