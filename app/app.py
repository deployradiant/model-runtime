import threading
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import uvicorn
import argparse
import os
from app.config import config
from .api import router as api_router
from .core import init_model_provider
from .health import router as health_router, set_readiness, get_readiness_filepath
from fastapi.middleware.cors import CORSMiddleware


if not config.is_cpu_mode():
    from pynvml import *
    import torch
    from .finetuning_api import router as finetuning_router


def app():
    app = FastAPI(debug=config.is_debug_mode())
    origins = [
        "http://localhost:*",
        "http://localhost:3000",
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if not config.is_cpu_mode():
        api_router.include_router(finetuning_router)

    app.include_router(api_router)
    app.include_router(health_router)

    @app.on_event("startup")
    async def startup():
        init_thread = threading.Thread(target=init_model)
        init_thread.start()

    @app.on_event("shutdown")
    def shutdown():
        os.remove(get_readiness_filepath())

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        exc_str = f"{exc}".replace("\n", " ").replace("   ", " ")
        print(f"{request}: {exc_str}")
        content = {"message": exc_str, "data": None}
        return JSONResponse(
            content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )

    return app


# Initializing the model provider can take a while. Handle it asynchronously to avoid the server
# failing liveness checks, but don't become ready until it's done.
def init_model():
    init_model_provider()
    set_readiness(True)


def server():
    global LM_MODEL_TYPE
    parser = argparse.ArgumentParser(description="Radiant platform on premise wrapper")
    parser.add_argument(
        "--port", type=int, default=8001, help="port to run the server on"
    )
    parser.add_argument(
        "--model_type", type=str, default=None, help="model type to run"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        default=False,
        help="Use the cpu model",
    )

    args = parser.parse_args()
    if args.model_type is not None:
        # this is a dirty hack
        os.environ["LM_MODEL_TYPE"] = args.model_type

    if args.cpu:
        os.environ["RADIANT_CPU_MODE"] = "True"

    if not config.is_cpu_mode():
        torch.cuda.empty_cache()

    uvicorn.run(
        "app.app:app",
        host="0.0.0.0",
        port=args.port,
        # N.B. Workers can't be set when reload is true
        reload=True,
    )
