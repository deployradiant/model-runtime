from fastapi import APIRouter, Response, status
import os

router = APIRouter(prefix="/health")

# Hack to share the readiness state between uvicorn workers
READINESS_FILEPATH = "/tmp/readiness"

@router.get("/liveness", status_code=status.HTTP_200_OK)
def get_liveness():
    return "OK"

@router.get("/readiness", status_code=status.HTTP_200_OK)
def get_readiness(response: Response):
    if not os.path.exists(READINESS_FILEPATH):
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return

    with open(READINESS_FILEPATH, "r") as f:
        ready = f.read() == "True"

    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return
    return "OK"

def set_readiness(new_readiness: bool):
    with open(READINESS_FILEPATH, "w") as f:
        f.write(str(new_readiness))

def get_readiness_filepath():
    return READINESS_FILEPATH
