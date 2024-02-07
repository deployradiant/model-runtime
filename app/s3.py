import os
from typing import List, Optional, Tuple
from app.types import FinetunedModelConfig, FinetuningTask
import boto3
import tempfile
import re
import hashlib

# N.B. these are credentials for a read-only IAM user that can only read from S3.
MODEL_BUCKET_REGION_NAME: str = os.getenv("MODEL_BUCKET_REGION_NAME", "us-east-1")
MODEL_BUCKET_ACCESS_KEY: str = os.getenv("MODEL_BUCKET_ACCESS_KEY")
MODEL_BUCKET_SECRET_KEY: str = os.getenv("MODEL_BUCKET_SECRET_KEY")
MODEL_BUCKET_NAME = os.getenv("MODEL_BUCKET_NAME", None)


s3_bucket = None

FINETUNED_MODEL_FOLDER = "finetuned"


def get_s3_bucket():
    global s3_bucket

    if s3_bucket is None and MODEL_BUCKET_NAME is not None:
        s3_bucket = boto3.resource(
            service_name="s3",
            region_name=MODEL_BUCKET_REGION_NAME,
            aws_access_key_id=MODEL_BUCKET_ACCESS_KEY,
            aws_secret_access_key=MODEL_BUCKET_SECRET_KEY,
        ).Bucket(MODEL_BUCKET_NAME)
    return s3_bucket


def get_s3_file_hash(key: str):
    if MODEL_BUCKET_NAME is None:
        return None
    return (
        boto3.client(
            service_name="s3",
            region_name=MODEL_BUCKET_REGION_NAME,
            aws_access_key_id=MODEL_BUCKET_ACCESS_KEY,
            aws_secret_access_key=MODEL_BUCKET_SECRET_KEY,
        )
        .head_object(Bucket=MODEL_BUCKET_NAME, Key=key)["ETag"]
        .strip('"')
    )


def get_s3_file_size(key: str):
    if MODEL_BUCKET_NAME is None:
        return None
    return boto3.client(
        service_name="s3",
        region_name=MODEL_BUCKET_REGION_NAME,
        aws_access_key_id=MODEL_BUCKET_ACCESS_KEY,
        aws_secret_access_key=MODEL_BUCKET_SECRET_KEY,
    ).head_object(Bucket=MODEL_BUCKET_NAME, Key=key)["ContentLength"]


def get_md5_hash(file_path: str):
    with open(file_path, "rb") as f:
        md5_hash = hashlib.md5(f.read()).hexdigest()
    return md5_hash


def load_model_from_s3(
    model_name: str, has_tokenizer=False
) -> Tuple[Optional[str], Optional[str]]:
    bucket = get_s3_bucket()
    local_cache_folder = os.path.join(os.path.expanduser("~"), ".cache", "radiant")
    model_path, tokenizer_path = None, None

    folders_to_load = ["model"]
    if has_tokenizer:
        folders_to_load += ["tokenizer"]

    for folder in folders_to_load:
        files = bucket.objects.filter(Prefix=model_name + "/" + folder)
        if len(list(files)) == 0:
            print(
                f"{folder} for {model_name} not found in S3 bucket {MODEL_BUCKET_NAME}"
            )
            continue

        cache_path = os.path.join(local_cache_folder, model_name, folder)

        if folder == "model":
            model_path = cache_path
        elif folder == "tokenizer":
            tokenizer_path = cache_path
        os.makedirs(
            cache_path,
            exist_ok=True,
        )

        for file in files:
            if file.key.endswith("/"):
                continue

            file_path = file.key[len(f"{model_name}/{folder}/") :]
            destination = os.path.join(cache_path, file_path)
            download_file_size = get_s3_file_size(file.key)

            if (
                os.path.exists(destination)
                and os.path.getsize(destination) == download_file_size
            ):
                print(
                    f"File {destination} already exists and matches size, skipping download."
                )
            else:
                if os.path.exists(destination):
                    print(
                        f"File {destination} already exists but sizes don't match, overwriting..."
                    )
                    os.remove(destination)
                else:
                    print(f"File {destination} not found, downloading...")

                bucket.download_file(file.key, destination)

    return model_path, tokenizer_path


def list_finetuned_models_from_s3() -> List[FinetunedModelConfig]:
    regex = re.compile(f"{FINETUNED_MODEL_FOLDER}/.+?/config.json")
    bucket = get_s3_bucket()
    if bucket is None or FINETUNED_MODEL_FOLDER is None:
        return []

    finetuned_model_configs = bucket.objects.filter(
        Prefix=f"{FINETUNED_MODEL_FOLDER}",
    )
    for config in finetuned_model_configs:
        if config.key.endswith("/"):
            continue

        if regex.match(config.key) is None:
            continue

        with tempfile.NamedTemporaryFile() as config_file:
            bucket.download_file(config.key, config_file.name)
            task = FinetuningTask.parse_file(config_file.name)
            yield FinetunedModelConfig(
                model_name=task.model_name,
                base_model=task.base_model,
            )


def upload_finetuned_model_to_s3(task: FinetuningTask, model_folder: str):
    bucket = get_s3_bucket()
    model_files = os.listdir(model_folder)
    for model_file in model_files:
        file_path = os.path.join(model_folder, model_file)
        if os.path.isdir(file_path):
            continue
        bucket.upload_file(
            file_path,
            f"{FINETUNED_MODEL_FOLDER}/{task.model_name}/model/{model_file}",
        )

    config_file_path = None
    with tempfile.NamedTemporaryFile("w", delete=False) as config_file:
        config_file.write(task.json())
        config_file_path = config_file.name

    try:
        bucket.upload_file(
            config_file_path,
            f"{FINETUNED_MODEL_FOLDER}/{task.model_name}/config.json",
        )
    finally:
        os.remove(config_file_path)
