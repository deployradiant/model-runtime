from .config import config

if not config.is_cpu_mode():
    import torch


def get_device_to_use() -> str:
    if not config.is_cpu_mode() and torch.cuda.is_available():
        return "cuda:0"
    else:
        return "cpu"
