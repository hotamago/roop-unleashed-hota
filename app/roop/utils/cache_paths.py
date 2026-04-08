import os
from pathlib import Path

from roop.utils.io import resolve_relative_path


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_runtime_temp_root() -> Path:
    return ensure_directory(Path(os.environ.get("TEMP", resolve_relative_path("../temp"))))


def get_gradio_temp_root() -> Path:
    return ensure_directory(get_runtime_temp_root() / "gradio")


def get_processing_cache_root() -> Path:
    return ensure_directory(Path(resolve_relative_path("../processing_cache")))


def get_jobs_root() -> Path:
    return ensure_directory(get_processing_cache_root() / "jobs")
