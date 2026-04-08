import hashlib
from pathlib import Path

import onnx

from roop.utils.cache_paths import get_processing_cache_root


_RESOLVED_MODEL_CACHE: dict[tuple[str, int, int], str] = {}


def _get_source_cache_key(model_path: Path) -> tuple[str, int, int]:
    stats = model_path.stat()
    return (str(model_path.resolve()), int(stats.st_mtime_ns), int(stats.st_size))


def _build_patched_model_path(model_path: Path, cache_key: tuple[str, int, int]) -> Path:
    digest = hashlib.sha256("|".join(map(str, cache_key)).encode("utf-8")).hexdigest()[:16]
    return get_processing_cache_root() / "onnx_batch" / f"{model_path.stem}-native-batch-{digest}.onnx"


def _patch_value_info_batch_dim(value_info, batch_dim_param: str) -> bool:
    value_type = getattr(value_info, "type", None)
    if value_type is None or not value_type.HasField("tensor_type"):
        return False
    shape = value_type.tensor_type.shape
    if len(shape.dim) < 1:
        return False
    first_dim = shape.dim[0]
    if not first_dim.HasField("dim_value") or first_dim.dim_value != 1:
        return False
    first_dim.ClearField("dim_value")
    first_dim.dim_param = batch_dim_param
    return True


def _has_dynamic_first_dim(value_info) -> bool:
    value_type = getattr(value_info, "type", None)
    if value_type is None or not value_type.HasField("tensor_type"):
        return False
    shape = value_type.tensor_type.shape
    if len(shape.dim) < 1:
        return False
    first_dim = shape.dim[0]
    return not first_dim.HasField("dim_value")


def _needs_native_batch_patch(model: onnx.ModelProto) -> bool:
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    for value_info in model.graph.input:
        if value_info.name in initializer_names:
            continue
        value_type = getattr(value_info, "type", None)
        if value_type is None or not value_type.HasField("tensor_type"):
            continue
        shape = value_type.tensor_type.shape
        if len(shape.dim) < 1:
            continue
        first_dim = shape.dim[0]
        if first_dim.HasField("dim_value") and first_dim.dim_value == 1:
            return True
    return False


def _has_safe_dynamic_batch_signature(model: onnx.ModelProto) -> bool:
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    for value_info in model.graph.input:
        if value_info.name in initializer_names:
            continue
        if not _has_dynamic_first_dim(value_info):
            return False
    for value_info in model.graph.output:
        if not _has_dynamic_first_dim(value_info):
            return False
    return True


def _is_safe_cached_model(path: Path, source_path: Path) -> bool:
    if not path.exists():
        return False
    if path.resolve() == source_path.resolve():
        return True
    try:
        return _has_safe_dynamic_batch_signature(onnx.load(str(path)))
    except Exception:
        return False


def ensure_native_batch_model(model_path: str, batch_dim_param: str = "batch") -> str:
    source_path = Path(model_path)
    if not source_path.exists():
        return model_path

    cache_key = _get_source_cache_key(source_path)
    cached_path = _RESOLVED_MODEL_CACHE.get(cache_key)
    if cached_path is not None and _is_safe_cached_model(Path(cached_path), source_path):
        return cached_path

    model = onnx.load(str(source_path))
    if not _needs_native_batch_patch(model):
        resolved_path = str(source_path)
        _RESOLVED_MODEL_CACHE[cache_key] = resolved_path
        return resolved_path

    patched_path = _build_patched_model_path(source_path, cache_key)
    if _is_safe_cached_model(patched_path, source_path):
        resolved_path = str(patched_path)
        _RESOLVED_MODEL_CACHE[cache_key] = resolved_path
        return resolved_path

    initializer_names = {initializer.name for initializer in model.graph.initializer}
    changed = False
    for value_info in model.graph.input:
        if value_info.name in initializer_names:
            continue
        changed = _patch_value_info_batch_dim(value_info, batch_dim_param) or changed
    for value_info in model.graph.output:
        changed = _patch_value_info_batch_dim(value_info, batch_dim_param) or changed

    if not changed:
        resolved_path = str(source_path)
        _RESOLVED_MODEL_CACHE[cache_key] = resolved_path
        return resolved_path

    try:
        patched_model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        patched_model = model

    try:
        onnx.checker.check_model(patched_model)
    except Exception:
        resolved_path = str(source_path)
        _RESOLVED_MODEL_CACHE[cache_key] = resolved_path
        return resolved_path
    if not _has_safe_dynamic_batch_signature(patched_model):
        resolved_path = str(source_path)
        _RESOLVED_MODEL_CACHE[cache_key] = resolved_path
        return resolved_path

    patched_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = patched_path.with_suffix(f"{patched_path.suffix}.tmp")
    onnx.save(patched_model, str(temp_path))
    temp_path.replace(patched_path)

    resolved_path = str(patched_path)
    _RESOLVED_MODEL_CACHE[cache_key] = resolved_path
    return resolved_path

