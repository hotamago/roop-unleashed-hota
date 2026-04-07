import psutil
import torch
from typing import Optional

import roop.globals


DEFAULT_STAGED_CHUNK_SIZE = 96
DEFAULT_PREFETCH_FRAMES = 24
DEFAULT_SWAP_BATCH_SIZE = 32
DEFAULT_MASK_BATCH_SIZE = 64
DEFAULT_ENHANCE_BATCH_SIZE = 8
DEFAULT_SINGLE_BATCH_WORKERS = 1
DEFAULT_DETECT_PACK_FRAME_COUNT = 256


def _bytes_to_gb(num_bytes: float) -> float:
    return max(0.0, num_bytes / (1024 ** 3))


def _round_gb(value: float) -> float:
    return round(max(0.0, value), 2)


def _clamp_int(value, default: int, minimum: int, maximum: int) -> int:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        resolved = default
    if resolved <= 0:
        resolved = default
    return max(minimum, min(resolved, maximum))


def provider_uses_gpu() -> bool:
    if not roop.globals.execution_providers:
        return False
    provider = str(roop.globals.execution_providers[0])
    return any(name in provider for name in ("CUDAExecutionProvider", "ROCMExecutionProvider", "TensorrtExecutionProvider"))


def get_available_ram_gb() -> float:
    return _bytes_to_gb(psutil.virtual_memory().available)


def get_available_vram_gb() -> Optional[float]:
    if not provider_uses_gpu():
        return None
    if not torch.cuda.is_available():
        return None
    try:
        free_bytes, _ = torch.cuda.mem_get_info(roop.globals.cuda_device_id)
        return _bytes_to_gb(free_bytes)
    except Exception:
        return None


def resolve_memory_plan(width: int = 0, height: int = 0) -> dict:
    cfg = roop.globals.CFG
    available_vram = get_available_vram_gb()
    chunk_size = _clamp_int(getattr(cfg, "staged_chunk_size", DEFAULT_STAGED_CHUNK_SIZE), DEFAULT_STAGED_CHUNK_SIZE, 8, 480)
    prefetch_frames = _clamp_int(getattr(cfg, "prefetch_frames", DEFAULT_PREFETCH_FRAMES), DEFAULT_PREFETCH_FRAMES, 1, 256)
    prefetch_frames = min(prefetch_frames, chunk_size)
    plan = {
        "available_ram_gb": _round_gb(get_available_ram_gb()),
        "available_vram_gb": None if available_vram is None else _round_gb(available_vram),
        "chunk_size": chunk_size,
        "prefetch_frames": prefetch_frames,
        "swap_batch_size": _clamp_int(getattr(cfg, "swap_batch_size", DEFAULT_SWAP_BATCH_SIZE), DEFAULT_SWAP_BATCH_SIZE, 1, 256),
        "mask_batch_size": _clamp_int(getattr(cfg, "mask_batch_size", DEFAULT_MASK_BATCH_SIZE), DEFAULT_MASK_BATCH_SIZE, 1, 512),
        "enhance_batch_size": _clamp_int(getattr(cfg, "enhance_batch_size", DEFAULT_ENHANCE_BATCH_SIZE), DEFAULT_ENHANCE_BATCH_SIZE, 1, 128),
        "single_batch_workers": _clamp_int(getattr(cfg, "single_batch_workers", DEFAULT_SINGLE_BATCH_WORKERS), DEFAULT_SINGLE_BATCH_WORKERS, 1, 8),
        "detect_pack_frame_count": _clamp_int(getattr(cfg, "detect_pack_frame_count", DEFAULT_DETECT_PACK_FRAME_COUNT), DEFAULT_DETECT_PACK_FRAME_COUNT, 8, 2048),
    }
    roop.globals.active_memory_plan = plan
    roop.globals.runtime_memory_status = describe_memory_plan(plan)
    return plan


def describe_memory_plan(plan: Optional[dict] = None) -> str:
    if plan is None:
        plan = roop.globals.active_memory_plan
    if not plan:
        return "Resource tuning: not computed yet"
    available_vram = plan.get("available_vram_gb")
    vram_text = "CPU / no VRAM telemetry" if available_vram is None else f"avail VRAM={available_vram:.2f} GB"
    return (
        f"Resource tuning: chunk={plan['chunk_size']} | detect pack={plan['detect_pack_frame_count']} | "
        f"prefetch={plan['prefetch_frames']} | swap batch={plan['swap_batch_size']} | "
        f"single-batch workers={plan['single_batch_workers']} | mask batch={plan['mask_batch_size']} | "
        f"enhance batch={plan['enhance_batch_size']} | avail RAM={plan['available_ram_gb']:.2f} GB | {vram_text}"
    )
