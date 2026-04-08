from types import SimpleNamespace

import roop.config.globals
from roop.config.settings import Settings
from roop.face_swap_models import (
    get_face_swap_model_choices,
    get_face_swap_upscale_choices,
    parse_face_swap_upscale_size,
)
from roop.memory.planner import describe_memory_plan, resolve_memory_plan, resolve_single_batch_workers
from roop.pipeline.options import ProcessOptions
from roop.pipeline.staged_executor.cache import get_staged_cache_options_snapshot


def test_settings_loads_and_persists_manual_stage_tuning(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("provider: cuda\n", encoding="utf-8")

    cfg = Settings(str(config_path))

    assert cfg.detect_pack_frame_count == 256
    assert cfg.staged_chunk_size == 96
    assert cfg.prefetch_frames == 24
    assert cfg.swap_batch_size == 32
    assert cfg.mask_batch_size == 64
    assert cfg.enhance_batch_size == 8
    assert cfg.single_batch_workers == 1
    assert cfg.face_swap_model == "inswapper_128"
    assert cfg.subsample_upscale == "256px"

    cfg.detect_pack_frame_count = 512
    cfg.staged_chunk_size = 128
    cfg.prefetch_frames = 48
    cfg.swap_batch_size = 40
    cfg.mask_batch_size = 96
    cfg.enhance_batch_size = 12
    cfg.single_batch_workers = 3
    cfg.face_swap_model = "hyperswap_1a_256"
    cfg.subsample_upscale = "512px"
    cfg.save()

    reloaded = Settings(str(config_path))
    assert reloaded.detect_pack_frame_count == 512
    assert reloaded.staged_chunk_size == 128
    assert reloaded.prefetch_frames == 48
    assert reloaded.swap_batch_size == 40
    assert reloaded.mask_batch_size == 96
    assert reloaded.enhance_batch_size == 12
    assert reloaded.single_batch_workers == 3
    assert reloaded.face_swap_model == "hyperswap_1a_256"
    assert reloaded.subsample_upscale == "512px"


def test_settings_normalizes_subsample_for_256_face_swap_models(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "face_swap_model: hyperswap_1b_256\nsubsample_upscale: 128px\n",
        encoding="utf-8",
    )

    cfg = Settings(str(config_path))

    assert cfg.face_swap_model == "hyperswap_1b_256"
    assert cfg.subsample_upscale == "256px"


def test_settings_rounds_hyperswap_upscale_to_supported_choice(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "face_swap_model: hyperswap_1c_256\nsubsample_upscale: 384px\n",
        encoding="utf-8",
    )

    cfg = Settings(str(config_path))

    assert cfg.face_swap_model == "hyperswap_1c_256"
    assert cfg.subsample_upscale == "512px"
    assert get_face_swap_upscale_choices(cfg.face_swap_model) == ["256px", "512px", "768px", "1024px"]


def test_face_swap_model_choices_include_inswapper_fp16():
    assert "inswapper_128_fp16" in get_face_swap_model_choices()
    assert get_face_swap_upscale_choices("inswapper_128_fp16") == ["128px", "256px", "384px", "512px", "768px", "1024px"]


def test_resolve_memory_plan_uses_manual_stage_tuning(monkeypatch):
    monkeypatch.setattr("roop.memory.planner.provider_uses_gpu", lambda: False)
    monkeypatch.setattr(
        roop.config.globals,
        "CFG",
        SimpleNamespace(
            detect_pack_frame_count=320,
            staged_chunk_size=96,
            prefetch_frames=120,
            swap_batch_size=48,
            mask_batch_size=96,
            enhance_batch_size=12,
            single_batch_workers=3,
        ),
        raising=False,
    )
    monkeypatch.setattr("roop.memory.planner.get_available_ram_gb", lambda: 24.0)
    monkeypatch.setattr("roop.memory.planner.get_available_vram_gb", lambda: 10.0)

    plan = resolve_memory_plan(1920, 1080)

    assert plan["chunk_size"] == 96
    assert plan["prefetch_frames"] == 96
    assert plan["swap_batch_size"] == 48
    assert plan["mask_batch_size"] == 96
    assert plan["enhance_batch_size"] == 12
    assert plan["single_batch_workers"] == 3
    assert plan["detect_pack_frame_count"] == 320
    assert "single-batch workers=3" in describe_memory_plan(plan)


def test_resolve_memory_plan_caps_gpu_single_batch_workers(monkeypatch):
    monkeypatch.setattr("roop.memory.planner.provider_uses_gpu", lambda: True)
    monkeypatch.setattr(
        roop.config.globals,
        "CFG",
        SimpleNamespace(
            detect_pack_frame_count=256,
            staged_chunk_size=128,
            prefetch_frames=48,
            swap_batch_size=8,
            mask_batch_size=8,
            enhance_batch_size=8,
            single_batch_workers=2,
        ),
        raising=False,
    )
    monkeypatch.setattr("roop.memory.planner.get_available_ram_gb", lambda: 24.0)
    monkeypatch.setattr("roop.memory.planner.get_available_vram_gb", lambda: 10.0)

    plan = resolve_memory_plan(1920, 1080)

    assert plan["single_batch_workers"] == 1
    assert plan["requested_single_batch_workers"] == 2
    assert plan["single_batch_workers_reason"] == "GPU-safe cap"
    assert "single-batch workers=1 (requested 2, GPU-safe cap)" in describe_memory_plan(plan)


def test_resolve_single_batch_workers_keeps_cpu_parallelism(monkeypatch):
    monkeypatch.setattr("roop.memory.planner.provider_uses_gpu", lambda: False)

    effective_workers, requested_workers, reason = resolve_single_batch_workers(4)

    assert effective_workers == 4
    assert requested_workers == 4
    assert reason is None


def test_process_options_coerce_subsample_size_for_256_face_swap_models():
    options = ProcessOptions(
        {"faceswap": {}},
        0.6,
        0.5,
        "all",
        0,
        "",
        None,
        1,
        128,
        False,
        False,
        face_swap_model="hyperswap_1a_256",
    )

    assert options.face_swap_model == "hyperswap_1a_256"
    assert options.face_swap_tile_size == 256
    assert options.subsample_size == 256


def test_process_options_rounds_to_supported_hyperswap_subsample_size():
    options = ProcessOptions(
        {"faceswap": {}},
        0.6,
        0.5,
        "all",
        0,
        "",
        None,
        1,
        384,
        False,
        False,
        face_swap_model="hyperswap_1a_256",
    )

    assert options.subsample_size == 512


def test_parse_face_swap_upscale_size_keeps_four_digit_values():
    assert parse_face_swap_upscale_size("1024px", "hyperswap_1a_256") == 1024


def test_staged_cache_snapshot_changes_when_face_swap_model_changes():
    base = {
        "processors": {"faceswap": {}},
        "face_distance_threshold": 0.6,
        "blend_ratio": 0.5,
        "swap_mode": "all",
        "selected_index": 0,
        "masking_text": "",
        "num_swap_steps": 1,
        "subsample_size": 256,
        "restore_original_mouth": False,
        "show_face_masking": False,
    }
    options_a = SimpleNamespace(face_swap_model="inswapper_128", **base)
    options_b = SimpleNamespace(face_swap_model="hyperswap_1b_256", **base)

    assert get_staged_cache_options_snapshot(options_a) != get_staged_cache_options_snapshot(options_b)
