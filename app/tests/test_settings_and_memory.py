from types import SimpleNamespace

import roop.globals
from roop.memory import describe_memory_plan, resolve_memory_plan
from settings import Settings


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

    cfg.detect_pack_frame_count = 512
    cfg.staged_chunk_size = 128
    cfg.prefetch_frames = 48
    cfg.swap_batch_size = 40
    cfg.mask_batch_size = 96
    cfg.enhance_batch_size = 12
    cfg.single_batch_workers = 3
    cfg.save()

    reloaded = Settings(str(config_path))
    assert reloaded.detect_pack_frame_count == 512
    assert reloaded.staged_chunk_size == 128
    assert reloaded.prefetch_frames == 48
    assert reloaded.swap_batch_size == 40
    assert reloaded.mask_batch_size == 96
    assert reloaded.enhance_batch_size == 12
    assert reloaded.single_batch_workers == 3


def test_resolve_memory_plan_uses_manual_stage_tuning(monkeypatch):
    monkeypatch.setattr(
        roop.globals,
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
    monkeypatch.setattr("roop.memory.get_available_ram_gb", lambda: 24.0)
    monkeypatch.setattr("roop.memory.get_available_vram_gb", lambda: 10.0)

    plan = resolve_memory_plan(1920, 1080)

    assert plan["chunk_size"] == 96
    assert plan["prefetch_frames"] == 96
    assert plan["swap_batch_size"] == 48
    assert plan["mask_batch_size"] == 96
    assert plan["enhance_batch_size"] == 12
    assert plan["single_batch_workers"] == 3
    assert plan["detect_pack_frame_count"] == 320
    assert "single-batch workers=3" in describe_memory_plan(plan)
