from types import SimpleNamespace

import numpy as np

import roop.globals
from roop.ProcessMgr import ProcessMgr


class FakeSingleBatchProcessor:
    supports_parallel_single_batch = True
    batch_size_limit = 1

    def __init__(self, worker_id=0):
        self.worker_id = worker_id
        self.released = False
        self.create_calls = 0

    def CreateWorkerProcessor(self):
        self.create_calls += 1
        return FakeSingleBatchProcessor(worker_id=self.worker_id + self.create_calls)

    def Release(self):
        self.released = True

    def Run(self, source_face, target_face, frame):
        return np.full_like(frame, self.worker_id + 1), 1


def test_process_mgr_parallelizes_single_batch_processors(monkeypatch):
    mgr = ProcessMgr(None)
    mgr.options = SimpleNamespace(num_swap_steps=1, subsample_size=256)
    monkeypatch.setattr(mgr, "get_single_batch_worker_count", lambda processor: 2)
    monkeypatch.setattr(
        mgr,
        "run_prepared_swap_task",
        lambda prepared_task, processor: np.full((1, 1, 3), processor.worker_id + len(prepared_task["cache_key"]), dtype=np.uint8),
    )

    prepared_tasks = [
        {"cache_key": "task_a", "current_frames": [], "input_face": None, "target_face": None},
        {"cache_key": "task_b", "current_frames": [], "input_face": None, "target_face": None},
        {"cache_key": "task_c", "current_frames": [], "input_face": None, "target_face": None},
    ]
    processor = FakeSingleBatchProcessor()

    outputs = mgr.run_swap_tasks_parallel_single_batch(prepared_tasks, processor)

    assert set(outputs.keys()) == {"task_a", "task_b", "task_c"}
    assert all(isinstance(value, np.ndarray) for value in outputs.values())
    assert processor.released is False


def test_process_mgr_reuses_single_batch_worker_sessions_across_calls(monkeypatch):
    mgr = ProcessMgr(None)
    mgr.options = SimpleNamespace(num_swap_steps=1, subsample_size=256)
    monkeypatch.setattr(mgr, "get_single_batch_worker_count", lambda processor: 3)
    monkeypatch.setattr(
        mgr,
        "run_prepared_swap_task",
        lambda prepared_task, processor: np.full((1, 1, 3), processor.worker_id + 1, dtype=np.uint8),
    )

    prepared_tasks = [
        {"cache_key": "task_a", "current_frames": [], "input_face": None, "target_face": None},
        {"cache_key": "task_b", "current_frames": [], "input_face": None, "target_face": None},
    ]
    processor = FakeSingleBatchProcessor()

    mgr.run_swap_tasks_parallel_single_batch(prepared_tasks, processor)
    mgr.run_swap_tasks_parallel_single_batch(prepared_tasks, processor)

    assert processor.create_calls == 2
    assert len(mgr.single_batch_worker_pools[id(processor)]["workers"]) == 3

    mgr.processors = [processor]
    mgr.release_resources()

    assert processor.released is True


def test_process_mgr_single_batch_worker_count_uses_memory_plan(monkeypatch):
    mgr = ProcessMgr(None)
    processor = FakeSingleBatchProcessor()
    monkeypatch.setattr(roop.globals, "active_memory_plan", {"single_batch_workers": 3}, raising=False)

    assert mgr.get_single_batch_worker_count(processor) == 3


def test_process_mgr_single_batch_worker_count_caps_gpu_without_memory_plan(monkeypatch):
    mgr = ProcessMgr(None)
    processor = FakeSingleBatchProcessor()
    monkeypatch.setattr(roop.globals, "active_memory_plan", None, raising=False)
    monkeypatch.setattr(roop.globals, "CFG", SimpleNamespace(single_batch_workers=4), raising=False)
    monkeypatch.setattr("roop.ProcessMgr.resolve_single_batch_workers", lambda configured_workers: (1, configured_workers, "GPU-safe cap"))

    assert mgr.get_single_batch_worker_count(processor) == 1


def test_process_mgr_parallelizes_single_batch_enhancers(monkeypatch):
    mgr = ProcessMgr(None)
    mgr.input_face_datas = [SimpleNamespace(faces=[])]
    mgr.deserialize_face = lambda payload: payload
    monkeypatch.setattr(mgr, "get_single_batch_worker_count", lambda processor: 2)

    tasks = [
        {"cache_key": "task_a", "input_index": 0, "target_face": {"id": "a"}},
        {"cache_key": "task_b", "input_index": 0, "target_face": {"id": "b"}},
    ]
    current_frames = [
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.zeros((2, 2, 3), dtype=np.uint8),
    ]

    outputs = mgr.run_enhance_tasks_batch(tasks, current_frames, FakeSingleBatchProcessor(), batch_size=8)

    assert set(outputs.keys()) == {"task_a", "task_b"}
    assert all(isinstance(value, np.ndarray) for value in outputs.values())
