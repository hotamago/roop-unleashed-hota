from roop.pipeline.staged_executor.executor import StagedBatchExecutor

ensure_chunk_detect = StagedBatchExecutor.ensure_chunk_detect
ensure_detect_cache = StagedBatchExecutor.ensure_detect_cache
ensure_full_detect_stage = StagedBatchExecutor.ensure_full_detect_stage

__all__ = [
    "ensure_chunk_detect",
    "ensure_detect_cache",
    "ensure_full_detect_stage",
]

