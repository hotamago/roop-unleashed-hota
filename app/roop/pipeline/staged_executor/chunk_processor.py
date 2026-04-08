from roop.pipeline.staged_executor.executor import StagedBatchExecutor

flatten_tasks = StagedBatchExecutor.flatten_tasks
iter_chunk_source_frames_with_meta = StagedBatchExecutor.iter_chunk_source_frames_with_meta

__all__ = ["flatten_tasks", "iter_chunk_source_frames_with_meta"]

