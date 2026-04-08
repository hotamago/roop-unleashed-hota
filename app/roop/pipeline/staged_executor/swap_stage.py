from roop.pipeline.staged_executor.executor import StagedBatchExecutor

ensure_full_swap_stage = StagedBatchExecutor.ensure_full_swap_stage
ensure_swap_stage = StagedBatchExecutor.ensure_swap_stage

__all__ = ["ensure_full_swap_stage", "ensure_swap_stage"]

