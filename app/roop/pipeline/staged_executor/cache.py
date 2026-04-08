from roop.pipeline.staged_executor.executor import (
    chunked,
    get_entry_file_identity,
    get_entry_job_key,
    get_entry_job_relpath,
    get_entry_signature,
    get_jobs_root,
    get_staged_cache_manifest_signature,
    get_staged_cache_options_snapshot,
    hash_facesets,
    hash_numpy,
    hash_target_faces,
    json_dumps,
    make_json_safe,
    normalize_cache_image,
    read_cache_blob,
    read_json,
    sanitize_job_path_segment,
    write_cache_blob,
    write_json,
)
from roop.pipeline.staged_executor.executor import StagedBatchExecutor as _LegacyExecutor

cleanup_job_dir = _LegacyExecutor.cleanup_job_dir
prepare_job = _LegacyExecutor.prepare_job

__all__ = [
    "chunked",
    "cleanup_job_dir",
    "get_entry_file_identity",
    "get_entry_job_key",
    "get_entry_job_relpath",
    "get_entry_signature",
    "get_jobs_root",
    "get_staged_cache_manifest_signature",
    "get_staged_cache_options_snapshot",
    "hash_facesets",
    "hash_numpy",
    "hash_target_faces",
    "json_dumps",
    "make_json_safe",
    "normalize_cache_image",
    "prepare_job",
    "read_cache_blob",
    "read_json",
    "sanitize_job_path_segment",
    "write_cache_blob",
    "write_json",
]

