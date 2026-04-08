# Refactor Task Checklist

Status: completed on 2026-04-08

## Phase 1: Create Directory Structure
- [x] Create all new directories with `__init__.py` files
- [x] Create `roop/config/`, `roop/core/`, `roop/face/`, `roop/pipeline/staged_executor/`, `roop/media/`, `roop/onnx/`, `roop/memory/`, `roop/progress/`, `roop/utils/`

## Phase 2: Config & Types Layer
- [x] Create `roop/config/globals.py` (cleaned globals)
- [x] Create `roop/config/types.py` (merged typing + FaceSet)
- [x] Create `roop/config/settings.py` (moved from `app/settings.py`)

## Phase 3: Utils, Media, ONNX, Memory, Progress
- [x] Create `roop/utils/io.py` (file helpers from `utilities.py`)
- [x] Create `roop/utils/download.py` (`conditional_download`)
- [x] Create `roop/utils/platform.py` (platform detection)
- [x] Create `roop/utils/cache_paths.py` (from `cache_paths.py`)
- [x] Create `roop/utils/template_parser.py` (from `template_parser.py`)
- [x] Create `roop/utils/vr.py` (from `vr_util.py`)
- [x] Create `roop/media/capturer.py` (from `capturer.py`)
- [x] Create `roop/media/ffmpeg_ops.py` (from `util_ffmpeg.py`)
- [x] Create `roop/media/ffmpeg_writer.py` (from `ffmpeg_writer.py`)
- [x] Create `roop/media/video_io.py` (from `video_io.py`)
- [x] Create `roop/media/stream_writer.py` (from `StreamWriter.py`)
- [x] Create `roop/onnx/batch.py` (from `onnx_batch.py`)
- [x] Create `roop/onnx/runtime.py` (from `onnx_runtime.py`)
- [x] Create `roop/memory/planner.py` (from `memory.py`)
- [x] Create `roop/progress/status.py` (from `progress_status.py`)

## Phase 4: Face Module
- [x] Create `roop/face/analyser.py`
- [x] Create `roop/face/detection.py`
- [x] Create `roop/face/alignment.py`
- [x] Create `roop/face/rotation.py`
- [x] Create `roop/face/geometry.py`

## Phase 5: Pipeline Module
- [x] Create `roop/pipeline/options.py`
- [x] Create `roop/pipeline/entry.py`
- [x] Create `roop/pipeline/faceset.py`
- [x] Create `roop/pipeline/face_targeting.py`
- [x] Create `roop/pipeline/face_serializer.py`
- [x] Create `roop/pipeline/swap_stage.py`
- [x] Create `roop/pipeline/mask_stage.py`
- [x] Create `roop/pipeline/enhance_stage.py`
- [x] Create `roop/pipeline/compose_stage.py`
- [x] Create `roop/pipeline/batch_executor.py`
- [x] Create `roop/processors/base.py`

## Phase 6: Staged Executor Decomposition
- [x] Create `roop/pipeline/staged_executor/cache.py`
- [x] Create `roop/pipeline/staged_executor/video_cache.py`
- [x] Create `roop/pipeline/staged_executor/video_iter.py`
- [x] Create `roop/pipeline/staged_executor/progress.py`
- [x] Create `roop/pipeline/staged_executor/detect_stage.py`
- [x] Create `roop/pipeline/staged_executor/swap_stage.py`
- [x] Create `roop/pipeline/staged_executor/mask_stage.py`
- [x] Create `roop/pipeline/staged_executor/enhance_stage.py`
- [x] Create `roop/pipeline/staged_executor/compose_stage.py`
- [x] Create `roop/pipeline/staged_executor/chunk_processor.py`
- [x] Create `roop/pipeline/staged_executor/executor.py`

## Phase 7: Core Application
- [x] Create `roop/core/providers.py`
- [x] Create `roop/core/resources.py`
- [x] Create `roop/core/app.py`

## Phase 8: Update External Consumers & Cleanup
- [x] Update `app/run.py` imports
- [x] Update `app/ui/main.py` imports
- [x] Update `app/ui/tabs/faceswap_tab.py` imports
- [x] Update `app/ui/tabs/facemgr_tab.py` imports
- [x] Update `app/ui/tabs/settings_tab.py` imports
- [x] Update `app/ui/tabs/extras_tab.py` imports
- [x] Update `app/ui/globals.py` imports
- [x] Update processor imports
- [x] Update test imports
- [x] Delete old files
- [x] Clean up `__pycache__` directories

## Phase 9: Verification
- [x] Run import validation
- [x] Run test suite
- [x] Verify UI launch

## Verification Notes
- Import validation passed for `roop.core.app`, `roop.pipeline.staged_executor.executor`, and `roop.face.detection`.
- Full test suite passed: `81 passed` with `python -m pytest app/tests -q`.
- Gradio UI launch smoke passed on `http://127.0.0.1:7867` and was cleanly closed after startup.
