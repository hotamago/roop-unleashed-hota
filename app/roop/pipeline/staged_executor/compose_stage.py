import os
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from threading import local

import cv2

import roop.config.globals
import roop.media.ffmpeg_ops as ffmpeg
import roop.utils as util
from roop.media.ffmpeg_writer import FFMPEG_VideoWriter
from roop.media.video_io import open_video_capture, resolve_video_writer_config
from roop.pipeline.batch_executor import ProcessMgr, eNoFaceAction
from .cache import write_json

try:
    from roop.media.stream_writer import StreamWriter
except Exception:
    StreamWriter = None


def get_composite_progress_detail():
    return "Streaming source decode + cached face compositing + video encode"


def get_compose_worker_count(executor):
    requested_workers = getattr(roop.config.globals.CFG, "max_threads", 1)
    try:
        resolved_workers = int(requested_workers)
    except (TypeError, ValueError):
        resolved_workers = 1
    resolved_workers = max(1, min(resolved_workers, 8))
    if roop.config.globals.no_face_action == eNoFaceAction.USE_LAST_SWAPPED:
        return 1
    return resolved_workers


def should_use_cached_fallback_fast_path():
    return roop.config.globals.no_face_action in (
        eNoFaceAction.USE_ORIGINAL_FRAME,
        eNoFaceAction.USE_LAST_SWAPPED,
        eNoFaceAction.SKIP_FRAME,
    )


def can_direct_encode_without_processing(executor, entry):
    return (
        executor.output_method == "File"
        and not executor.swap_enabled
        and executor.mask_name is None
        and executor.enhancer_name is None
        and not util.has_extension(entry.filename, ["gif"])
        and bool(getattr(entry, "finalname", None))
    )


def ensure_direct_video_output(executor, entry, index, frame_count, endframe):
    destination = util.replace_template(entry.finalname, index=index)
    Path(os.path.dirname(destination)).mkdir(parents=True, exist_ok=True)
    executor.update_progress(
        "composite",
        detail="Direct ffmpeg video encode without processing stages",
        step_completed=0,
        step_total=frame_count,
        step_unit="frames",
        force_log=True,
    )
    ffmpeg.transcode_video_range(
        entry.filename,
        destination,
        entry.startframe,
        endframe,
        entry.fps or util.detect_fps(entry.filename),
        include_audio=not roop.config.globals.skip_audio,
    )
    executor.completed_units += frame_count
    executor.update_progress(
        "mux",
        detail="Completed direct video encode",
        step_completed=frame_count,
        step_total=frame_count,
        step_unit="frames",
        force_log=True,
    )


def get_composite_segment_dir(intermediate_video):
    return intermediate_video.parent / "composite_segments"


def get_composite_segment_key(pack_data):
    return f"{int(pack_data['start_sequence']):06d}_{int(pack_data['end_sequence']):06d}"


def get_composite_segment_path(intermediate_video, pack_data):
    return get_composite_segment_dir(intermediate_video) / f"{get_composite_segment_key(pack_data)}.mp4"


def read_last_video_frame(video_path):
    capture = open_video_capture(str(video_path))
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count > 1:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        ok, frame = capture.read()
        if ok and frame is not None:
            return frame
        return None
    finally:
        capture.release()


def create_stage_video_writer(filename, size, fps, codec, crf, ffmpeg_params, quality_args):
    return FFMPEG_VideoWriter(
        filename,
        size,
        fps,
        codec=codec,
        crf=crf,
        ffmpeg_params=ffmpeg_params,
        quality_args=quality_args,
    )


def prepare_fallback_mgr(executor, fallback_mgr=None, last_result_frame=None):
    if fallback_mgr is None:
        fallback_mgr = executor.get_fallback_mgr()
    if fallback_mgr.last_swapped_frame is None and last_result_frame is not None:
        fallback_mgr.last_swapped_frame = last_result_frame.copy()
        fallback_mgr.num_frames_no_face = 0
    return fallback_mgr


def resolve_cached_fallback_frame(executor, frame, fallback_mgr=None, last_result_frame=None):
    no_face_action = roop.config.globals.no_face_action
    if no_face_action == eNoFaceAction.USE_ORIGINAL_FRAME:
        return frame
    if no_face_action == eNoFaceAction.SKIP_FRAME:
        return None
    fallback_mgr = prepare_fallback_mgr(executor, fallback_mgr, last_result_frame)
    if no_face_action == eNoFaceAction.USE_LAST_SWAPPED:
        max_reuse_frame = max(0, int(getattr(getattr(fallback_mgr, "options", None), "max_num_reuse_frame", 0)))
        if fallback_mgr.last_swapped_frame is not None and fallback_mgr.num_frames_no_face < max_reuse_frame:
            fallback_mgr.num_frames_no_face += 1
            return fallback_mgr.last_swapped_frame.copy()
        return frame
    return fallback_mgr.process_frame(frame)


def compose_frame_from_cache(executor, compose_mgr, frame, frame_meta, input_cache, enhance_cache, fallback_mgr=None, last_result_frame=None):
    if frame_meta["fallback"]:
        return resolve_cached_fallback_frame(executor, frame, fallback_mgr, last_result_frame)

    result = frame.copy()
    for task_meta in frame_meta["tasks"]:
        fake_frame = input_cache[task_meta["cache_key"]]
        enhanced_frame = enhance_cache.get(task_meta["cache_key"]) if executor.enhancer_name is not None else None
        result = compose_mgr.compose_task(result, task_meta, fake_frame, enhanced_frame)
    if fallback_mgr is not None and result is not None:
        fallback_mgr.last_swapped_frame = result.copy()
        fallback_mgr.num_frames_no_face = 0
    return result


def ensure_full_compose_stage(executor, entry, endframe, fps, detect_dir, swap_dir, mask_dir, enhance_dir, intermediate_video, stages, manifest, memory_plan):
    frame_count = manifest.get("frame_count", max(endframe - entry.startframe, 1))
    composite_state = manifest.setdefault("composite_state", {})
    completed_frames = int(composite_state.get("completed_frames", 0) or 0)
    segment_states = composite_state.setdefault("segments", {})
    pack_list = list(executor.iter_detect_packs(detect_dir) or [])
    expected_segment_keys = [get_composite_segment_key(pack_data) for pack_data in pack_list]
    segment_dir = get_composite_segment_dir(intermediate_video)

    if stages["composite"] and intermediate_video.exists() and completed_frames >= frame_count:
        executor.completed_units += frame_count
        executor.update_progress("composite", detail="Reusing encoded composite video cache", step_completed=frame_count, step_total=frame_count, step_unit="frames", rate_enabled=False, force_log=True)
        return

    intermediate_video.unlink(missing_ok=True)
    segment_dir.mkdir(parents=True, exist_ok=True)
    for segment_path in segment_dir.glob("*.mp4"):
        if segment_path.stem not in expected_segment_keys:
            segment_path.unlink(missing_ok=True)

    completed_frames = 0
    for pack_data in pack_list:
        segment_key = get_composite_segment_key(pack_data)
        segment_path = get_composite_segment_path(intermediate_video, pack_data)
        segment_state = segment_states.get(segment_key) or {}
        segment_frame_count = len(pack_data.get("frames", []))
        if segment_path.exists() and segment_state.get("completed"):
            segment_state["frame_count"] = segment_frame_count
            completed_frames += segment_frame_count
        else:
            if segment_path.exists():
                segment_path.unlink(missing_ok=True)
            segment_state = {"completed": False, "frame_count": segment_frame_count}
        segment_states[segment_key] = segment_state

    stages["composite"] = False
    composite_state["completed_frames"] = completed_frames
    composite_state["frame_count"] = frame_count
    write_json(intermediate_video.parent / "manifest.json", manifest)
    compose_mgr = ProcessMgr(None)
    compose_mgr.initialize(roop.config.globals.INPUT_FACESETS, roop.config.globals.TARGET_FACES, executor.build_stage_options([]))
    fallback_mgr = None
    last_result_frame = None
    processed_frames = completed_frames
    initial_completed_frames = completed_frames
    computed_frames = 0
    last_progress_emit_at = 0.0
    progress_emit_frames = 8
    cap = open_video_capture(entry.filename)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    writer_config = resolve_video_writer_config(roop.config.globals.video_encoder, roop.config.globals.video_quality)

    def load_pack_state(pack_data):
        if pack_data is None:
            return None, {}, {}, {}
        frame_lookup = {frame_meta["frame_number"]: frame_meta for frame_meta in pack_data["frames"]}
        task_keys = [
            task_meta["cache_key"]
            for frame_meta in pack_data["frames"]
            for task_meta in frame_meta.get("tasks", [])
        ]
        input_cache = executor.read_stage_cache_keys(
            executor.get_stage_pack_path(mask_dir if executor.mask_name else swap_dir, pack_data["start_sequence"], pack_data["end_sequence"]),
            task_keys,
        )
        enhance_cache = (
            executor.read_stage_cache_keys(
                executor.get_stage_pack_path(enhance_dir, pack_data["start_sequence"], pack_data["end_sequence"]),
                task_keys,
            )
            if executor.enhancer_name is not None
            else {}
        )
        return pack_data["frames"][-1]["frame_number"], frame_lookup, input_cache, enhance_cache

    try:
        from .video_iter import iter_video_chunk

        compose_worker_count = get_compose_worker_count(executor)

        def emit_progress(force=False):
            nonlocal last_progress_emit_at
            if not force:
                if processed_frames <= 0:
                    return
                if processed_frames < frame_count and processed_frames % progress_emit_frames != 0:
                    now = time.time()
                    if (now - last_progress_emit_at) < 0.25:
                        return
            last_progress_emit_at = time.time()
            executor.update_progress("composite", detail=get_composite_progress_detail(), step_completed=processed_frames, step_total=frame_count, step_unit="frames", rate_completed=computed_frames, rate_total=max(frame_count - initial_completed_frames, computed_frames))

        def cache_last_result_frame(result):
            nonlocal last_result_frame, fallback_mgr
            if result is None:
                return
            last_result_frame = result
            if fallback_mgr is not None:
                fallback_mgr.last_swapped_frame = result.copy()
                fallback_mgr.num_frames_no_face = 0

        if processed_frames and roop.config.globals.no_face_action == eNoFaceAction.USE_LAST_SWAPPED:
            for completed_pack in reversed(pack_list):
                completed_segment_path = get_composite_segment_path(intermediate_video, completed_pack)
                completed_segment_state = segment_states.get(get_composite_segment_key(completed_pack)) or {}
                if completed_segment_path.exists() and completed_segment_state.get("completed"):
                    last_result_frame = read_last_video_frame(completed_segment_path)
                    if last_result_frame is not None:
                        break

        for pack_data in pack_list:
            segment_key = get_composite_segment_key(pack_data)
            segment_path = get_composite_segment_path(intermediate_video, pack_data)
            segment_state = segment_states.setdefault(segment_key, {})
            segment_frame_count = len(pack_data.get("frames", []))
            if segment_path.exists() and segment_state.get("completed"):
                executor.completed_units += segment_frame_count
                executor.update_progress("composite", detail="Reusing encoded composite segment cache", step_completed=processed_frames, step_total=frame_count, step_unit="frames", rate_enabled=False, force_log=True)
                continue

            segment_path.unlink(missing_ok=True)
            current_pack_end, frame_lookup, input_cache, enhance_cache = load_pack_state(pack_data)
            pack_frames = pack_data.get("frames", [])
            if not pack_frames:
                segment_state["completed"] = True
                segment_state["frame_count"] = 0
                write_json(intermediate_video.parent / "manifest.json", manifest)
                continue

            pack_start_frame = int(pack_frames[0]["frame_number"])
            pack_end_frame = int(pack_frames[-1]["frame_number"]) + 1
            segment_writer = create_stage_video_writer(
                str(segment_path),
                (width, height),
                fps,
                codec=writer_config["codec"],
                crf=roop.config.globals.video_quality,
                ffmpeg_params=writer_config["ffmpeg_params"],
                quality_args=writer_config["quality_args"],
            )
            pack_processed_frames = 0
            checkpoint_completed_frames = processed_frames

            def write_composed_frames(results):
                nonlocal processed_frames, pack_processed_frames, computed_frames
                valid_results = [result for result in results if result is not None]
                if valid_results:
                    write_many = getattr(segment_writer, "write_frames", None)
                    if callable(write_many):
                        write_many(valid_results)
                    else:
                        for result in valid_results:
                            segment_writer.write_frame(result)
                for result in results:
                    executor.completed_units += 1
                    processed_frames += 1
                    computed_frames += 1
                    pack_processed_frames += 1
                    cache_last_result_frame(result)
                composite_state["completed_frames"] = processed_frames
                emit_progress(force=processed_frames >= frame_count)

            def write_composed_frame(result):
                write_composed_frames([result])

            try:
                if compose_worker_count == 1:
                    for frame_number, frame in iter_video_chunk(entry.filename, pack_start_frame, pack_end_frame, memory_plan["prefetch_frames"]):
                        frame_meta = frame_lookup.get(frame_number, {"tasks": [], "fallback": True})
                        if frame_meta["fallback"]:
                            if roop.config.globals.no_face_action == eNoFaceAction.USE_LAST_SWAPPED:
                                fallback_mgr = prepare_fallback_mgr(executor, fallback_mgr, last_result_frame)
                            elif not should_use_cached_fallback_fast_path():
                                fallback_mgr = executor.get_fallback_mgr()
                        result = compose_frame_from_cache(
                            executor,
                            compose_mgr,
                            frame,
                            frame_meta,
                            input_cache,
                            enhance_cache,
                            fallback_mgr,
                            last_result_frame,
                        )
                        write_composed_frame(result)
                else:
                    worker_state = local()
                    worker_managers = []
                    pending_futures = {}
                    buffered_results = {}
                    next_result_index = 0
                    max_in_flight = max(compose_worker_count * 2, 1)

                    def get_thread_compose_mgr():
                        thread_compose_mgr = getattr(worker_state, "compose_mgr", None)
                        if thread_compose_mgr is None:
                            thread_compose_mgr = ProcessMgr(None)
                            thread_compose_mgr.initialize(roop.config.globals.INPUT_FACESETS, roop.config.globals.TARGET_FACES, executor.build_stage_options([]))
                            worker_state.compose_mgr = thread_compose_mgr
                            worker_managers.append(thread_compose_mgr)
                        return thread_compose_mgr

                    def compose_non_fallback_frame(frame, frame_meta, source_cache, enhanced_cache):
                        thread_compose_mgr = get_thread_compose_mgr()
                        return compose_frame_from_cache(executor, thread_compose_mgr, frame, frame_meta, source_cache, enhanced_cache)

                    def flush_done_futures(done_futures):
                        nonlocal next_result_index
                        for done_future in done_futures:
                            frame_index = pending_futures.pop(done_future)
                            buffered_results[frame_index] = done_future.result()
                        while next_result_index in buffered_results:
                            result = buffered_results.pop(next_result_index)
                            write_composed_frame(result)
                            next_result_index += 1

                    def flush_all_pending():
                        while pending_futures:
                            done_futures, _ = wait(set(pending_futures), return_when=FIRST_COMPLETED)
                            flush_done_futures(done_futures)

                    with ThreadPoolExecutor(max_workers=compose_worker_count, thread_name_prefix="staged_compose") as pool:
                        frame_index = 0
                        for frame_number, frame in iter_video_chunk(entry.filename, pack_start_frame, pack_end_frame, memory_plan["prefetch_frames"]):
                            frame_meta = frame_lookup.get(frame_number, {"tasks": [], "fallback": True})
                            if frame_meta["fallback"]:
                                flush_all_pending()
                                if roop.config.globals.no_face_action == eNoFaceAction.USE_LAST_SWAPPED:
                                    fallback_mgr = prepare_fallback_mgr(executor, fallback_mgr, last_result_frame)
                                elif not should_use_cached_fallback_fast_path():
                                    fallback_mgr = executor.get_fallback_mgr()
                                result = compose_frame_from_cache(
                                    executor,
                                    compose_mgr,
                                    frame,
                                    frame_meta,
                                    input_cache,
                                    enhance_cache,
                                    fallback_mgr,
                                    last_result_frame,
                                )
                                write_composed_frame(result)
                                frame_index += 1
                                next_result_index = frame_index
                                continue

                            if len(pending_futures) >= max_in_flight:
                                done_futures, _ = wait(set(pending_futures), return_when=FIRST_COMPLETED)
                                flush_done_futures(done_futures)

                            future = pool.submit(compose_non_fallback_frame, frame, frame_meta, input_cache, enhance_cache)
                            pending_futures[future] = frame_index
                            frame_index += 1

                        flush_all_pending()

                    for worker_manager in worker_managers:
                        worker_manager.release_resources()
            finally:
                segment_writer.close()

            segment_completed = roop.config.globals.processing and pack_processed_frames >= segment_frame_count and segment_path.exists()
            if not segment_completed:
                segment_state["completed"] = False
                segment_state["frame_count"] = segment_frame_count
                processed_frames = checkpoint_completed_frames
                composite_state["completed_frames"] = checkpoint_completed_frames
                segment_path.unlink(missing_ok=True)
                write_json(intermediate_video.parent / "manifest.json", manifest)
                break

            segment_state["completed"] = True
            segment_state["frame_count"] = segment_frame_count
            composite_state["completed_frames"] = processed_frames
            write_json(intermediate_video.parent / "manifest.json", manifest)

        emit_progress(force=True)
    finally:
        compose_mgr.release_resources()

    completed_segment_paths = []
    for pack_data in pack_list:
        segment_key = get_composite_segment_key(pack_data)
        segment_path = get_composite_segment_path(intermediate_video, pack_data)
        segment_state = segment_states.get(segment_key) or {}
        if segment_state.get("completed") and segment_path.exists():
            completed_segment_paths.append(str(segment_path))

    completed_successfully = roop.config.globals.processing and processed_frames >= frame_count and len(completed_segment_paths) == len(pack_list)
    if not completed_successfully:
        stages["composite"] = False
        composite_state["completed_frames"] = sum(
            int((segment_states.get(get_composite_segment_key(pack_data)) or {}).get("frame_count", 0))
            for pack_data in pack_list
            if (segment_states.get(get_composite_segment_key(pack_data)) or {}).get("completed")
        )
        intermediate_video.unlink(missing_ok=True)
    else:
        intermediate_video.unlink(missing_ok=True)
        if len(completed_segment_paths) == 1:
            shutil.copyfile(completed_segment_paths[0], intermediate_video)
        else:
            ffmpeg.join_videos(completed_segment_paths, str(intermediate_video), True)
        stages["composite"] = True
        composite_state["completed_frames"] = frame_count
    write_json(intermediate_video.parent / "manifest.json", manifest)


def ensure_full_encode_stage(executor, entry, index, intermediate_video, frame_count, endframe):
    destination = util.replace_template(entry.finalname, index=index)
    Path(os.path.dirname(destination)).mkdir(parents=True, exist_ok=True)
    if util.has_extension(entry.filename, ["gif"]):
        executor.update_progress("mux", detail="Creating final GIF", step_completed=frame_count, step_total=frame_count, step_unit="frames", force_log=True)
        ffmpeg.create_gif_from_video(str(intermediate_video), destination)
    elif roop.config.globals.skip_audio:
        if os.path.isfile(destination):
            os.remove(destination)
        shutil.move(str(intermediate_video), destination)
    else:
        executor.update_progress("mux", detail="Restoring source audio", step_completed=frame_count, step_total=frame_count, step_unit="frames", force_log=True)
        ffmpeg.restore_audio(str(intermediate_video), entry.filename, entry.startframe, endframe, destination)
        if intermediate_video.exists() and os.path.isfile(destination):
            os.remove(str(intermediate_video))


def compose_image_from_cache(executor, image, job_dir, frame_meta):
    if frame_meta["fallback"]:
        fallback_mgr = executor.get_fallback_mgr() if not should_use_cached_fallback_fast_path() else None
        return resolve_cached_fallback_frame(executor, image, fallback_mgr)
    compose_mgr = ProcessMgr(None)
    compose_mgr.initialize(roop.config.globals.INPUT_FACESETS, roop.config.globals.TARGET_FACES, executor.build_stage_options([]))
    task_keys = [task_meta["cache_key"] for task_meta in frame_meta["tasks"]]
    input_cache = executor.read_stage_cache_keys(executor.get_stage_cache_path(job_dir / ("mask" if executor.mask_name else "swap")), task_keys)
    enhance_cache = executor.read_stage_cache_keys(executor.get_stage_cache_path(job_dir / "enhance"), task_keys) if executor.enhancer_name is not None else {}
    result = image.copy()
    try:
        for task_meta in frame_meta["tasks"]:
            fake_frame = input_cache[task_meta["cache_key"]]
            enhanced_frame = enhance_cache.get(task_meta["cache_key"]) if executor.enhancer_name is not None else None
            result = compose_mgr.compose_task(result, task_meta, fake_frame, enhanced_frame)
    finally:
        compose_mgr.release_resources()
    return result


def compose_chunk(executor, entry, chunk_dir, chunk_meta, chunk_state, memory_plan, chunk_video):
    if chunk_state["stages"]["composite"] and chunk_video.exists():
        return
    compose_mgr = ProcessMgr(None)
    compose_mgr.initialize(roop.config.globals.INPUT_FACESETS, roop.config.globals.TARGET_FACES, executor.build_stage_options([]))
    frame_lookup = {frame_meta["frame_number"]: frame_meta for frame_meta in chunk_meta["frames"]}
    task_keys = [task_meta["cache_key"] for frame_meta in chunk_meta["frames"] for task_meta in frame_meta.get("tasks", [])]
    input_cache = executor.read_stage_cache_keys(executor.get_stage_cache_path(chunk_dir / ("mask" if executor.mask_name else "swap")), task_keys)
    enhance_cache = executor.read_stage_cache_keys(executor.get_stage_cache_path(chunk_dir / "enhance"), task_keys) if executor.enhancer_name is not None else {}
    cap = open_video_capture(entry.filename)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    output_to_file = executor.output_method != "Virtual Camera"
    output_to_cam = executor.output_method in ("Virtual Camera", "Both") and StreamWriter is not None
    writer_config = resolve_video_writer_config(roop.config.globals.video_encoder, roop.config.globals.video_quality)
    writer = (
        FFMPEG_VideoWriter(
            str(chunk_video),
            (width, height),
            entry.fps or util.detect_fps(entry.filename),
            codec=writer_config["codec"],
            crf=roop.config.globals.video_quality,
            ffmpeg_params=writer_config["ffmpeg_params"],
            quality_args=writer_config["quality_args"],
        )
        if output_to_file
        else None
    )
    stream = StreamWriter((width, height), int(entry.fps or util.detect_fps(entry.filename))) if output_to_cam else None
    fallback_mgr = None
    last_result_frame = None

    try:
        from .video_iter import iter_video_chunk

        processed_frames = 0
        total_frames = max(chunk_meta["end"] - chunk_meta["start"], 1)
        for frame_number, frame in iter_video_chunk(entry.filename, chunk_meta["start"], chunk_meta["end"], memory_plan["prefetch_frames"]):
            frame_meta = frame_lookup.get(frame_number, {"tasks": [], "fallback": True})
            if frame_meta["fallback"]:
                if roop.config.globals.no_face_action == eNoFaceAction.USE_LAST_SWAPPED:
                    fallback_mgr = prepare_fallback_mgr(executor, fallback_mgr, last_result_frame)
                elif not should_use_cached_fallback_fast_path():
                    fallback_mgr = executor.get_fallback_mgr()
                result = resolve_cached_fallback_frame(executor, frame, fallback_mgr, last_result_frame)
            else:
                result = frame.copy()
                for task_meta in frame_meta["tasks"]:
                    fake_frame = input_cache[task_meta["cache_key"]]
                    enhanced_frame = enhance_cache.get(task_meta["cache_key"]) if executor.enhancer_name is not None else None
                    result = compose_mgr.compose_task(result, task_meta, fake_frame, enhanced_frame)
                fallback_mgr = executor.fallback_mgr
                if fallback_mgr is not None and result is not None:
                    fallback_mgr.last_swapped_frame = result.copy()
                    fallback_mgr.num_frames_no_face = 0
            if result is not None:
                if writer is not None:
                    writer.write_frame(result)
                if stream is not None:
                    stream.WriteToStream(result)
                last_result_frame = result.copy()
            executor.completed_units += 1
            processed_frames += 1
            executor.update_progress("composite", detail="Compositing output frames", step_completed=processed_frames, step_total=total_frames, step_unit="frames")
    finally:
        compose_mgr.release_resources()
        if writer is not None:
            writer.close()
        if stream is not None:
            stream.Close()
    chunk_state["stages"]["composite"] = True


def should_skip_completed_output(_executor, entry, manifest):
    return manifest.get("status") == "completed" and bool(getattr(entry, "finalname", None)) and os.path.isfile(entry.finalname)


__all__ = [
    "can_direct_encode_without_processing",
    "compose_chunk",
    "compose_frame_from_cache",
    "compose_image_from_cache",
    "create_stage_video_writer",
    "ensure_direct_video_output",
    "ensure_full_compose_stage",
    "ensure_full_encode_stage",
    "get_composite_progress_detail",
    "get_composite_segment_dir",
    "get_composite_segment_key",
    "get_composite_segment_path",
    "get_compose_worker_count",
    "read_last_video_frame",
    "should_skip_completed_output",
]
