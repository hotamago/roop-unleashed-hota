import json
from pathlib import Path

import cv2
import numpy as np

from roop.pipeline.staged_executor.executor import normalize_cache_image


class VideoStageCache:
    """
    Compatibility-oriented stage cache with random-access reads.

    The on-disk layout matches the planned `*.mp4` + `*.idx.bin` naming, while
    storing PNG-compressed frames for deterministic, dependency-light tests.
    """

    def __init__(self, image_format=".png"):
        self.image_format = image_format

    def _resolve_paths(self, cache_path):
        base_path = Path(cache_path)
        if base_path.suffix.lower() == ".idx.bin":
            index_path = base_path
            video_path = base_path.with_suffix("").with_suffix(".mp4")
        elif base_path.suffix.lower() == ".mp4":
            video_path = base_path
            index_path = base_path.with_suffix(".idx.bin")
        else:
            video_path = base_path.with_suffix(".mp4")
            index_path = base_path.with_suffix(".idx.bin")
        return video_path, index_path

    def write(self, cache_path, cache_map):
        video_path, index_path = self._resolve_paths(cache_path)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        index = {}
        offset = 0
        with video_path.open("wb") as handle:
            for cache_key in sorted(cache_map):
                image = normalize_cache_image(cache_map[cache_key])
                ok, encoded = cv2.imencode(self.image_format, image)
                if not ok:
                    raise ValueError(f"failed to encode cache image for {cache_key}")
                payload = encoded.tobytes()
                handle.write(payload)
                index[cache_key] = {
                    "offset": offset,
                    "length": len(payload),
                    "shape": list(image.shape),
                }
                offset += len(payload)

        index_path.write_text(json.dumps(index, sort_keys=True), encoding="utf-8")
        return video_path

    def _read_index(self, index_path):
        if not index_path.exists():
            return {}
        return json.loads(index_path.read_text(encoding="utf-8"))

    def _decode_entry(self, video_path, entry):
        with video_path.open("rb") as handle:
            handle.seek(int(entry["offset"]))
            encoded = handle.read(int(entry["length"]))
        image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"failed to decode cache payload from {video_path}")
        return image

    def read(self, cache_path):
        video_path, index_path = self._resolve_paths(cache_path)
        index = self._read_index(index_path)
        return {cache_key: self._decode_entry(video_path, entry) for cache_key, entry in index.items()}

    def read_keys(self, cache_path, keys):
        video_path, index_path = self._resolve_paths(cache_path)
        index = self._read_index(index_path)
        selected = {}
        for cache_key in keys:
            entry = index.get(cache_key)
            if entry is None:
                continue
            selected[cache_key] = self._decode_entry(video_path, entry)
        return selected


__all__ = ["VideoStageCache"]

