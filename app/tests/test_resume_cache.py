from pathlib import Path
from types import SimpleNamespace

import roop.globals
import ui.globals
from roop.FaceSet import FaceSet
from roop.ProcessEntry import ProcessEntry
import ui.tabs.faceswap_tab as faceswap_tab


def make_face(mask_offsets=None):
    return SimpleNamespace(mask_offsets=list(mask_offsets or faceswap_tab.default_mask_offsets()))


def test_build_resume_payload_captures_sources_targets_and_settings(monkeypatch):
    source_face_set = FaceSet()
    source_face_set.faces.append(make_face([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    roop.globals.INPUT_FACESETS.append(source_face_set)
    roop.globals.TARGET_FACES.append(SimpleNamespace())
    ui.globals.ui_input_face_refs.append({"type": "image_face", "path": "C:/source.png", "face_index": 0})
    ui.globals.ui_target_face_refs.append({"path": "C:/target.mp4", "frame_number": 12, "face_index": 1})
    faceswap_tab.list_files_process[:] = [ProcessEntry("C:/target.mp4", 5, 25, 30.0)]
    faceswap_tab.selected_preview_index = 0
    faceswap_tab.SELECTED_INPUT_FACE_INDEX = 0
    faceswap_tab.SELECTED_TARGET_FACE_INDEX = 0

    payload = faceswap_tab.build_resume_payload({"output_method": "File", "detection": "Selected face"})

    assert payload["sources"][0]["type"] == "image_face"
    assert payload["sources"][0]["mask_offsets"] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert payload["targets"]["files"][0]["startframe"] == 5
    assert payload["targets"]["selected_faces"][0]["frame_number"] == 12
    assert payload["settings"]["output_method"] == "File"


def test_write_and_read_resume_payload_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(faceswap_tab, "get_resume_cache_root", lambda: str(tmp_path))
    monkeypatch.setattr(faceswap_tab, "list_resume_cache_files", lambda: [], raising=False)
    source_face_set = FaceSet()
    source_face_set.faces.append(make_face())
    roop.globals.INPUT_FACESETS.append(source_face_set)
    ui.globals.ui_input_face_refs.append({"type": "image_face", "path": "C:/source.png", "face_index": 0})
    faceswap_tab.list_files_process[:] = [ProcessEntry("C:/target.mp4", 0, 100, 24.0)]

    payload = faceswap_tab.build_resume_payload({"output_method": "File"})
    resume_path = faceswap_tab.write_resume_payload(payload)
    reloaded = faceswap_tab.read_resume_payload(resume_path)

    assert resume_path.endswith(".json")
    assert reloaded["version"] == faceswap_tab.RESUME_CACHE_VERSION
    assert reloaded["targets"]["files"][0]["filename"].endswith("target.mp4")


def test_write_resume_payload_snapshots_source_files_into_resume_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(faceswap_tab, "get_resume_cache_root", lambda: str(tmp_path))
    monkeypatch.setattr(faceswap_tab, "list_resume_cache_files", lambda: [], raising=False)
    source_path = tmp_path / "gradio" / "source.png"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"source-image")
    source_face_set = FaceSet()
    source_face_set.faces.append(make_face())
    roop.globals.INPUT_FACESETS.append(source_face_set)
    ui.globals.ui_input_face_refs.append({"type": "image_face", "path": str(source_path), "face_index": 0})
    faceswap_tab.list_files_process[:] = [ProcessEntry("C:/target.mp4", 0, 100, 24.0)]

    payload = faceswap_tab.build_resume_payload({"output_method": "File"})
    resume_path = faceswap_tab.write_resume_payload(payload)
    reloaded = faceswap_tab.read_resume_payload(resume_path)

    cached_path = reloaded["sources"][0]["resume_cached_path"]
    assert cached_path.startswith(str(tmp_path))
    assert Path(cached_path).read_bytes() == b"source-image"


def test_restore_input_faces_from_resume_uses_cached_snapshot_when_original_path_is_missing(tmp_path, monkeypatch):
    cached_source = tmp_path / "resume_cache" / "cached-source.png"
    cached_source.parent.mkdir(parents=True, exist_ok=True)
    cached_source.write_bytes(b"cached-image")
    restored_paths = []

    def fake_append_image_source(source_path, source_ref):
        restored_paths.append(source_path)
        face_set = FaceSet()
        face_set.faces.append(make_face(source_ref.get("mask_offsets")))
        roop.globals.INPUT_FACESETS.append(face_set)
        ui.globals.ui_input_face_refs.append(dict(source_ref))

    monkeypatch.setattr(faceswap_tab, "append_image_source", fake_append_image_source)

    faceswap_tab.restore_input_faces_from_resume([
        {
            "type": "image_face",
            "path": str(tmp_path / "missing" / "source.png"),
            "resume_cached_path": str(cached_source),
            "face_index": 0,
            "mask_offsets": list(faceswap_tab.default_mask_offsets()),
        }
    ])

    assert restored_paths == [str(cached_source)]


def test_write_resume_payload_reuses_existing_file_for_equivalent_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(faceswap_tab, "get_resume_cache_root", lambda: str(tmp_path))
    monkeypatch.setattr(faceswap_tab, "list_resume_cache_files", lambda: [], raising=False)
    source_path = tmp_path / "gradio" / "source.png"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"source-image")
    source_face_set = FaceSet()
    source_face_set.faces.append(make_face())
    roop.globals.INPUT_FACESETS.append(source_face_set)
    ui.globals.ui_input_face_refs.append({"type": "image_face", "path": str(source_path), "face_index": 0})
    faceswap_tab.list_files_process[:] = [ProcessEntry("C:/target.mp4", 0, 100, 24.0)]

    now_values = iter([1111111111, 2222222222])
    stamp_values = iter(["20260101_010101", "20270101_020202"])
    monkeypatch.setattr(faceswap_tab.time, "time", lambda: next(now_values))
    monkeypatch.setattr(faceswap_tab.time, "strftime", lambda fmt: next(stamp_values))

    first_resume_path = faceswap_tab.write_resume_payload(faceswap_tab.build_resume_payload({"output_method": "File"}))
    second_resume_path = faceswap_tab.write_resume_payload(faceswap_tab.build_resume_payload({"output_method": "File"}))

    assert first_resume_path == second_resume_path
    assert len(list(tmp_path.glob("*.json"))) == 1
    assert len(list(tmp_path.glob("*_assets"))) == 1
