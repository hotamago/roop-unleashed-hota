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
