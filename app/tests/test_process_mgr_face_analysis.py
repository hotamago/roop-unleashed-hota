from types import SimpleNamespace

import numpy as np

from roop.face.analyser import HybridFaceAnalyser
from roop.pipeline.batch_executor import ProcessMgr


def test_process_mgr_limits_face_analysis_modules_for_non_matching_modes():
    mgr = ProcessMgr(None)
    mgr.options = SimpleNamespace(swap_mode="all")

    modules = mgr.resolve_face_analysis_modules()

    assert modules == ["landmark_3d_68", "landmark_2d_106", "detection"]


def test_process_mgr_adds_recognition_only_for_selected_matching():
    mgr = ProcessMgr(None)
    mgr.options = SimpleNamespace(swap_mode="selected")

    modules = mgr.resolve_face_analysis_modules()

    assert modules == ["landmark_3d_68", "landmark_2d_106", "detection", "recognition"]


def test_process_mgr_adds_genderage_for_gender_filtering():
    mgr = ProcessMgr(None)
    mgr.options = SimpleNamespace(swap_mode="all_female")

    modules = mgr.resolve_face_analysis_modules()

    assert modules == ["landmark_3d_68", "landmark_2d_106", "detection", "genderage"]


def test_hybrid_face_analyser_custom_detector_preserves_compatibility_landmarks(monkeypatch):
    class FakeCompatModel:
        def get(self, _frame, face):
            face["landmark_2d_106"] = np.zeros((106, 2), dtype=np.float32)

    fake_detector = SimpleNamespace(
        detect=lambda _frame, max_num=0: (
            np.array([[1.0, 2.0, 11.0, 12.0, 0.9]], dtype=np.float32),
            np.array([[[2.0, 3.0], [5.0, 3.0], [3.5, 5.0], [2.5, 8.0], [5.5, 8.0]]], dtype=np.float32),
        )
    )
    fake_landmarker = SimpleNamespace(
        detect=lambda _frame, _bbox, _kps: (np.zeros((68, 2), dtype=np.float32), 0.88)
    )
    compat_analyser = SimpleNamespace(models={"detection": object(), "landmark_2d_106": FakeCompatModel()})

    monkeypatch.setattr("roop.face.analyser.create_face_detector", lambda *args, **kwargs: fake_detector)
    monkeypatch.setattr("roop.face.analyser.create_face_landmarker", lambda *args, **kwargs: fake_landmarker)

    analyser = HybridFaceAnalyser(
        compat_analyser,
        "scrfd",
        "fan_68_5",
        ["CPUExecutionProvider"],
        (640, 640),
    )
    faces = analyser.get(np.zeros((16, 16, 3), dtype=np.uint8))

    assert len(faces) == 1
    assert faces[0].landmark_2d_106.shape == (106, 2)
    assert faces[0].landmark_2d_68.shape == (68, 2)
    assert faces[0].kps.shape == (5, 2)
