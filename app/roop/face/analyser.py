import threading
from typing import Any

import insightface

import roop.config.globals
from roop.utils import conditional_download, resolve_relative_path


FACE_ANALYSER = None
THREAD_LOCK_ANALYSER = threading.Lock()


def release_face_analyser():
    global FACE_ANALYSER
    with THREAD_LOCK_ANALYSER:
        if FACE_ANALYSER is not None:
            del FACE_ANALYSER
            FACE_ANALYSER = None


def get_face_analyser() -> Any:
    global FACE_ANALYSER

    with THREAD_LOCK_ANALYSER:
        if FACE_ANALYSER is None or roop.config.globals.g_current_face_analysis != roop.config.globals.g_desired_face_analysis:
            conditional_download(
                resolve_relative_path("../models"),
                [],
            )
            model_path = resolve_relative_path("..")
            allowed_modules = roop.config.globals.g_desired_face_analysis
            roop.config.globals.g_current_face_analysis = roop.config.globals.g_desired_face_analysis
            if roop.config.globals.CFG.force_cpu:
                print("Forcing CPU for Face Analysis")
                FACE_ANALYSER = insightface.app.FaceAnalysis(
                    name="buffalo_l",
                    root=model_path,
                    providers=["CPUExecutionProvider"],
                    allowed_modules=allowed_modules,
                )
            else:
                FACE_ANALYSER = insightface.app.FaceAnalysis(
                    name="buffalo_l",
                    root=model_path,
                    providers=roop.config.globals.execution_providers,
                    allowed_modules=allowed_modules,
                )
            FACE_ANALYSER.prepare(
                ctx_id=0,
                det_size=(640, 640) if roop.config.globals.default_det_size else (320, 320),
            )
    return FACE_ANALYSER


__all__ = ["get_face_analyser", "release_face_analyser"]
