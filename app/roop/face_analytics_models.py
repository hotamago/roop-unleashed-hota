DEFAULT_FACE_DETECTOR_MODEL = "insightface"
DEFAULT_FACE_LANDMARKER_MODEL = "insightface_2d106"
DEFAULT_FACE_MASKER_MODEL = "legacy_xseg"

FACE_DETECTOR_MODEL_SET = {
    "insightface": {
        "label": "InsightFace buffalo_l",
        "family": "insightface",
        "filenames": [],
        "urls": [],
        "size_choices": ["320x320", "640x640"],
    },
    "retinaface": {
        "label": "RetinaFace 10G",
        "family": "retinaface",
        "filenames": ["retinaface_10g.onnx"],
        "urls": ["https://huggingface.co/facefusion/models-3.0.0/resolve/main/retinaface_10g.onnx"],
        "size_choices": ["160x160", "320x320", "480x480", "512x512", "640x640"],
    },
    "scrfd": {
        "label": "SCRFD 2.5G",
        "family": "scrfd",
        "filenames": ["scrfd_2.5g.onnx"],
        "urls": ["https://huggingface.co/facefusion/models-3.0.0/resolve/main/scrfd_2.5g.onnx"],
        "size_choices": ["160x160", "320x320", "480x480", "512x512", "640x640"],
    },
    "yolo_face": {
        "label": "YOLO Face 8N",
        "family": "yolo_face",
        "filenames": ["yoloface_8n.onnx"],
        "urls": ["https://huggingface.co/facefusion/models-3.0.0/resolve/main/yoloface_8n.onnx"],
        "size_choices": ["640x640"],
    },
    "yunet": {
        "label": "YuNet 2023 Mar",
        "family": "yunet",
        "filenames": ["yunet_2023_mar.onnx"],
        "urls": ["https://huggingface.co/facefusion/models-3.4.0/resolve/main/yunet_2023_mar.onnx"],
        "size_choices": ["640x640"],
    },
}

FACE_LANDMARKER_MODEL_SET = {
    "insightface_2d106": {
        "label": "InsightFace 2D106",
        "family": "insightface_2d106",
        "filenames": [],
        "urls": [],
    },
    "2dfan4": {
        "label": "2DFAN4 68pt",
        "family": "2dfan4",
        "filenames": ["2dfan4.onnx"],
        "urls": ["https://huggingface.co/facefusion/models-3.0.0/resolve/main/2dfan4.onnx"],
        "size": 256,
    },
    "peppa_wutz": {
        "label": "Peppa Wutz 68pt",
        "family": "peppa_wutz",
        "filenames": ["peppa_wutz.onnx"],
        "urls": ["https://huggingface.co/facefusion/models-3.0.0/resolve/main/peppa_wutz.onnx"],
        "size": 256,
    },
    "fan_68_5": {
        "label": "FAN 68 from 5pt",
        "family": "fan_68_5",
        "filenames": ["fan_68_5.onnx"],
        "urls": ["https://huggingface.co/facefusion/models-3.0.0/resolve/main/fan_68_5.onnx"],
    },
}

FACE_MASKER_MODEL_SET = {
    "legacy_xseg": {
        "label": "Legacy XSeg",
        "family": "legacy_xseg",
        "filenames": ["xseg.onnx"],
        "urls": ["https://huggingface.co/countfloyd/deepfake/resolve/main/xseg.onnx"],
        "size": 256,
    },
    "xseg_1": {
        "label": "XSeg 1",
        "family": "xseg",
        "filenames": ["xseg_1.onnx"],
        "urls": ["https://huggingface.co/facefusion/models-3.1.0/resolve/main/xseg_1.onnx"],
        "size": 256,
    },
    "xseg_2": {
        "label": "XSeg 2",
        "family": "xseg",
        "filenames": ["xseg_2.onnx"],
        "urls": ["https://huggingface.co/facefusion/models-3.1.0/resolve/main/xseg_2.onnx"],
        "size": 256,
    },
    "xseg_3": {
        "label": "XSeg 3",
        "family": "xseg",
        "filenames": ["xseg_3.onnx"],
        "urls": ["https://huggingface.co/facefusion/models-3.2.0/resolve/main/xseg_3.onnx"],
        "size": 256,
    },
    "bisenet_resnet_18": {
        "label": "BiSeNet ResNet-18",
        "family": "parser",
        "filenames": ["bisenet_resnet_18.onnx"],
        "urls": ["https://huggingface.co/facefusion/models-3.1.0/resolve/main/bisenet_resnet_18.onnx"],
        "size": 512,
    },
    "bisenet_resnet_34": {
        "label": "BiSeNet ResNet-34",
        "family": "parser",
        "filenames": ["bisenet_resnet_34.onnx"],
        "urls": ["https://huggingface.co/facefusion/models-3.0.0/resolve/main/bisenet_resnet_34.onnx"],
        "size": 512,
    },
}

FACE_MASK_REGION_SET = {
    "skin": 1,
    "left-eyebrow": 2,
    "right-eyebrow": 3,
    "left-eye": 4,
    "right-eye": 5,
    "glasses": 6,
    "nose": 10,
    "mouth": 11,
    "upper-lip": 12,
    "lower-lip": 13,
}
DEFAULT_FACE_MASK_REGIONS = list(FACE_MASK_REGION_SET.keys())


def _get_model_key(model_name, model_set, default_key):
    candidate = str(model_name or "").strip()
    if candidate in model_set:
        return candidate
    return default_key


def _get_model_config(model_name, model_set, default_key):
    return model_set[_get_model_key(model_name, model_set, default_key)]


def _get_model_choices(model_set):
    return list(model_set.keys())


def get_face_detector_model_key(model_name=None) -> str:
    return _get_model_key(model_name, FACE_DETECTOR_MODEL_SET, DEFAULT_FACE_DETECTOR_MODEL)


def get_face_detector_model_config(model_name=None) -> dict:
    return _get_model_config(model_name, FACE_DETECTOR_MODEL_SET, DEFAULT_FACE_DETECTOR_MODEL)


def get_face_detector_model_choices() -> list[str]:
    return _get_model_choices(FACE_DETECTOR_MODEL_SET)


def get_face_detector_size_choices(model_name=None) -> list[str]:
    return list(get_face_detector_model_config(model_name).get("size_choices") or ["640x640"])


def get_face_detector_model_hint(model_name=None) -> str:
    model_key = get_face_detector_model_key(model_name)
    config = get_face_detector_model_config(model_key)
    size_choices = ", ".join(get_face_detector_size_choices(model_key))
    if model_key == "insightface":
        return (
            f"Selected `{model_key}`. Uses the bundled buffalo_l detector and keeps the existing "
            f"106-point InsightFace analytics path. Compatible detector sizes: {size_choices}."
        )
    return (
        f"Selected `{model_key}`. Downloads the FaceFusion detector on first use, then feeds its "
        f"5-point detections back into the existing InsightFace analytics stack. Compatible sizes: {size_choices}."
    )


def get_face_landmarker_model_key(model_name=None) -> str:
    return _get_model_key(model_name, FACE_LANDMARKER_MODEL_SET, DEFAULT_FACE_LANDMARKER_MODEL)


def get_face_landmarker_model_config(model_name=None) -> dict:
    return _get_model_config(model_name, FACE_LANDMARKER_MODEL_SET, DEFAULT_FACE_LANDMARKER_MODEL)


def get_face_landmarker_model_choices() -> list[str]:
    return _get_model_choices(FACE_LANDMARKER_MODEL_SET)


def get_face_landmarker_model_hint(model_name=None) -> str:
    model_key = get_face_landmarker_model_key(model_name)
    if model_key == "insightface_2d106":
        return "Selected `insightface_2d106`. Keeps the current 106-point landmark path from buffalo_l."
    return (
        f"Selected `{model_key}`. Adds a FaceFusion-style 68-point landmarker while keeping the "
        "existing 106-point compatibility landmarks for swap, recognition, and mouth restore."
    )


def get_face_masker_model_key(model_name=None) -> str:
    return _get_model_key(model_name, FACE_MASKER_MODEL_SET, DEFAULT_FACE_MASKER_MODEL)


def get_face_masker_model_config(model_name=None) -> dict:
    return _get_model_config(model_name, FACE_MASKER_MODEL_SET, DEFAULT_FACE_MASKER_MODEL)


def get_face_masker_model_choices() -> list[str]:
    return _get_model_choices(FACE_MASKER_MODEL_SET)


def get_face_masker_model_family(model_name=None) -> str:
    return str(get_face_masker_model_config(model_name).get("family") or "legacy_xseg")


def get_face_masker_model_size(model_name=None) -> int:
    return int(get_face_masker_model_config(model_name).get("size") or 256)


def get_face_masker_model_hint(model_name=None) -> str:
    model_key = get_face_masker_model_key(model_name)
    family = get_face_masker_model_family(model_key)
    if family == "parser":
        regions = ", ".join(DEFAULT_FACE_MASK_REGIONS)
        return (
            f"Selected `{model_key}`. Uses a FaceFusion face parser and converts selected regions "
            f"into the existing inverse mask workflow. Default regions: {regions}."
        )
    if family == "legacy_xseg":
        return "Selected `legacy_xseg`. Keeps the current DFL-style XSeg mask behavior."
    return f"Selected `{model_key}`. Uses a FaceFusion XSeg occluder model with the existing mask workflow."


def _resolve_model_paths(model_config: dict) -> list[str]:
    from roop.utils import resolve_relative_path

    return [
        resolve_relative_path(f"../models/{filename}")
        for filename in model_config.get("filenames") or []
    ]


def _ensure_model_downloaded(model_config: dict) -> list[str]:
    from roop.utils import conditional_download, resolve_relative_path

    urls = list(model_config.get("urls") or [])
    if urls:
        conditional_download(resolve_relative_path("../models"), urls)
    return _resolve_model_paths(model_config)


def get_face_detector_model_paths(model_name=None) -> list[str]:
    return _resolve_model_paths(get_face_detector_model_config(model_name))


def ensure_face_detector_model_downloaded(model_name=None) -> list[str]:
    return _ensure_model_downloaded(get_face_detector_model_config(model_name))


def get_face_landmarker_model_paths(model_name=None) -> list[str]:
    return _resolve_model_paths(get_face_landmarker_model_config(model_name))


def ensure_face_landmarker_model_downloaded(model_name=None) -> list[str]:
    return _ensure_model_downloaded(get_face_landmarker_model_config(model_name))


def get_face_masker_model_paths(model_name=None) -> list[str]:
    return _resolve_model_paths(get_face_masker_model_config(model_name))


def ensure_face_masker_model_downloaded(model_name=None) -> list[str]:
    return _ensure_model_downloaded(get_face_masker_model_config(model_name))


__all__ = [
    "DEFAULT_FACE_DETECTOR_MODEL",
    "DEFAULT_FACE_LANDMARKER_MODEL",
    "DEFAULT_FACE_MASKER_MODEL",
    "DEFAULT_FACE_MASK_REGIONS",
    "FACE_MASK_REGION_SET",
    "ensure_face_detector_model_downloaded",
    "ensure_face_landmarker_model_downloaded",
    "ensure_face_masker_model_downloaded",
    "get_face_detector_model_choices",
    "get_face_detector_model_config",
    "get_face_detector_model_hint",
    "get_face_detector_model_key",
    "get_face_detector_model_paths",
    "get_face_detector_size_choices",
    "get_face_landmarker_model_choices",
    "get_face_landmarker_model_config",
    "get_face_landmarker_model_hint",
    "get_face_landmarker_model_key",
    "get_face_landmarker_model_paths",
    "get_face_masker_model_choices",
    "get_face_masker_model_config",
    "get_face_masker_model_family",
    "get_face_masker_model_hint",
    "get_face_masker_model_key",
    "get_face_masker_model_paths",
    "get_face_masker_model_size",
]
