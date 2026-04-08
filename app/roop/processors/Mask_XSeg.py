import cv2
import numpy as np
import roop.config.globals

from roop.config.types import Frame
from roop.face_analytics_models import (
    DEFAULT_FACE_MASK_REGIONS,
    FACE_MASK_REGION_SET,
    ensure_face_masker_model_downloaded,
    get_face_masker_model_family,
    get_face_masker_model_key,
    get_face_masker_model_size,
)
from roop.onnx.runtime import resolve_model_path_for_processor
from roop.onnx.session import create_inference_session
from roop.processors.base import BaseProcessor


class Mask_XSeg(BaseProcessor):
    plugin_options: dict = None
    model_xseg = None
    processorname = "mask_xseg"
    type = "mask"
    supports_batch = True

    def _current_model_key(self):
        return get_face_masker_model_key(
            getattr(self, "active_model_key", None)
            or getattr(getattr(roop.config.globals, "CFG", None), "face_masker_model", None)
        )

    def _current_model_family(self):
        return get_face_masker_model_family(self._current_model_key())

    def _current_model_size(self):
        return get_face_masker_model_size(self._current_model_key())

    def Initialize(self, plugin_options: dict):
        model_key = get_face_masker_model_key(getattr(roop.config.globals.CFG, "face_masker_model", None))
        current_device = plugin_options.get("devicename")
        if self.plugin_options is not None:
            if (
                self.plugin_options.get("devicename") != current_device
                or getattr(self, "active_model_key", None) != model_key
            ):
                self.Release()

        self.plugin_options = plugin_options
        self.active_model_key = model_key
        self.active_model_family = get_face_masker_model_family(model_key)
        self.model_size = get_face_masker_model_size(model_key)
        self.parser_region_ids = np.array(
            [FACE_MASK_REGION_SET[name] for name in DEFAULT_FACE_MASK_REGIONS],
            dtype=np.int32,
        )

        if self.model_xseg is None:
            model_path = ensure_face_masker_model_downloaded(model_key)[0]
            model_path = resolve_model_path_for_processor(model_path, self.processorname)
            self.model_xseg = create_inference_session(model_path, self.processorname)
            self.model_inputs = self.model_xseg.get_inputs()
            self.model_outputs = self.model_xseg.get_outputs()
            self.input_name = self.model_inputs[0].name
            self.output_name = self.model_outputs[0].name
            self.batch_size_limit = self._resolve_batch_size_limit()
            self.devicename = current_device.replace("mps", "cpu")

    def Run(self, img1, keywords: str) -> Frame:
        batch_outputs = self.RunBatch([img1], keywords, batch_size=1)
        if batch_outputs:
            return batch_outputs[0]
        return np.zeros((self._current_model_size(), self._current_model_size(), 1), dtype=np.float32)

    def _preprocess_frame(self, image):
        model_family = self._current_model_family()
        model_size = self._current_model_size()
        temp_frame = cv2.resize(image, (model_size, model_size), cv2.INTER_CUBIC)
        if model_family == "parser":
            temp_frame = temp_frame[:, :, ::-1].astype(np.float32) / 255.0
            temp_frame = np.subtract(temp_frame, np.array([0.485, 0.456, 0.406], dtype=np.float32))
            temp_frame = np.divide(temp_frame, np.array([0.229, 0.224, 0.225], dtype=np.float32))
            temp_frame = temp_frame.transpose(2, 0, 1)
            return np.ascontiguousarray(temp_frame)
        return np.ascontiguousarray(temp_frame.astype(np.float32) / 255.0)

    def _normalize_mask_output(self, result):
        model_family = self._current_model_family()
        if model_family == "parser":
            if result.ndim == 4 and result.shape[0] == 1:
                result = result[0]
            if result.ndim == 3 and result.shape[0] > 1:
                region_mask = np.isin(result.argmax(0), self.parser_region_ids).astype(np.float32)
            elif result.ndim == 3 and result.shape[-1] > 1:
                region_mask = np.isin(result.argmax(-1), self.parser_region_ids).astype(np.float32)
            else:
                region_mask = np.asarray(result, dtype=np.float32)
                if region_mask.ndim == 3:
                    region_mask = region_mask[..., 0]
            region_mask = (cv2.GaussianBlur(region_mask.clip(0, 1), (0, 0), 5).clip(0.5, 1) - 0.5) * 2
            return (1.0 - region_mask[..., None]).astype(np.float32)

        if result.ndim == 4 and result.shape[0] == 1:
            result = result[0]
        result = np.clip(result, 0, 1.0)
        result[result < 0.1] = 0
        if result.ndim == 2:
            result = result[..., None]
        return 1.0 - result

    def RunBatch(self, images, keywords: str, batch_size=1):
        del keywords
        outputs = []
        if not images:
            return outputs

        model_family = self._current_model_family()
        model_size = self._current_model_size()
        effective_batch_size = self._effective_batch_size(batch_size)
        batch_shape = (
            (effective_batch_size, 3, model_size, model_size)
            if model_family == "parser"
            else (effective_batch_size, model_size, model_size, 3)
        )
        batch_input = self._get_batch_buffer("mask_input", batch_shape, np.float32)

        for batch_start in range(0, len(images), effective_batch_size):
            current_images = images[batch_start:batch_start + effective_batch_size]
            if not current_images:
                continue
            for index, image in enumerate(current_images):
                batch_input[index] = self._preprocess_frame(image)
            batch_outputs = self.model_xseg.run(
                None,
                {getattr(self, "input_name", self.model_inputs[0].name): np.ascontiguousarray(batch_input[: len(current_images)])},
            )[0]
            for result in batch_outputs:
                outputs.append(self._normalize_mask_output(result))
        return outputs

    def Release(self):
        if self.model_xseg is not None:
            del self.model_xseg
        self.model_xseg = None
        self.active_model_key = None
        self.active_model_family = None
        self.batch_size_limit = None
        self._clear_batch_buffers()
