from types import SimpleNamespace

import numpy as np

from roop.processors.Mask_XSeg import Mask_XSeg


class FakeMaskSession:
    def run(self, _, inputs):
        batch = inputs["input"]
        outputs = np.stack(
            [
                np.full((256, 256, 1), 0.25, dtype=np.float32),
                np.full((256, 256, 1), 0.75, dtype=np.float32),
            ][: batch.shape[0]],
            axis=0,
        )
        return [outputs]


def test_mask_xseg_run_batch_preserves_full_mask_shape():
    processor = Mask_XSeg()
    processor.model_xseg = FakeMaskSession()
    processor.model_inputs = [SimpleNamespace(name="input")]

    images = [
        np.zeros((64, 64, 3), dtype=np.uint8),
        np.ones((64, 64, 3), dtype=np.uint8),
    ]

    outputs = processor.RunBatch(images, "", batch_size=8)

    assert len(outputs) == 2
    assert outputs[0].shape == (256, 256, 1)
    assert outputs[1].shape == (256, 256, 1)
    assert np.allclose(outputs[0], 0.75)
    assert np.allclose(outputs[1], 0.25)


def test_mask_xseg_parser_model_outputs_inverse_region_mask():
    processor = Mask_XSeg()
    processor.active_model_key = "bisenet_resnet_18"
    processor.parser_region_ids = np.array([1], dtype=np.int32)
    processor.model_xseg = SimpleNamespace(
        run=lambda _, inputs: [np.stack([np.stack([np.zeros((512, 512), dtype=np.float32), np.ones((512, 512), dtype=np.float32)], axis=0)], axis=0)]
    )
    processor.model_inputs = [SimpleNamespace(name="input")]

    outputs = processor.RunBatch([np.zeros((64, 64, 3), dtype=np.uint8)], "", batch_size=4)

    assert len(outputs) == 1
    assert outputs[0].shape == (512, 512, 1)
    assert np.allclose(outputs[0], 0.0)
