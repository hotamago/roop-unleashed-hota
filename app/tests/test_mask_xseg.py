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
