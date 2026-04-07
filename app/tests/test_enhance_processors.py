import numpy as np

from roop.processors.Enhance_GFPGAN import Enhance_GFPGAN


class FakeInputMeta:
    def __init__(self, shape):
        self.shape = shape


class FakeEnhanceSession:
    def __init__(self, fail_on_batch_over_one=False):
        self.fail_on_batch_over_one = fail_on_batch_over_one
        self.calls = []

    def get_inputs(self):
        return [FakeInputMeta([1, 3, 512, 512])]

    def run(self, _, inputs):
        batch = inputs["input"]
        self.calls.append(batch.shape[0])
        if self.fail_on_batch_over_one and batch.shape[0] > 1:
            raise RuntimeError(
                "[ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Got invalid dimensions for input: input "
                "for the following indices index: 0 Got: 25 Expected: 1"
            )
        return [batch.copy()]


def test_gfpgan_detects_single_batch_model():
    processor = Enhance_GFPGAN()
    processor.model_gfpgan = FakeEnhanceSession()
    processor.batch_size_limit = processor._resolve_batch_size_limit()

    assert processor._resolve_batch_size_limit() == 1
    assert processor._effective_batch_size(32) == 1


def test_gfpgan_falls_back_to_single_batch_when_ort_rejects_batch():
    processor = Enhance_GFPGAN()
    processor.model_gfpgan = FakeEnhanceSession(fail_on_batch_over_one=True)
    processor.batch_size_limit = None
    processor.supports_batch = True

    temp_frames = [
        np.full((64, 64, 3), 32, dtype=np.uint8),
        np.full((64, 64, 3), 64, dtype=np.uint8),
    ]

    outputs = processor.RunBatch([None, None], [None, None], temp_frames, batch_size=8)

    assert len(outputs) == 2
    assert processor.batch_size_limit == 1
    assert processor.supports_batch is False
    assert processor.model_gfpgan.calls == [2, 1, 1]
