import roop.config.globals


def test_resolve_model_path_for_faceswap_keeps_original_model(monkeypatch):
    try:
        import roop.onnx.runtime as onnx_runtime
    except ImportError as exc:
        raise AssertionError("roop.onnx.runtime helper should exist") from exc

    ensure_calls = {"count": 0}

    monkeypatch.setattr(
        onnx_runtime,
        "ensure_native_batch_model",
        lambda model_path: ensure_calls.__setitem__("count", ensure_calls["count"] + 1) or f"{model_path}.patched",
    )

    resolved_path = onnx_runtime.resolve_model_path_for_processor("model.onnx", "faceswap")

    assert resolved_path == "model.onnx"
    assert ensure_calls["count"] == 0


def test_resolve_model_path_for_gfpgan_keeps_original_model(monkeypatch):
    try:
        import roop.onnx.runtime as onnx_runtime
    except ImportError as exc:
        raise AssertionError("roop.onnx.runtime helper should exist") from exc

    ensure_calls = {"count": 0}

    def fake_ensure_native_batch_model(model_path):
        ensure_calls["count"] += 1
        return f"{model_path}.patched"

    monkeypatch.setattr(onnx_runtime, "ensure_native_batch_model", fake_ensure_native_batch_model)

    resolved_path = onnx_runtime.resolve_model_path_for_processor("model.onnx", "gfpgan")

    assert resolved_path == "model.onnx"
    assert ensure_calls["count"] == 0


def test_get_execution_providers_for_faceswap_keeps_tensorrt():
    try:
        import roop.onnx.runtime as onnx_runtime
    except ImportError as exc:
        raise AssertionError("roop.onnx.runtime helper should exist") from exc

    roop.config.globals.execution_providers = [
        ("TensorrtExecutionProvider", {"device_id": 0}),
        ("CUDAExecutionProvider", {"device_id": 0}),
        "CPUExecutionProvider",
    ]

    providers = onnx_runtime.get_execution_providers_for_processor("faceswap")

    assert providers[0][0] == "TensorrtExecutionProvider"
    assert providers[1][0] == "CUDAExecutionProvider"


def test_get_execution_providers_for_gfpgan_drops_tensorrt():
    try:
        import roop.onnx.runtime as onnx_runtime
    except ImportError as exc:
        raise AssertionError("roop.onnx.runtime helper should exist") from exc

    roop.config.globals.execution_providers = [
        ("TensorrtExecutionProvider", {"device_id": 0}),
        ("CUDAExecutionProvider", {"device_id": 0}),
        "CPUExecutionProvider",
    ]

    providers = onnx_runtime.get_execution_providers_for_processor("gfpgan")

    assert providers[0][0] == "CUDAExecutionProvider"
    assert all(
        provider != "TensorrtExecutionProvider"
        and not (isinstance(provider, tuple) and provider[0] == "TensorrtExecutionProvider")
        for provider in providers
    )

