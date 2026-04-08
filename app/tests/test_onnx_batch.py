from pathlib import Path

import onnx
from onnx import TensorProto, helper


def _make_identity_model(path: Path, input_shape, output_shape):
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        "identity",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)],
    )
    model = helper.make_model(graph, producer_name="pytest")
    onnx.save(model, path)


def test_ensure_native_batch_model_rewrites_static_batch_to_symbolic(tmp_path, monkeypatch):
    try:
        import roop.onnx.batch as onnx_batch
    except ImportError as exc:
        raise AssertionError("roop.onnx.batch helper should exist") from exc

    model_path = tmp_path / "static-batch.onnx"
    _make_identity_model(model_path, [1, 3], [1, 3])

    cache_root = tmp_path / "processing_cache"
    monkeypatch.setattr(onnx_batch, "get_processing_cache_root", lambda: cache_root)

    patched_path = Path(onnx_batch.ensure_native_batch_model(str(model_path)))

    assert patched_path != model_path
    assert patched_path.exists()

    patched_model = onnx.load(str(patched_path))
    input_dim = patched_model.graph.input[0].type.tensor_type.shape.dim[0]
    output_dim = patched_model.graph.output[0].type.tensor_type.shape.dim[0]

    assert input_dim.dim_param == "batch"
    assert output_dim.dim_param == "batch"
    assert not input_dim.HasField("dim_value")
    assert not output_dim.HasField("dim_value")


def test_ensure_native_batch_model_keeps_dynamic_model_path(tmp_path):
    try:
        import roop.onnx.batch as onnx_batch
    except ImportError as exc:
        raise AssertionError("roop.onnx.batch helper should exist") from exc

    model_path = tmp_path / "dynamic-batch.onnx"
    _make_identity_model(model_path, ["batch", 3], ["batch", 3])

    resolved_path = onnx_batch.ensure_native_batch_model(str(model_path))

    assert resolved_path == str(model_path)


def test_ensure_native_batch_model_rejects_patch_when_output_stays_static(tmp_path, monkeypatch):
    try:
        import roop.onnx.batch as onnx_batch
    except ImportError as exc:
        raise AssertionError("roop.onnx.batch helper should exist") from exc

    model_path = tmp_path / "static-output.onnx"
    _make_identity_model(model_path, [1, 3], [1, 3])

    cache_root = tmp_path / "processing_cache"
    monkeypatch.setattr(onnx_batch, "get_processing_cache_root", lambda: cache_root)

    def fake_infer_shapes(model):
        inferred = onnx.ModelProto()
        inferred.CopyFrom(model)
        output_dim = inferred.graph.output[0].type.tensor_type.shape.dim[0]
        output_dim.ClearField("dim_param")
        output_dim.dim_value = 1
        return inferred

    monkeypatch.setattr(onnx_batch.onnx.shape_inference, "infer_shapes", fake_infer_shapes)

    resolved_path = onnx_batch.ensure_native_batch_model(str(model_path))

    assert resolved_path == str(model_path)
    assert not (cache_root / "onnx_batch").exists()


def test_ensure_native_batch_model_ignores_existing_unsafe_cached_patch(tmp_path, monkeypatch):
    try:
        import roop.onnx.batch as onnx_batch
    except ImportError as exc:
        raise AssertionError("roop.onnx.batch helper should exist") from exc

    model_path = tmp_path / "cached-static-output.onnx"
    _make_identity_model(model_path, [1, 3], [1, 3])

    cache_root = tmp_path / "processing_cache"
    monkeypatch.setattr(onnx_batch, "get_processing_cache_root", lambda: cache_root)

    cache_key = onnx_batch._get_source_cache_key(model_path)
    patched_path = onnx_batch._build_patched_model_path(model_path, cache_key)
    patched_model = onnx.load(str(model_path))
    patched_model.graph.input[0].type.tensor_type.shape.dim[0].ClearField("dim_value")
    patched_model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "batch"
    patched_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(patched_model, patched_path)

    resolved_path = onnx_batch.ensure_native_batch_model(str(model_path))

    resolved_model = onnx.load(resolved_path)
    input_dim = resolved_model.graph.input[0].type.tensor_type.shape.dim[0]
    output_dim = resolved_model.graph.output[0].type.tensor_type.shape.dim[0]
    assert not input_dim.HasField("dim_value")
    assert not output_dim.HasField("dim_value")

