import roop.config.globals
from roop.core.app import batch_process_regular
from roop.pipeline.entry import ProcessEntry


def test_batch_process_regular_sets_staged_video_finalname_before_executor(tmp_path, monkeypatch):
    source_path = tmp_path / "clip.mp4"
    source_path.write_bytes(b"video")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    entry = ProcessEntry(str(source_path), 0, 10, 30.0)

    class FakeExecutor:
        def __init__(self, _output_method, _progress, _options):
            return None

        def run(self, files):
            assert files[0].finalname is not None
            assert files[0].finalname.endswith("__temp.mp4")

    monkeypatch.setattr(roop.config.globals, "output_path", str(output_dir), raising=False)
    monkeypatch.setattr(roop.config.globals.CFG, "output_video_format", "mp4", raising=False)
    monkeypatch.setattr(roop.config.globals.CFG, "output_image_format", "png", raising=False)
    monkeypatch.setattr("roop.core.app.release_resources", lambda: None)
    monkeypatch.setattr("roop.core.app.limit_resources", lambda: None)
    monkeypatch.setattr("roop.core.app.set_processing_message", lambda *args, **kwargs: None)
    monkeypatch.setattr("roop.core.app.end_processing", lambda _message: None)
    monkeypatch.setattr("roop.core.app.StagedBatchExecutor", FakeExecutor)

    batch_process_regular("File", [entry], None, "", "Smart staged processing", None, False, 1, None, 0)


def test_batch_process_regular_sets_staged_image_finalname_before_executor(tmp_path, monkeypatch):
    source_path = tmp_path / "frame.png"
    source_path.write_bytes(b"image")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    entry = ProcessEntry(str(source_path), 0, 0, 0.0)

    class FakeExecutor:
        def __init__(self, _output_method, _progress, _options):
            return None

        def run(self, files):
            assert files[0].finalname is not None
            assert files[0].finalname.endswith(".png")
            assert "__temp" not in files[0].finalname

    monkeypatch.setattr(roop.config.globals, "output_path", str(output_dir), raising=False)
    monkeypatch.setattr(roop.config.globals.CFG, "output_video_format", "mp4", raising=False)
    monkeypatch.setattr(roop.config.globals.CFG, "output_image_format", "png", raising=False)
    monkeypatch.setattr(roop.config.globals.CFG, "output_template", "{file}_{i}", raising=False)
    monkeypatch.setattr("roop.core.app.release_resources", lambda: None)
    monkeypatch.setattr("roop.core.app.limit_resources", lambda: None)
    monkeypatch.setattr("roop.core.app.set_processing_message", lambda *args, **kwargs: None)
    monkeypatch.setattr("roop.core.app.end_processing", lambda _message: None)
    monkeypatch.setattr("roop.core.app.StagedBatchExecutor", FakeExecutor)

    batch_process_regular("File", [entry], None, "", "Smart staged processing", None, False, 1, None, 0)
