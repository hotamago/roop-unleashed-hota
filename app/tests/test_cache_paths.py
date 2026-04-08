import os
from types import SimpleNamespace

import roop.config.globals
from roop.utils.cache_paths import get_jobs_root
from ui.main import prepare_environment


def test_jobs_root_uses_persistent_processing_cache(tmp_path, monkeypatch):
    def fake_resolve(path):
        if path == '../processing_cache':
            return str(tmp_path / "processing_cache")
        if path == '../temp':
            return str(tmp_path / "temp")
        raise AssertionError(path)

    monkeypatch.setattr("roop.utils.cache_paths.resolve_relative_path", fake_resolve)
    monkeypatch.setenv("TEMP", str(tmp_path / "runtime_temp"))

    jobs_root = get_jobs_root()

    assert jobs_root == tmp_path / "processing_cache" / "jobs"
    assert str(jobs_root).startswith(str(tmp_path / "processing_cache"))


def test_prepare_environment_separates_gradio_temp_from_runtime_temp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(roop.config.globals, "CFG", SimpleNamespace(use_os_temp_folder=False), raising=False)

    prepare_environment()

    assert os.environ["GRADIO_TEMP_DIR"] != os.environ["TEMP"]
    assert os.environ["GRADIO_TEMP_DIR"].startswith(os.environ["TEMP"])
    assert os.path.isdir(os.environ["GRADIO_TEMP_DIR"])

