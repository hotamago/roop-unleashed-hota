from roop.media.ffmpeg_writer import FFMPEG_VideoWriter


class _DummyPipe:
    def close(self):
        return None


class _DummyProcess:
    def __init__(self, cmd):
        self.cmd = cmd
        self.stdin = _DummyPipe()
        self.stderr = _DummyPipe()

    def wait(self):
        return 0


def test_ffmpeg_writer_uses_custom_quality_args(monkeypatch):
    captured = {}

    def fake_popen(cmd, **_kwargs):
        captured["cmd"] = cmd
        return _DummyProcess(cmd)

    monkeypatch.setattr("roop.media.ffmpeg_writer.sp.Popen", fake_popen)

    writer = FFMPEG_VideoWriter(
        "out.mp4",
        (16, 16),
        24,
        codec="h264_nvenc",
        quality_args=["-cq", "18"],
        ffmpeg_params=["-preset", "p1"],
    )
    writer.close()

    assert "-cq" in captured["cmd"]
    assert "18" in captured["cmd"]
    assert "-crf" not in captured["cmd"]

