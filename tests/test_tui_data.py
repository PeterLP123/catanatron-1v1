import io
import json

import pytest

from catanatron.gym.tui_data import _tail_lines, read_json_safe, read_jsonl_safe


@pytest.mark.parametrize("ending", [b"", b"\n", b"\r\n"])
def test_jsonl_tail_preserves_unicode_and_limits_physical_lines(tmp_path, ending):
    path = tmp_path / "events.jsonl"
    rows = [{"id": i, "message": "zażółć"} for i in range(1000)]
    path.write_bytes(
        b"\n".join(json.dumps(row, ensure_ascii=False).encode() for row in rows)
        + ending
    )
    assert read_jsonl_safe(path, limit=3) == rows[-3:]
    assert read_jsonl_safe(path, limit=2000) == rows
    assert read_jsonl_safe(path, limit=0) == []
    with pytest.raises(ValueError, match="nonnegative"):
        read_jsonl_safe(path, limit=-1)


def test_tail_does_not_scan_the_log_prefix():
    class MeasuredFile(io.BytesIO):
        bytes_read = 0

        def read(self, size=-1):
            result = super().read(size)
            self.bytes_read += len(result)
            return result

    source = MeasuredFile(b'{"old": true}\n' * 100_000 + b'{"latest": true}\n')
    assert _tail_lines(source, 1) == [b'{"latest": true}']
    assert source.bytes_read < source.getbuffer().nbytes // 10


@pytest.mark.parametrize("limit", [None, 100])
def test_jsonl_ignores_non_objects_and_incomplete_records(tmp_path, limit):
    path = tmp_path / "events.jsonl"
    path.write_bytes(b'null\n[]\n1\n"text"\n\n{"id": 1}\n\xff\n{"partial":')
    assert read_jsonl_safe(path, limit=limit) == [{"id": 1}]


@pytest.mark.parametrize("contents", [b"null", b"{", b"\xff"])
def test_manifest_reader_handles_invalid_or_non_object_json(tmp_path, contents):
    path = tmp_path / "manifest.json"
    path.write_bytes(contents)
    assert read_json_safe(path, {}) == {}
