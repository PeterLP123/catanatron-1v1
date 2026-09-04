import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from catanatron.file_utils import write_json_atomic


def test_failed_validation_preserves_previous_file_and_cleans_up(tmp_path):
    target = tmp_path / "manifest.json"
    target.write_text('{"previous": true}\n')
    previous = target.read_bytes()

    def reject(path):
        assert path.parent == target.parent
        assert json.loads(path.read_text()) == {"replacement": True}
        assert target.read_bytes() == previous
        raise ValueError("invalid model")

    with pytest.raises(ValueError, match="invalid model"):
        write_json_atomic(target, {"replacement": True}, validate=reject)
    assert target.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [target]


def test_concurrent_manifest_builds_publish_exactly_one_complete_file(tmp_path):
    target = tmp_path / "manifest.json"
    ready = threading.Barrier(2)

    def publish(writer):
        try:
            write_json_atomic(
                target,
                {"writer": writer},
                overwrite=False,
                validate=lambda path: ready.wait(timeout=5),
            )
        except FileExistsError:
            return None
        return writer

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(publish, (1, 2)))
    winners = [result for result in results if result is not None]
    assert len(winners) == 1
    assert json.loads(target.read_text()) == {"writer": winners[0]}
    assert list(tmp_path.iterdir()) == [target]


def test_manifest_publication_rejects_dangling_symlink(tmp_path):
    target = tmp_path / "manifest.json"
    target.symlink_to(tmp_path / "missing.json")
    with pytest.raises(FileExistsError):
        write_json_atomic(target, {}, overwrite=False)
    assert target.is_symlink()
    assert not target.exists()
