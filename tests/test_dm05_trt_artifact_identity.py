"""Engine provenance must describe the bytes actually passed to TensorRT."""

import hashlib
from types import SimpleNamespace
from unittest.mock import MagicMock

from opendm.infer.dm05_trt_utils import deserialize_engine


def test_engine_identity_survives_path_replacement(tmp_path):
    path = tmp_path / "vision.engine"
    original = b"loaded engine generation"
    path.write_bytes(original)
    loaded_engine = object()

    def load_and_replace(payload):
        assert payload == original
        path.write_bytes(b"next engine generation")
        return loaded_engine

    runtime = MagicMock()
    runtime.__enter__.return_value = runtime
    runtime.deserialize_cuda_engine.side_effect = load_and_replace
    trt = SimpleNamespace(Logger=MagicMock(), Runtime=MagicMock(return_value=runtime))
    identity = {}

    _, engine = deserialize_engine(trt, path, "test", artifact_record=identity)

    assert engine is loaded_engine
    assert identity == {
        "bytes": len(original),
        "sha256": hashlib.sha256(original).hexdigest(),
    }
    assert hashlib.sha256(path.read_bytes()).hexdigest() != identity["sha256"]
