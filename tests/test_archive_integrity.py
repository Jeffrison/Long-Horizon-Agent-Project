import hashlib
import json
import pytest
from eval.verify_archives import verify


def test_artifact_changes_cannot_pass_manifest_check(tmp_path):
    path=tmp_path/"evidence.json"
    path.write_bytes(b"original")
    (tmp_path/"artifact-sha256.json").write_text(json.dumps({path.name:hashlib.sha256(path.read_bytes()).hexdigest()}))
    assert verify(tmp_path)==1
    path.write_bytes(b"changed")
    with pytest.raises(ValueError,match="checksum mismatch"):
        verify(tmp_path)


def test_manifest_cannot_reach_outside_artifact_directory(tmp_path):
    (tmp_path/"artifact-sha256.json").write_text(json.dumps({"../outside":"invalid"}))
    with pytest.raises(ValueError,match="invalid artifact path"):
        verify(tmp_path)
