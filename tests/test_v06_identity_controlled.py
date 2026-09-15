import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_locked_v06_identity_controlled.py"
EXPECTED = "153e2af5c2e5578ae54094e8d7d43ca8536b5976a27901a56f086b42ba581aac"


def _module():
    spec = importlib.util.spec_from_file_location("build_v06", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_v06_deterministic_hash_count_and_invariants(tmp_path):
    mod = _module()
    report = mod.build(120, tmp_path)
    raw = (tmp_path / "items.jsonl").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED
    assert report["items"] == 840
    items = [json.loads(x) for x in raw.decode().splitlines() if x.strip()]
    assert len({x["item_id"] for x in items}) == 840
    counts = {}
    for item in items:
        counts[item["family"]] = counts.get(item["family"], 0) + 1
        assert item["correct_option_0"] != item["correct_option_1"]
        assert item["history_0"] != item["history_1"]
        assert item["owner_alias"] in item["history_0"]
        assert item["owner_alias"] in item["history_1"]
        assert item["identity_self"] != item["identity_other"]
        assert item["owner_alias"] in item["identity_self"]
        assert item["owner_alias"] in item["identity_other"]
        assert item["foil_alias"] in item["identity_self"]
        assert item["foil_alias"] in item["identity_other"]
    assert len(counts) == 7 and set(counts.values()) == {120}
