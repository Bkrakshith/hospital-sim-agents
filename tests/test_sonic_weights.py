"""GEAR-SONIC weights: cache location, the doctor check and scripts/fetch_sonic.py (nothing is downloaded)."""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

from wardzero_agents import doctor
from wardzero_agents.isaac_host.sonic import SonicController
from wardzero_agents.isaac_host.sonic import weights as W

ROOT = Path(__file__).resolve().parents[1]


def _fetch_module():
    spec = importlib.util.spec_from_file_location("fetch_sonic_under_test", ROOT / "scripts" / "fetch_sonic.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _tiny_files(monkeypatch, *mods):
    """Swap the pinned files for three tiny ones with known hashes."""
    files, blobs = [], {}
    for i, rel in enumerate(["policy/sonic_v1_1/model_encoder.onnx", "policy/sonic_v1_1/model_decoder.onnx",
                             "planner/target_vel/V2/planner_sonic.onnx"]):
        data = f"weights {i}".encode()
        files.append(W.WeightFile(f"hf/{i}", rel, len(data), hashlib.sha256(data).hexdigest()))
        blobs[rel] = data
    for m in (W, *mods):
        monkeypatch.setattr(m, "FILES", tuple(files))
    return blobs


def _write(root: Path, blobs: dict):
    for rel, data in blobs.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)


def test_pinned_files_match_the_upstream_layout_and_hashes():
    assert {f.rel for f in W.FILES} == {"policy/sonic_v1_1/model_encoder.onnx", "policy/sonic_v1_1/model_decoder.onnx",
                                        "planner/target_vel/V2/planner_sonic.onnx",
                                        "policy/sonic_v1_1/observation_config.yaml"}
    assert all(len(f.sha256) == 64 for f in W.FILES)
    assert W.PLANNER.sha256.startswith("39b553e1") and W.PLANNER.sha256.endswith("60edea")  # sonic_prep.md 1
    assert W.ENCODER.sha256.startswith("fb97de22") and W.DECODER.sha256.startswith("34bae857")
    assert 900e6 < W.TOTAL_BYTES < 1e9


def test_cache_dir_is_outside_the_repo_and_follows_the_env(monkeypatch, tmp_path):
    monkeypatch.delenv(W.ENV_DIR, raising=False)
    d = W.resolve_dir()
    assert d == Path("~/.cache/wardzero/sonic").expanduser() and ROOT not in d.parents
    monkeypatch.setenv(W.ENV_DIR, str(tmp_path))
    assert W.resolve_dir() == tmp_path
    assert W.resolve_dir(tmp_path / "x") == tmp_path / "x"


def test_doctor_reports_missing_weights_with_the_fetch_command(monkeypatch, tmp_path):
    blobs = _tiny_files(monkeypatch)
    [c] = doctor.check_sonic(tmp_path)
    assert c.status == doctor.NOTE and str(tmp_path) in c.what
    assert any("scripts/fetch_sonic.py" in step for step in c.fix)
    _write(tmp_path, blobs)
    ok = doctor.check_sonic(tmp_path, have_ort=lambda: True)
    assert [x.status for x in ok] == [doctor.OK]
    no_ort = doctor.check_sonic(tmp_path, have_ort=lambda: False)
    assert [x.status for x in no_ort] == [doctor.OK, doctor.NOTE]
    (tmp_path / "planner/target_vel/V2/planner_sonic.onnx").write_bytes(b"cut")
    assert doctor.check_sonic(tmp_path)[0].status == doctor.NOTE  # wrong size counts as missing


def test_controller_without_weights_says_how_to_fetch_them(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_sonic.py"):
        SonicController.from_weights(tmp_path)


def test_fetch_use_existing_verifies_and_links_without_copying(monkeypatch, tmp_path):
    fetch = _fetch_module()
    blobs = _tiny_files(monkeypatch, fetch)
    src, dest = tmp_path / "gear_sonic_weights", tmp_path / "cache"
    _write(src, blobs)
    monkeypatch.setattr(fetch, "download", lambda *a: pytest.fail("must not download"))
    monkeypatch.setattr(sys, "argv", ["fetch_sonic.py", "--dest", str(dest), "--use-existing", str(src)])
    assert fetch.main() == 0
    for rel in blobs:
        assert (dest / rel).is_symlink() and (dest / rel).resolve() == (src / rel).resolve()
    assert not W.missing_files(dest)
    monkeypatch.setattr(sys, "argv", ["fetch_sonic.py", "--dest", str(dest), "--check"])
    assert fetch.main() == 0
    (src / "policy/sonic_v1_1/model_decoder.onnx").write_bytes(b"weights X")  # same size, other bytes
    assert fetch.main() == 1
    monkeypatch.setattr(sys, "argv", ["fetch_sonic.py", "--dest", str(dest), "--use-existing", str(src)])
    assert fetch.main() == 1


def test_fetch_downloads_only_what_is_missing(monkeypatch, tmp_path):
    fetch = _fetch_module()
    blobs = _tiny_files(monkeypatch, fetch)
    have = "policy/sonic_v1_1/model_encoder.onnx"
    _write(tmp_path, {have: blobs[have]})
    got = []

    def fake_download(wf, target):
        got.append(wf.rel)
        target.write_bytes(blobs[wf.rel])

    monkeypatch.setattr(fetch, "download", fake_download)
    monkeypatch.setattr(sys, "argv", ["fetch_sonic.py", "--dest", str(tmp_path)])
    assert fetch.main() == 0
    assert sorted(got) == sorted(set(blobs) - {have})
    assert not W.missing_files(tmp_path)


def test_notice_credits_gear_sonic():
    text = (ROOT / "NOTICE").read_text()
    assert "Licensed by NVIDIA Corporation under the NVIDIA Open Model License." in text
    assert "GR00T-WholeBodyControl" in text and "Apache License 2.0" in text and "arXiv:2511.07820" in text
