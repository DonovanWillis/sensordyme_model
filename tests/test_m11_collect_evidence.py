"""M11 Phase C: the one evidence command (scripts/hw_validation/collect_evidence.py).

Pins what a remote collaborator's bundle may and may not contain: SIMULATED is never PASS, the
report is built from allow-listed fields, and no secret survives into any written file.
"""

from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "hw_validation"))

import collect_evidence as ce  # noqa: E402

TOKEN = "0123456789abcdef0123456789abcdef"  # the shape of `openssl rand -hex 16`
DEVICE_KEY = "dk_live_9f8e7d6c5b4a39281706"


def _run(tmp_path: Path, *extra: str, system=None, env=None) -> tuple[int, Path]:
    out = tmp_path / "out"
    code = ce.main(["--fake", "--minutes", "0.02", "--interval", "0.3", "--out", str(out),
                    "--results", str(tmp_path / "results"), *extra],
                   system=system, env=env or {}, sleep=lambda _s: None)
    folders = [p for p in out.iterdir() if p.is_dir()]
    assert len(folders) == 1
    return code, folders[0]


def test_dry_run_is_simulated_never_pass(tmp_path):
    code, folder = _run(tmp_path)
    assert code == ce.EXIT[ce.SIMULATED] == 3
    report = json.loads((folder / "report.json").read_text())
    assert report["verdict"] == "SIMULATED"
    assert report["simulated"] is True and report["is_pi_measurement"] is False
    assert "NOT hardware evidence" in report["evidence_level"]
    assert folder.name.endswith("_SIMULATED")
    assert "DRY RUN" in (folder / "report.md").read_text()


def test_report_carries_every_requested_section(tmp_path):
    _code, folder = _run(tmp_path)
    r = json.loads((folder / "report.json").read_text())
    for key in ("platform", "accelerator", "camera", "drivers", "software", "model", "decoder", "latency_ms",
                "processing_fps", "cpu_memory_thermal", "monitoring", "tests", "remote_vs_local", "privacy"):
        assert key in r, key
    assert r["decoder"]["configured"] == "legacy"
    assert r["decoder"]["every_n_frames"] == 3
    assert r["model"]["pin"].startswith("e69136d3")
    assert set(r["latency_ms"]) >= {"hailo_inference", "decoder_cpu", "face_landmarks", "frame_to_decision"}
    host = r["cpu_memory_thermal"]
    assert host["cpu_util_pct"]["n"] >= 1 and host["soc_temp_c"]["n"] >= 1 and host["service_rss_mb"]["n"] >= 1
    assert r["camera"]["configured"] == {"width": 640, "height": 480, "fps": 30, "rotate_180": False}
    assert r["privacy"] == {"frames_collected": False, "video_or_audio_collected": False, "config_dump": False,
                            "health_dump": False, "hostname_recorded": False}
    assert any("LOCAL" in x["where"] for x in r["remote_vs_local"])
    assert any("REMOTE" in x["where"] for x in r["remote_vs_local"])


def test_bundle_hashes_match(tmp_path):
    _code, folder = _run(tmp_path)
    sums = dict(line.split("  ")[::-1] for line in (folder / "SHA256SUMS").read_text().splitlines())
    for name, digest in sums.items():
        assert ce.hc.sha256_file(folder / name) == digest
    bundle = folder.with_suffix(".tgz")
    sidecar = (folder.parent / f"{bundle.name}.sha256").read_text().split()[0]
    assert sidecar == ce.hc.sha256_file(bundle)
    with tarfile.open(bundle) as tar:
        assert {Path(n).name for n in tar.getnames()} >= {"report.json", "report.md", "SHA256SUMS"}


class LeakySystem(ce.EvidenceFakeSystem):
    """A device whose answers contain secrets in every place a careless report would copy them."""

    def health(self, stopped: bool = False) -> dict:
        h = super().health(stopped)
        dev = h["device"]
        dev["monitoring"]["reasons"] = [f"uplink refused key {DEVICE_KEY}", "joined psk=hunter2-wifi on 192.168.1.23",
                                        f"token={TOKEN}", "owner owner@example.com", "mac aa:bb:cc:dd:ee:ff",
                                        "claim 1A2B-3C4D-5E6F"]
        dev["network"] = {"state": "ok", "detail": "HomeNetwork-SSID", "metrics": {"ssid": "HomeNetwork-SSID"}}
        return h


def test_no_secret_reaches_any_written_file(tmp_path):
    env = {"SENSORDYME_TOKEN": TOKEN, "SENSORDYME_BEARER": "bearer-session-value-XYZ123"}
    _code, folder = _run(tmp_path, system=LeakySystem(Path("/home/pi/sensordyme")), env=env)
    blobs = [p.read_bytes() for p in folder.iterdir()]
    with tarfile.open(folder.with_suffix(".tgz")) as tar:
        blobs += [tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()]  # type: ignore[union-attr]
    for blob in blobs:
        for secret in (TOKEN, DEVICE_KEY, "hunter2-wifi", "192.168.1.23", "owner@example.com", "aa:bb:cc:dd:ee:ff",
                       "1A2B-3C4D-5E6F", "HomeNetwork-SSID", "bearer-session-value-XYZ123", "5151515151515151"):
            assert secret.encode() not in blob, secret
    report = json.loads((folder / "report.json").read_text())
    assert report["sanitizer"]["redactions"], "the redactions must be counted"


def test_sanitizer_keeps_hashes_and_commits():
    s = ce.Sanitizer({"token": TOKEN})
    sha = "e69136d3ebf3cc8ebaf3da5417a59962b139f3d3d67d715e9a9b1ae10963960e"
    commit = "5ed3e5966b53a7830abe4adcaa88d187d0a0f912"
    out = s.clean({"model": sha, "commit": commit, "pci": "0001:01:00.0", "kernel": "6.6.51+rpt-rpi-2712",
                   "loopback": "http://127.0.0.1:8899", "leak": f"X-Device-Token {TOKEN}"})
    assert out["model"] == sha and out["commit"] == commit and out["pci"] == "0001:01:00.0"
    assert out["loopback"] == "http://127.0.0.1:8899"
    assert TOKEN not in out["leak"] and "[REDACTED:token]" in out["leak"]
    assert s.hits == {"token": 1}


def test_config_facts_drop_credentials(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "role: driver_safety\nmodule: driver_attention\n"
        "camera: {width: 640, height: 480, fps: 30, rotate_180: false}\n"
        "accelerator: {yolo_decoder: legacy, yolo_hef_sha256: abc, yolo_hef: models/x.hef}\n"
        f"cloud: {{enabled: true, device_key: {DEVICE_KEY}, base_url: 'http://backend.local:8787', command_auth: permissive}}\n"
        f"dashboard: {{token: {TOKEN}, host: 0.0.0.0}}\n"
        "network: {provisioning: {auth: legacy, setup_code_file: /x, wifi_psk: hunter2}}\n"
        "modules: {driver_attention: {yolo_every_n_frames: 3}}\n")
    facts = ce.config_facts(cfg)
    text = json.dumps(facts)
    for secret in (DEVICE_KEY, TOKEN, "backend.local", "hunter2", "setup_code_file", "0.0.0.0"):  # noqa: S104 - asserted ABSENT
        assert secret not in text, secret
    assert facts["accelerator"]["yolo_decoder"] == "legacy"
    assert facts["cloud"] == {"enabled": True, "command_auth": "permissive"}
    assert facts["network"] == {"provisioning": {"auth": "legacy"}}
    assert facts["modules.driver_attention"] == {"yolo_every_n_frames": 3}
    assert "dashboard" not in facts


def test_results_rollup_reads_stage_preflight_and_cabin_reports(tmp_path):
    d = tmp_path / "results"
    (d / "stages").mkdir(parents=True)
    (d / "stages" / "parity_1.json").write_text(json.dumps(
        {"stage": "parity", "verdict": "FAIL", "simulated": False, "utc": "t",
         "checks": [{"check": "[reference] agreement", "ok": False}, {"check": "x", "ok": True}]}))
    (d / "sunday_preflight_1.json").write_text(json.dumps(
        {"tool": "sunday_preflight", "verdict": "BLOCKED", "finished": "t",
         "checks": [{"id": "camera_listing", "status": "BLOCKED"}]}))
    (d / "unrelated.json").write_text(json.dumps({"hello": 1}))
    rows = ce.results_rollup([d])
    kinds = {r["kind"]: r for r in rows}
    assert set(kinds) == {"hwv_stage", "preflight"}
    assert kinds["hwv_stage"]["failed_checks"] == ["[reference] agreement"]
    assert kinds["preflight"]["failed_checks"] == ["camera_listing"]
    assert all(len(r["sha256"]) == 64 for r in rows)


def test_refuses_to_write_inside_the_install_tree(tmp_path):
    install = tmp_path / "sensordyme"
    install.mkdir()
    code = ce.main(["--fake", "--install-dir", str(install), "--out", str(install / "evidence")], env={},
                   sleep=lambda _s: None)
    assert code == ce.EXIT[ce.BLOCKED]
    assert not (install / "evidence").exists()


@pytest.mark.parametrize(("key", "denied"), [("yolo_decoder", False), ("device_key", True), ("base_url", True),
                                        ("setup_code_file", True), ("yolo_hef_sha256", False), ("token", True)])
def test_deny_rule_is_word_based(key, denied):
    assert ce._denied(key) is denied
