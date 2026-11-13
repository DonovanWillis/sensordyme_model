"""M8.4 acceptance: every meaningful digital-twin suite, one command, one evidence summary.

    make acceptance                       # = python scripts/sim/acceptance.py --out data/acceptance
    python scripts/sim/acceptance.py --out DIR [--quick] [--only NAME ...] [--list]

Runs, each in its own process (the driver suites load real MediaPipe graphs;
one process for everything once exceeded a 1 GiB fleet job cap):

  * the deterministic virtual-device suites (SimClock: journeys, failure
    injection, reliability, storage integrity, truth invariants, events,
    lifecycle, onboarding, streaming, security end to end);
  * a short sim-clock soak (scripts/sim/twin_soak.py simclock): device-vs-backend
    event integrity and the InvariantMonitor over a lossy link;
  * the live /stream measurement, short (scripts/sim/stream_soak.py), judged only
    on completing and delivering frames -- no timing thresholds;
  * the Swift onboarding harness against the simulated BLE peripheral
    (python -m sim.ble_harness --all), where swiftc exists.

Writes DIR/acceptance.json and DIR/acceptance.md: per suite pass/fail/skip
counts, duration, exit code; the commit (and whether the tree was dirty); the
host platform; and `simulated: true` -- none of this is a hardware result.
Exit status is nonzero if any suite failed. Skips are reported, never hidden.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[2]


@dataclass
class Suite:
    name: str
    what: str
    tests: tuple[str, ...] = ()
    command: tuple[str, ...] = ()
    #: Skip (reported) unless this is true on the host.
    needs: Optional[str] = None
    quick: bool = True


SUITES: tuple[Suite, ...] = (
    # ---- M8.4 (this stream)
    Suite("twin_journeys", "full user journeys through the app's API, invariants attached",
          ("tests/test_m84_twin_journeys.py", "tests/test_m84_twin_fidelity.py")),
    Suite("twin_faults", "lossy/duplicate/reordered link, malformed NPU tensors, ack loss, backend restart, "
                         "read-only data dir", ("tests/test_m84_twin_faults.py",)),
    Suite("reliability_recovery", "repeated inference failure, fault storms, no duplicate actuation",
          ("tests/test_m84_reliability_recovery.py",)),
    Suite("reliability_storage", "full card on a real volume, writers, spool migration, retention",
          ("tests/test_m84_reliability_storage.py",)),
    # ---- the virtual device and its truth model (M6.5 - M8.3)
    Suite("sim_e2e", "virtual device x real backend, end to end", ("tests/test_sim_e2e.py",)),
    Suite("sim_reliability", "multi-stage reliability scenarios", ("tests/test_sim_reliability.py",)),
    Suite("sim_stress", "backlog, retry storm, races, restarts", ("tests/test_sim_stress.py",)),
    Suite("sim_units_seams", "sim building blocks and production seams",
          ("tests/test_sim_units.py", "tests/test_sim_seams.py")),
    Suite("truth_failure_matrix", "M8.3 failure matrix rows", ("tests/test_m83_truth_failure_matrix.py",)),
    Suite("truth_invariants_ack", "truth invariants and protection acknowledgements",
          ("tests/test_m83_truth_invariants.py", "tests/test_m83_truth_protection_ack.py")),
    Suite("health_freshness", "stale / reordered / frozen health", ("tests/test_m81_health_freshness_sim.py",)),
    Suite("events", "event pipeline: delivery evidence, dedupe, clocks",
          ("tests/test_m83_events_sim.py", "tests/test_m83_events_backend.py", "tests/test_m83_events_delivery.py")),
    Suite("lifecycle", "boot, LKG config, spool, power cuts",
          ("tests/test_m8_lifecycle_e2e.py", "tests/test_m8_lifecycle_boot.py", "tests/test_m8_lifecycle_uplink.py")),
    Suite("nodes_and_modes", "shared perception, driver activation, nodes",
          ("tests/test_m7_shared_perception_sim.py", "tests/test_m8_driver_activation_sim.py",
           "tests/test_m8_nodes_e2e.py")),
    Suite("security_e2e", "credentials, command replay across restarts", ("tests/test_m8_security_e2e.py",)),
    Suite("stream", "/stream framing, liveness, viewers", ("tests/test_m83_stream_device.py",
                                                            "tests/test_m83_stream_runtime.py")),
    Suite("onboarding", "BLE provisioning peripheral (+ Swift app code where swiftc exists)",
          ("tests/test_m8_onboarding_peripheral.py", "tests/test_m8_onboarding_setup_code.py",
           "tests/test_m8_onboarding_crosslang.py", "tests/test_m8_onboarding_e2e.py")),
    # ---- scripts
    Suite("soak_simclock_short", "0.35 h of device time on the SimClock through a lossy link",
          command=("scripts/sim/twin_soak.py", "simclock", "--hours", "0.35", "--out", "{out}/soak_simclock")),
    Suite("stream_soak_short", "live /stream on the virtual device (measurement only, no thresholds)",
          command=("scripts/sim/stream_soak.py", "--out", "{out}/stream", "--seconds", "6",
                   "--soak-minutes", "0.5", "--only", "attentive"), quick=False),
    Suite("ble_harness", "the app's onboarding code vs the simulated peripheral, every scenario",
          command=("-m", "sim.ble_harness", "--all", "--evidence-dir", "{out}/ble"), needs="swiftc", quick=False),
)


def _commit() -> dict:
    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                                  timeout=20).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    return {"commit": git("rev-parse", "HEAD") or os.environ.get("SDJOB_COMMIT", "unknown"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def _junit(path: Path) -> dict:
    counts = {"passed": 0, "failed": 0, "skipped": 0, "xfailed": 0, "errors": 0}
    failures: list[str] = []
    skips: list[str] = []
    try:
        root = ET.parse(path).getroot()  # noqa: S314 - our own junit output
    except (OSError, ET.ParseError):
        return {**counts, "failures": ["no junit report"], "skips": []}
    for case in root.iter("testcase"):
        name = f"{case.get('classname', '')}::{case.get('name', '')}"
        if case.find("failure") is not None:
            counts["failed"] += 1
            failures.append(name)
        elif case.find("error") is not None:
            counts["errors"] += 1
            failures.append(name)
        elif (skipped := case.find("skipped")) is not None:
            if "xfail" in (skipped.get("type", "") + skipped.get("message", "")).lower():
                counts["xfailed"] += 1
            else:
                counts["skipped"] += 1
                skips.append(f"{name}: {skipped.get('message', '')[:160]}")
        else:
            counts["passed"] += 1
    return {**counts, "failures": failures, "skips": skips}


def run_suite(suite: Suite, out: Path, py: str) -> dict:
    started = time.monotonic()
    log = out / "logs" / f"{suite.name}.txt"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "SD_M84_EVIDENCE_DIR": str(out / "journeys")}
    if suite.needs == "swiftc" and shutil.which("swiftc") is None:
        return {"name": suite.name, "what": suite.what, "result": "skipped", "reason": "needs swiftc (macOS)",
                "duration_s": 0.0}
    if suite.tests:
        totals = {"passed": 0, "failed": 0, "skipped": 0, "xfailed": 0, "errors": 0}
        failures: list[str] = []
        skips: list[str] = []
        codes = []
        for test in suite.tests:
            junit = out / "junit" / f"{Path(test).stem}.xml"
            junit.parent.mkdir(parents=True, exist_ok=True)
            with open(log, "a") as handle:
                code = subprocess.run([py, "-m", "pytest", "-p", "no:cacheprovider", "-q", "-rs",
                                       f"--junitxml={junit}", test], cwd=REPO, env=env, stdout=handle,
                                      stderr=subprocess.STDOUT, check=False).returncode
            codes.append(code)
            parsed = _junit(junit)
            for key in totals:
                totals[key] += parsed[key]
            failures += parsed["failures"]
            skips += parsed["skips"]
        # pytest exit 5 = nothing collected (e.g. a whole module skipped by importorskip)
        bad = [c for c in codes if c not in (0, 5)]
        result = "failed" if bad or totals["failed"] or totals["errors"] else (
            "passed" if totals["passed"] else "skipped")
        return {"name": suite.name, "what": suite.what, "result": result, **totals, "exit_codes": codes,
                "failures": failures[:30], "skips": skips[:30], "duration_s": round(time.monotonic() - started, 1),
                "log": str(log.relative_to(out))}
    argv = [py, *(a.format(out=out) for a in suite.command)]
    with open(log, "w") as handle:
        code = subprocess.run(argv, cwd=REPO, env=env, stdout=handle,
                              stderr=subprocess.STDOUT, check=False).returncode
    record = {"name": suite.name, "what": suite.what, "result": "passed" if code == 0 else "failed",
              "exit_code": code, "duration_s": round(time.monotonic() - started, 1), "log": str(log.relative_to(out))}
    record.update(_script_facts(suite, out))
    if record.get("problems"):
        record["result"] = "failed"
    return record


def _script_facts(suite: Suite, out: Path) -> dict:
    """The few numbers a script run is judged on (and reported with)."""
    if suite.name == "soak_simclock_short":
        try:
            data = json.loads((out / "soak_simclock" / "summary_simclock.json").read_text())
        except (OSError, ValueError):
            return {"problems": ["no soak summary"]}
        events = data.get("events", {})
        problems = []
        if events.get("lost") or events.get("duplicate_keys") or events.get("duplicate_type_timestamp"):
            problems.append(f"event integrity: {events}")
        if (data.get("invariants") or {}).get("violations"):
            problems.append(f"invariant violations: {data.get('violations')}")
        return {"facts": {"sim_seconds": data.get("sim_seconds_completed"), "events": events,
                          "invariants": data.get("invariants"), "link": data.get("link")}, "problems": problems}
    if suite.name == "stream_soak_short":
        try:
            data = json.loads((out / "stream" / "stream_measurements.json").read_text())
        except (OSError, ValueError):
            return {"problems": ["no stream measurements"]}
        text = json.dumps(data)
        return {"facts": {"bytes": len(text)}, "problems": [] if '"frames"' in text else ["no frames measured"]}
    return {}


def _fidelity() -> dict:
    """The twin's fidelity map (sim/fidelity.py): what was real, what was a stand-in."""
    sys.path.insert(0, str(REPO))
    from sim import fidelity

    return {"by_kind": fidelity.summary(),
            "stand_ins": {name: entry["fakes"] for name, entry in fidelity.FIDELITY.items()
                          if entry["code"] != "real"}}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=str(REPO / "data" / "acceptance"))  # data/ is git-ignored
    parser.add_argument("--quick", action="store_true", help="skip the live-time scripts (stream soak, BLE harness)")
    parser.add_argument("--only", nargs="*", default=None, help="suite names to run")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)
    if args.list:
        for suite in SUITES:
            print(f"{suite.name:24s} {suite.what}")
        return 0
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    chosen = [s for s in SUITES if (args.only is None or s.name in args.only) and (s.quick or not args.quick)]
    started = time.time()
    results = []
    for suite in chosen:
        record = run_suite(suite, out, args.python)
        results.append(record)
        counts = " ".join(f"{k}={record[k]}" for k in ("passed", "failed", "skipped", "xfailed") if k in record)
        print(f"{record['result'].upper():7s} {suite.name:24s} {record['duration_s']:7.1f}s {counts}", flush=True)
    summary = {
        "simulated": True,
        "label": "Digital twin (virtual device: scripted camera and models, real runtime/uplink/backend). "
                 "NOT a hardware result.",
        **_commit(), "platform": platform.platform(), "python": platform.python_version(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started)),
        "duration_s": round(time.time() - started, 1),
        "ok": all(r["result"] != "failed" for r in results),
        "fidelity": _fidelity(),
        "totals": {k: sum(int(r.get(k, 0) or 0) for r in results) for k in ("passed", "failed", "skipped", "xfailed")},
        "suites": results,
    }
    (out / "acceptance.json").write_text(json.dumps(summary, indent=1))
    lines = [f"# M8.4 acceptance -- {'PASS' if summary['ok'] else 'FAIL'} (SIMULATED)", "",
             (f"commit `{summary['commit']}`{' (dirty tree)' if summary['dirty'] else ''}, {summary['platform']}, "
              f"{summary['duration_s']} s. {summary['label']}"), "",
             "| suite | result | passed | failed | skipped | xfailed | s | what |", "|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['name']} | {r['result']} | {r.get('passed', '')} | {r.get('failed', '')} | "
                     f"{r.get('skipped', '')} | {r.get('xfailed', '')} | {r['duration_s']} | {r['what']} |")
    for r in results:
        for item in r.get("failures", []) + r.get("problems", []):
            lines.append(f"- FAIL {r['name']}: {item}")
        for item in r.get("skips", []):
            lines.append(f"- skip {r['name']}: {item}")
    (out / "acceptance.md").write_text("\n".join(lines) + "\n")
    t = summary["totals"]
    print(f"{'PASS' if summary['ok'] else 'FAIL'}: {t['passed']} passed, {t['failed']} failed, {t['skipped']} skipped, "
          f"{t['xfailed']} xfailed in {summary['duration_s']} s -> {out / 'acceptance.md'}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
