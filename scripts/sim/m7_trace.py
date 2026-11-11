"""Mission 7 end-to-end trace on the virtual device: real backend, real runtime, app-side HTTP only.

    python scripts/sim/m7_trace.py --out <dir>

Drives the flows of docs/M7_IMPLEMENTATION_MAP.md through the same endpoints the
iPhone app calls (no direct runtime calls, no injected commands) and records:
every app request and its status, every command the backend returned to the
device on /v1/ingest, the device-reported state after each step, new events,
and the shared detector's call count against frames analysed. Writes
m7_trace.json and m7_trace.md. Everything is SIMULATED: scripted perception,
simulated clock; it is evidence that the software path works, not of accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sim.harness import SimStack
from sim.scenario import (
    ATTENTIVE,
    CALIBRATION_S,
    PHONE_IN_LAP,
    Faults,
    Scenario,
    Segment,
)

C = CALIBRATION_S
PROMPT = "Alert me if someone stays in the marked area for 10 seconds"
# A scripted timeline (seconds since boot). The scene changes on its own; the
# trace only acts through the app's endpoints.
SCRIPT = Scenario("m7_trace", "M7 end-to-end trace", (
    Segment(0.0, ATTENTIVE),
    Segment(C + 30, PHONE_IN_LAP),
    Segment(C + 40, ATTENTIVE),
    Segment(C + 55, ATTENTIVE, Faults(detector_failing=True)),
    Segment(C + 70, ATTENTIVE),
    Segment(C + 200, ATTENTIVE, Faults(network_down=True)),
    Segment(C + 215, ATTENTIVE),
))


class Tap:
    """Records the command in every /v1/ingest response (the device's view)."""

    def __init__(self, inner, trace):
        self._inner, self._trace = inner, trace

    def post(self, url, **kwargs):
        response = self._inner.post(url, **kwargs)
        try:
            command = (response.json() or {}).get("command") or {}
        except ValueError:
            command = {}
        if command:
            self._trace.command({k: v for k, v in command.items() if k != "command_auth"},
                                signed="command_auth" in command)
        return response

    def __getattr__(self, name):
        return getattr(self._inner, name)


class Trace:
    def __init__(self, stack):
        self.stack, self.steps, self._pending_cmds, self._seen_events = stack, [], [], 0

    def command(self, command, signed):
        self._pending_cmds.append({"keys": sorted(command), "signed": signed,
                                   **{k: command[k] for k in ("set_preset", "set_nodes", "rules_version",
                                                              "zones_version", "set_protection") if k in command}})

    def call(self, method, path, **kwargs):
        response = getattr(self.stack.api, method)(path, headers=self.stack.headers, **kwargs)
        try:
            body = response.json()
        except ValueError:
            body = None
        return response.status_code, body

    def snapshot(self):
        row = self.stack.row()
        status = row.get("status") or {}
        device = self.stack.device
        perception = status.get("perception") or {}
        caps = status.get("capabilities") or {}
        return {
            "t_s": round(device.scenario_clock.now(), 1),
            "active_preset": row.get("active_preset"),
            "monitoring_state": status.get("monitoring_state"),
            "reasons": status.get("reasons"),
            "impaired": caps.get("impaired"),
            "components": {k: v.get("state") for k, v in (perception.get("components") or {}).items()},
            "nodes": {n["id"]: n.get("state") for n in perception.get("nodes") or []},
            "rules_active": (status.get("rules") or {}).get("active"),
            "last_rejection": status.get("last_rejection"),
            "stale": status.get("stale"),
        }

    def step(self, name, *, calls=(), note=None):
        events = self.stack.events()  # oldest first
        new = events[self._seen_events:]
        self._seen_events = len(events)
        entry = {"step": name, "calls": list(calls), "commands_to_device": self._pending_cmds,
                 "state": self.snapshot(), "new_events": sorted({e["type"] for e in new})}
        if note:
            entry["note"] = note
        self._pending_cmds = []
        self.steps.append(entry)
        print(json.dumps(entry, default=str)[:600], flush=True)
        return entry


def until(device, predicate, limit_s=30.0, step_s=1.0):
    waited = 0.0
    while waited < limit_s:
        if predicate():
            return waited
        device.advance(step_s)
        waited += step_s
    return None


def run(out: Path) -> dict:
    with tempfile.TemporaryDirectory() as tmp, SimStack(Path(tmp)) as stack:
        trace = Trace(stack)
        device_id = stack.claim()
        device = stack.boot(SCRIPT, transport=Tap(stack.transport, trace))
        dev = f"/v1/devices/{device_id}"
        device.advance(C + 2)
        trace.step("boot + calibration (role driver_safety)")

        # ---- A. the Driver Safety ("Distracted Driving") card ---------------------------
        presets = trace.call("get", f"{dev}/presets")
        tiers = trace.call("get", "/v1/capability-tiers")
        ds = next((p for p in (presets[1] or {}).get("presets", []) if p.get("id") == "driver_safety"), {})
        trace.step("app opens the card: capabilities and tier", calls=[
            {"GET": f"{dev}/presets", "status": presets[0],
             "driver_safety": {k: ds.get(k) for k in ("available", "ready", "capability_states", "validation",
                                                      "device_validation") if k in ds}},
            {"GET": "/v1/capability-tiers", "status": tiers[0],
             "driver_safety_node": next((n for n in ((tiers[1] or {}).get("nodes") or [])
                                         if isinstance(n, dict) and n.get("id") == "driver_safety"), None)}])
        code, body = trace.call("post", f"{dev}/preset", json={"preset": "driver_safety:strict"})
        waited = until(device, lambda: stack.row().get("active_preset") == "driver_safety:strict"
                       and stack.status().get("monitoring_state") != "failed")
        trace.step("activate Driver Safety (strict) and wait for the device's acknowledgment",
                   calls=[{"POST": f"{dev}/preset", "body": {"preset": "driver_safety:strict"}, "status": code,
                           "response": body}], note=f"acknowledged after {waited} s of device syncs")
        device.advance(C + 42 - device.scenario_clock.now())
        trace.step("scripted distraction: phone in hand, looking down (C+30..C+40)")
        device.advance(C + 62 - device.scenario_clock.now())
        trace.step("detector fault injected at C+55 (scripted)")
        device.advance(C + 110 - device.scenario_clock.now())
        trace.step("detector restored at C+70; module retry backoff elapsed")
        history = trace.call("get", f"{dev}/events?limit=200")
        trace.step("app event history", calls=[{"GET": f"{dev}/events", "status": history[0],
                                                 "types": sorted({e["type"] for e in (history[1] or {}).get("events", [])})}])

        # ---- D. safety floor ------------------------------------------------------------
        code, body = trace.call("post", f"{dev}/preset", json={"preset": "elderly_monitoring"})
        device.advance(3.0)
        trace.step("try to replace driver protection with an incompatible mode",
                   calls=[{"POST": f"{dev}/preset", "body": {"preset": "elderly_monitoring"}, "status": code,
                           "response": body}])

        # ---- B/C. companion node + Jupiter custom rule beside the preset -----------------
        calls = []
        code, body = trace.call("post", f"{dev}/nodes", json={"node": "people_counter", "enabled": True})
        calls.append({"POST": f"{dev}/nodes", "body": {"node": "people_counter", "enabled": True},
                      "status": code, "response": body})
        code, preview = trace.call("post", "/v1/ai/configure", json={"device_id": device_id, "prompt": PROMPT})
        calls.append({"POST": "/v1/ai/configure", "prompt": PROMPT, "status": code,
                      "rules": (preview or {}).get("rules"), "explanation": (preview or {}).get("explanation"),
                      "unsupported": (preview or {}).get("unsupported")})
        zones = sorted({r.get("zone") for r in (preview or {}).get("rules") or [] if r.get("zone")})
        for zone in zones:
            polygon = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
            code_z, body_z = trace.call("post", f"{dev}/zones", json={"zones": [
                {"name": zone, "polygon": polygon, "kind": "restricted"}]})
            calls.append({"POST": f"{dev}/zones", "zone": zone, "status": code_z, "response": body_z})
        code, body = trace.call("post", f"{dev}/preset", json={
            "preset": "", "draft_version": (preview or {}).get("draft_version"),
            "rules": (preview or {}).get("rules")})
        calls.append({"POST": f"{dev}/preset", "body": {"preset": "", "draft_version": "…", "rules": "as reviewed"},
                      "status": code, "response": body})
        until(device, lambda: any(n["id"] == "rules"
                                  for n in ((stack.status().get("perception") or {}).get("nodes") or [])))
        trace.step("enable the people counter and activate a Jupiter custom rule beside Driver Safety", calls=calls)

        runtime = device.runtime
        calls0, frames0 = device.sim.detector.calls, runtime.perception.frames
        device.advance(30.0)
        calls_n, frames_n = device.sim.detector.calls - calls0, runtime.perception.frames - frames0
        report = runtime.monitoring_report().to_dict()["perception"]
        detector = report["components"]["object_detector"]
        shared = {"frames": frames_n, "detector_runs": calls_n,
                  "consumers": detector.get("consumers"), "nodes": [n["id"] for n in report["nodes"]]}
        trace.step("shared inference over 30 s with preset + rules + companion active",
                   note=f"detector runs {calls_n} over {frames_n} frames; consumers {detector.get('consumers')}")

        # ---- E. duplicates, outage, restart ---------------------------------------------
        node_before = runtime._companions.get("people_counter")
        code, body = trace.call("post", f"{dev}/nodes", json={"node": "people_counter", "enabled": True})
        device.advance(4.0)
        trace.step("duplicate node request", calls=[{"POST": f"{dev}/nodes", "status": code, "response": body}],
                   note=f"running node rebuilt: {device.runtime._companions.get('people_counter') is not node_before}")
        device.advance(C + 208 - device.scenario_clock.now())
        trace.step("network outage (C+200..C+215, scripted)")
        device.advance(C + 240 - device.scenario_clock.now())
        trace.step("network restored")
        device = stack.restart(SCRIPT, transport=Tap(stack.transport, trace))
        device.advance(C + 6)
        trace.step("device restart: preset, rules and nodes restored from device state")

        result = {"trace": trace.steps, "shared_inference": shared, "simulated": True}
    out.mkdir(parents=True, exist_ok=True)
    (out / "m7_trace.json").write_text(json.dumps(result, indent=1, default=str))
    lines = ["# M7 virtual-device trace (SIMULATED perception)", ""]
    for s in result["trace"]:
        st = s["state"]
        lines += [f"## {s['step']}", "",
                  (f"- state: preset `{st['active_preset']}`, monitoring `{st['monitoring_state']}`, "
                   f"components {st['components']}, nodes {st['nodes']}, rules {st['rules_active']}, "
                   f"stale {st['stale']}")]
        if st.get("reasons"):
            lines.append(f"- reasons: {st['reasons']}")
        for c in s["calls"]:
            lines.append(f"- app call: `{json.dumps(c, default=str)[:400]}`")
        for c in s["commands_to_device"]:
            lines.append(f"- backend → device: `{json.dumps(c, default=str)[:300]}`")
        if s["new_events"]:
            lines.append(f"- new events: {s['new_events']}")
        if s.get("note"):
            lines.append(f"- {s['note']}")
        lines.append("")
    (out / "m7_trace.md").write_text("\n".join(lines))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    r = run(Path(a.out))
    print(json.dumps(r["shared_inference"]))
