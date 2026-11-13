#!/usr/bin/env python3
"""Buzzer pattern check: drive each production alert pattern through the
production GPIOBuzzerSink and log when the GPIO was COMMANDED on and off.

RUNS ON THE PI with the service stopped (the service owns the buzzer pin).

What software can and cannot establish, stated plainly:
  * CAN: that the pin was claimed, that each on()/off() call returned, and when
    (commanded timing vs the pattern definition in core/alerts.py).
  * CANNOT: that anything was audible. There is no feedback line from the
    buzzer (docs/HARDWARE_VERIFICATION.md M2-2). Audibility, loudness and
    distinguishability need a person and a recording: the phone-audio / SPL
    procedure in docs/HAILO_HARDWARE_VALIDATION.md, whose annotations
    `analyse` below compares against the same expected timeline.
  * Electrical timing at the pin (rise time, a transistor stage) needs a logic
    analyser or scope on the pin; the commanded times here are the upper bound
    of what software knows.

    buzzer_check.py run --out ~/hwval/buzzer_$(date +%s).json [--patterns chirp,double]
    buzzer_check.py analyse --timeline buzzer.json --annotations audio.csv
        (CSV columns: pattern, step, onset_ms, offset_ms; times from the audio
         waveform, any origin: each pattern is aligned on its first onset)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import hwv_common as hc  # noqa: E402

CATEGORY_FOR_PATTERN = {"chirp": "notice", "single": "warning", "double": "warning",
                        "urgent": "warning", "critical": "critical", "impaired": "impaired"}


def expected_timeline(steps) -> list[tuple[float, float]]:
    """(on_s, off_s) steps -> [(on_at_ms, off_at_ms)] relative to the first onset."""
    t, out = 0.0, []
    for on_s, off_s in steps:
        out.append((t * 1000.0, (t + on_s) * 1000.0))
        t += on_s + off_s
    return out


def commanded_timeline(events: list[tuple[str, float]]) -> list[tuple[float, float]]:
    """[('on'|'off', t_s)] -> [(on_ms, off_ms)] relative to the first 'on'.

    Repeated 'on' while already on (overlapping requests) are folded, matching
    the sink's own _off_at logic.
    """
    out, start, on_at = [], None, None
    for kind, t in events:
        if start is None:
            if kind != "on":
                continue
            start = t
        if kind == "on" and on_at is None:
            on_at = t
        elif kind == "off" and on_at is not None:
            out.append(((on_at - start) * 1000.0, (t - start) * 1000.0))
            on_at = None
    return out


def compare(expected: list[tuple[float, float]], observed: list[tuple[float, float]]) -> dict:
    if len(expected) != len(observed):
        return {"steps_expected": len(expected), "steps_observed": len(observed), "ok": False}
    onset = [o[0] - e[0] for e, o in zip(expected, observed, strict=True)]
    dur = [(o[1] - o[0]) - (e[1] - e[0]) for e, o in zip(expected, observed, strict=True)]
    return {"steps_expected": len(expected), "steps_observed": len(observed),
            "onset_error_ms": [round(v, 1) for v in onset],
            "duration_error_ms": [round(v, 1) for v in dur],
            "max_abs_error_ms": round(max(abs(v) for v in onset + dur), 1), "ok": True}


class _RecordingBuzzer:
    def __init__(self, buzzer, log: list) -> None:
        self._buzzer, self._log, self._lock = buzzer, log, threading.Lock()

    def on(self) -> None:
        self._buzzer.on()
        with self._lock:
            self._log.append(("on", time.perf_counter()))

    def off(self) -> None:
        self._buzzer.off()
        with self._lock:
            self._log.append(("off", time.perf_counter()))

    def __getattr__(self, name):
        return getattr(self._buzzer, name)


def cmd_run(args) -> int:
    ok, why = hc.service_gate(hc.service_state(), allow_unknown=args.no_service_check)
    print(f"service: {why}")
    if not ok:
        print("REFUSED.\n" + hc.operator_commands(), file=sys.stderr)
        return 2
    import yaml

    sys.path.insert(0, str(Path(args.code_root).resolve()))
    from core.alerts import PATTERNS, GPIOBuzzerSink, _PatternPlayer

    config = yaml.safe_load((Path(args.install_dir) / "config.yaml").read_text()) or {}
    pin = int(args.pin if args.pin is not None else (config.get("alerts") or {}).get("gpio_buzzer_pin", 17))
    names = [p for p in args.patterns.split(",") if p] if args.patterns else list(PATTERNS)
    unknown = [n for n in names if n not in PATTERNS]
    if unknown:
        print(f"REFUSED: unknown pattern(s) {unknown}; known: {sorted(PATTERNS)}", file=sys.stderr)
        return 2
    sink = GPIOBuzzerSink(pin)
    results = {"utc": hc.utc_now(), "pin": pin, "patterns": {},
               "statement": "commanded GPIO timing only; audibility is NOT verified by software"}
    try:
        for name in names:
            log: list = []
            sink._buzzer = _RecordingBuzzer(sink._buzzer, log)
            player = _PatternPlayer(sink._buzz)
            pattern = PATTERNS[name]
            print(f"[{time.strftime('%H:%M:%S')}] in 3 s: pattern '{name}' "
                  f"({len(pattern.steps)} pulse(s), {pattern.duration_s:.2f} s)", flush=True)
            time.sleep(3.0)
            player.play(pattern, CATEGORY_FOR_PATTERN.get(name, "warning"))
            time.sleep(pattern.duration_s + 1.0)
            sink._buzzer = sink._buzzer._buzzer
            expected = expected_timeline(pattern.steps)
            observed = commanded_timeline(log)
            results["patterns"][name] = {"steps": [list(s) for s in pattern.steps],
                                         "expected_ms": expected, "commanded_ms": observed,
                                         "comparison": compare(expected, observed),
                                         "readiness": sink.readiness()}
            print(f"    commanded: {results['patterns'][name]['comparison']}", flush=True)
            time.sleep(args.gap)
    finally:
        try:
            sink._buzzer.off()
            sink._buzzer.close()
        except Exception as exc:  # noqa: BLE001 - releasing the pin must not mask the result
            print(f"warning: releasing GPIO {pin}: {exc}", file=sys.stderr)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    hc.write_json(Path(args.out), results)
    print(f"-> {args.out}\nNow annotate the phone recording and run `analyse`.")
    return 0


def analyse(timeline: dict, rows: list[dict]) -> dict:
    by_pattern: dict[str, list[tuple[int, float, float]]] = {}
    for r in rows:
        by_pattern.setdefault(r["pattern"], []).append(
            (int(r["step"]), float(r["onset_ms"]), float(r["offset_ms"])))
    out = {}
    for name, steps in by_pattern.items():
        steps.sort()
        t0 = steps[0][1]
        observed = [(on - t0, off - t0) for _, on, off in steps]
        expected = [tuple(x) for x in timeline["patterns"][name]["expected_ms"]]
        out[name] = compare(expected, observed)
    missing = sorted(set(timeline["patterns"]) - set(by_pattern))
    return {"audible_patterns": out, "not_heard": missing}


def cmd_analyse(args) -> int:
    timeline = json.loads(Path(args.timeline).read_text())
    with open(args.annotations, newline="") as fh:
        result = analyse(timeline, list(csv.DictReader(fh)))
    print(json.dumps(result, indent=2))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Buzzer pattern timing (Pi) and audio analysis")
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--patterns", default=None, help="comma list (default: all)")
    r.add_argument("--pin", type=int, default=None, help="default: alerts.gpio_buzzer_pin")
    r.add_argument("--gap", type=float, default=3.0)
    r.add_argument("--install-dir", default=str(hc.DEFAULT_INSTALL_DIR))
    r.add_argument("--code-root", default=str(hc.REPO_ROOT))
    r.add_argument("--no-service-check", action="store_true")
    r.set_defaults(func=cmd_run)
    a = sub.add_parser("analyse")
    a.add_argument("--timeline", required=True)
    a.add_argument("--annotations", required=True)
    a.set_defaults(func=cmd_analyse)
    args = parser.parse_args(argv)
    # Lazily imported code may come from the INSTALLED tree (--code-root): leave no
    # __pycache__ behind in it. Set here, not at import, so importing this file has
    # no process-wide side effect.
    sys.dont_write_bytecode = True
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(hc.run_cli(main))
