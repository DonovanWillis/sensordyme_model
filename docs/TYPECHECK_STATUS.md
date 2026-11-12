# Type-check status

Pyright, pinned so runs compare: from the repo root,

    PYRIGHT_PYTHON_FORCE_VERSION=1.1.411 pyright --pythonpath <venv>/bin/python

Configuration: `pyrightconfig.json` (`typeCheckingMode: basic`, the rule levels
explained there; `cloud/` promotes the Optional rules to errors). There is no
repo `.venv`, so `--pythonpath` is always passed. Measured 2026-11-12 (M8.1).

| | errors | warnings |
|---|---:|---:|
| before (base 618880c) | 42 | 978 |
| after | **0** | 981 |

No rule was disabled or softened, `typeCheckingMode` is unchanged, and no
file-level `# pyright:` comment was added. No `# pyright: ignore` was needed;
two existing `# type: ignore[...]` comments were removed (in
`core/model_registry.py` and `scripts/sim/uitest_control.py`) by fixing what
they hid.

The +3 warnings are `reportMissingModuleSource` for the new `typings/onnx`
stub (one per import site: "stub found, package not installed"). That is the
same by-design warning every Pi-only stub in `typings/` produces and the reason
the config keeps that rule at warning. None of the warnings that existed
before were on a line this change touches; the 978 are otherwise unchanged.

## The 42 errors

Categories: **bug** (genuine, can misbehave at runtime) · **annotation**
(the code is right, its types or narrowing were wrong or missing) ·
**resolution** (an import that works at runtime but pyright could not
resolve: unmodelled `sys.path`, or an optional package with no stub).
No error was platform-specific (the Pi-only modules were already stubbed) and
none was a third-party typing defect.

| # | file:line (base) | rule | category | resolution |
|---|---|---|---|---|
| 1-2 | cloud/app.py:1183, 1184 | reportOptionalMemberAccess, reportOptionalSubscript | annotation | `x.get(k) if isinstance(x.get(k), dict) else {}` checks one call and returns another, so the result stayed Optional. New `_dict_field(source, key)` reads once and narrows; same object, same `{}` fallback. Used at the 5 sites that build these dicts. |
| 3-12 | cloud/app.py:1262, 1271, 1275, 1278, 1280, 1296, 1307, 1318, 1327, 1328 | reportOptionalMemberAccess | annotation | as above (`live_monitoring`, `reported_protection` in `_device_status`) |
| 13-14 | cloud/app.py:2436 (x2) | reportOptionalMemberAccess | annotation | as above (`health_monitoring`, `protection` in `_reported_role`) |
| 15-18 | core/model_registry.py:101, 105, 109, 113 | reportReturnType | annotation | `EvaluationStatus.__lt__/__le__/__gt__/__ge__` returned `int \| bool` because `_cmp` passed NotImplemented through an `-> int` helper (behind a `type: ignore`). Each operator now returns NotImplemented itself; results unchanged. |
| 19 | scripts/hw_validation/compare_float_vs_int8.py:170 | reportMissingImports | resolution | `onnx` is a worker-only optional dependency (in no requirements file, imported lazily). Minimal stub `typings/onnx/` covering only the names used. |
| 20 | scripts/release/make_release_manifest.py:56 | reportMissingImports | resolution | imports `integrity_manifest` after `sys.path.insert(scripts/hw_validation)`. Added `scripts/hw_validation` to `extraPaths` (as `scripts/` already is). |
| 21 | scripts/sim/uitest_control.py:157 | reportReturnType | **bug** | `Stack.token()` indexed the login reply blindly: see Bugs below. |
| 22-23 | scripts/start_limit_guard.py:60, 62 | reportReturnType | annotation | the comprehension inferred `list[float \| int]`, which list invariance refuses for `list[float]`. Annotated `recent: list[float]`. |
| 24-26 | services/runtime.py:233, 234, 235 | reportAssignmentType | annotation | the `MappingProxyType({})` class defaults (for runtimes built without `__init__`) clash with the `dict[...]` declared in `__init__`. Kept read-only and `cast` to the instance type, with the reason in a comment beside it: an annotation cannot say "read-only on the class, a dict on the instance". |
| 27 | tests/test_ble_pop.py:301 | reportPossiblyUnboundVariable | annotation | `reply` bound only inside a `for` loop; restructured (two lockouts, then assert the third), same handshakes. |
| 28-32 | tests/test_detector_bench.py:40, 51, 52, 62, 63 | reportAssignmentType (tuple unpack) | annotation | `research/detector_bench/models.decode_dfl` returns 3 or 4 arrays by `with_argmax`; `Literal` overloads now say which. |
| 33-39 | tests/test_m5_hw_validation.py:32-38 | reportMissingImports | resolution | same `sys.path.insert(scripts/hw_validation)` as #20; fixed by the same `extraPaths` entry. |
| 40 | tests/test_m5_hw_validation.py:594 | reportMissingImports | resolution | `onnx` (after `pytest.importorskip`); fixed by the stub of #19. Resolving it surfaced onnxruntime's `InferenceSession.run` union (dense/sparse/list/dict) as 4 warnings in that test; it now asserts the outputs are ndarrays. |
| 41 | tests/test_m6_preset_settings_device.py:102 | reportGeneralTypeIssues ("None" is not iterable) | annotation | unpacked `restored_preset()`'s `Optional[tuple]`; now asserts it is not None first. |
| 42 | tests/test_sim_reliability.py:472 | reportPossiblyUnboundVariable | annotation | `exited` read after the loop that binds it; the last exit code is kept in a variable bound before the loop. |

Totals: bug 1 · annotation 31 · resolution 10 (8 unmodelled `sys.path`,
2 missing optional-package stub) · platform-specific 0 · third-party 0.

## Bugs found

- **scripts/sim/uitest_control.py:153-157 `Stack.token()`** (UI-test
  controller). A refused demo login (`{"detail": ...}`) raised
  `KeyError('token')`; a non-object reply raised `TypeError`; a `null` token was
  cached as `None` and sent as `Authorization: Bearer None`, so every later
  controller call failed with an unexplained 401. It now requires a non-empty
  string and otherwise raises `RuntimeError("demo login failed: HTTP <status>,
  <reply>")`, caching nothing. Test: `tests/test_uitest_control.py` (5 of its 6
  cases fail on the old code).

## Remaining errors

None.

## Notes

- `typings/onnx/` shadows the real package's inline types if onnx is ever
  installed. If it is added to `requirements-dev.txt`, delete the stub.
- Outside the configured `include` (so not counted above): `research/` and
  `sim/`. Checked directly, `research/detector_bench/models.py` has 2 errors of
  the same shape as #28-32 at its `_class_nms(..., return_index=True)` call
  sites (lines ~130 and ~290); `Literal` overloads on `_class_nms` would clear
  them. `sim/device.py`, mentioned in earlier audits, is not in `include`.
- Earlier audits listed a `cv2` import error in services/runtime.py and
  unresolved `cv2`/`jwt`; with the pinned version and `--pythonpath` pointing at
  the dev venv those resolve, and they are not in the base run.
- Evidence: `~/sensordyme-audit/2026-11-10/m81-evidence/pyright/before.txt`
  and `after.txt`.
