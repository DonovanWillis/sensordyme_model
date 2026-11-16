# Release record: M11 hardware-activation candidate

**Status:** software-verified candidate for the **first physical install**.
- Not pushed, not merged.
- **No hardware gate has passed**: the Raspberry Pi 5 + Hailo-8L have not run it yet.

| Item | Value |
|---|---|
| Branch | `feat/m11-hardware-activation` (local) |
| Release commit (`$RC`) | `6df82b7a32a7cfc3a6f9848b0d94c9e2bc77942c` |
| Ancestry | contains M10.1 `5ed3e59`, M10 `629f906`, and the M8.5 release `4dfd761` / `f63ddd9`. `upstream/integration/all` (`ddd6e4c`, Donovan's merge of PR #3) has a tree identical to `4dfd761` |
| Transfer package | `~/sensordyme-release/transfer-6df82b7a32a7/` (59 files, 354.5 MB) |
| **sha256 of `SHA256SUMS`** (send this to the technician separately) | `f1852a80079c2c7f50c15a331ed6d8f69e7e6a8545c4d9a0c983da775cd4ad47` |
| Bundle | `bundle/sensordyme-6df82b7a32a7.tar.gz` sha256 `3abca6db8f3a78fcff87c92d2611213ac3f857329620143fdbe10ce31f77e3b9` |
| Bundle manifest | `bundle/sensordyme-6df82b7a32a7.manifest.json` sha256 `4b905f1c1e43aaab133c6dfc596ff5558785a33d7d1186be75061d5148c77086` |
| Production model | `models/damoyolo_tinynasL20_T.hef` sha256 `e69136d3ebf3cc8ebaf3da5417a59962b139f3d3d67d715e9a9b1ae10963960e` (18,554,430 bytes), decoder `legacy` |
| Wheels | 40 aarch64/cp311 wheels, 313.9 MB, hash-locked |

## Verification at `6df82b7`

| Gate | Result |
|---|---|
| ruff | clean |
| pyright | 0 errors |
| pytest (full) | **4,568 passed**, 6 skipped, 1 strict xfail (the documented escalation decision, OFF), 0 failed |
| Swift logic tests (`make test-ios`) | PASSED |
| iOS release-configuration audit | PASS |
| `make verify-release REF=HEAD` | **PASS** (the systemd step is SKIP on macOS) |
| `make acceptance-full` | **PASS: 896 / 896**, including all 30 owner journeys (simulated) and **28 / 28 XCUITest journeys in the full run**, with no audit-instrument retries; the Simulator was ready in 3 s |
| Transfer package | `TRANSFER PACKAGE OK`; independently re-verified `shasum -a 256 -c SHA256SUMS`: 59 / 59 OK |

Evidence: `~/sensordyme-audit/2026-11-10/m11-evidence/` (`acceptance-6df82b7/`, `product_thresholds-0c1222/`, `audits-092345e/`).

## How to hand it over

1. Give the technician the package directory (USB drive or a private share).
2. In chat, send **only** `$RC` and the `SHA256SUMS` hash above.
3. The technician follows `docs/FRIEND_HARDWARE_SETUP.md` step 9, plus the package's own `INSTALL.md`, which pins every command for this release. The README inside the bundle still names the M8.5 release in its Phase 3 example; the guide and `INSTALL.md` take precedence.

## What this release does not claim

- **INT8 parity, Pi latency, thermal and the processed frame rate:** not measured; hardware gates H1–H3.
- **Audible buzzer:** not checked; a person must hear it.
- **Real-world accuracy:** none, until independently labelled in-cabin sessions pass the power check (`docs/REAL_DATA_EVALUATION_PROTOCOL.md`).
- **Experimental items:** EXP-08, EXP-09, the dual detector and relapse escalation all stay experimental or off.
