"""Central logging for SensorDyme services.

Every new subsystem (health monitor, supervisor, provisioning, accel) logs through
here so all output carries a timestamp, a level, and a component name — both on a
dev Mac's terminal and in journald on the Pi (`journalctl -u sensordyme`).

The older modules print() directly; they keep working unchanged. New code does:

    from core.logging_setup import get_logger
    log = get_logger("health")
    log.warning("camera stale for %.1fs", age)

Level comes from (highest precedence first): the SENSORDYME_LOG_LEVEL env var,
the `logging.level` key in config.yaml (applied by services/runtime.py), or INFO.
"""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def setup_logging(level: str | None = None) -> None:
    """Configure the root sensordyme logger once; later calls only adjust the level."""
    global _CONFIGURED  # noqa: PLW0603 - configure-once flag for the process
    resolved = (os.environ.get("SENSORDYME_LOG_LEVEL") or level or "INFO").upper()
    root = logging.getLogger("sensordyme")
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S%z",
            )
        )
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True
    root.setLevel(getattr(logging, resolved, logging.INFO))


def get_logger(component: str) -> logging.Logger:
    """A namespaced logger ("sensordyme.<component>"), configuring logging on first use."""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(f"sensordyme.{component}")


def describe_failure(exc: BaseException) -> str:
    """An outbound-request failure, fit for a log line: its type and HTTP status only.

    M8.3: the `repr()` of a `requests` exception carries the request URL, and for
    some providers the URL *is* the credential -- an ntfy topic is the whole
    secret of an account's alert channel (`https://ntfy.sh/<topic>`), a Twilio
    URL names the account SID. `%r` of such an exception put the secret into the
    backend's log on every failed delivery. Nothing here is read from the URL or
    the message; a status code and a class name are enough to act on.
    """
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    name = type(exc).__name__
    return f"{name} (HTTP {status})" if isinstance(status, int) else name


def mask_phone(number: object) -> str:
    """A phone number reduced to its last two digits ("***42"), for logs (M8.3)."""
    digits = "".join(ch for ch in str(number or "") if ch.isdigit())
    return f"***{digits[-2:]}" if len(digits) >= 2 else "***"
