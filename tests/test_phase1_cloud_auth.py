"""Two ways the account API answered a bad request with a broken server.

`/v1/account/ntfy` stored whatever it was given after a .strip(), and
`send_push` interpolates that value straight into `f"{NTFY_BASE}/{topic}"`.
The topic IS the credential — ntfy has no accounts, so the string is the only
thing standing between an account's safety alerts and anyone who knows it —
and it also chose the entire request path.

`current_user` did `int(payload["uid"])` unguarded on the hot auth path. A
validly-signed token whose claims were not the shape this file expects raised
KeyError or ValueError out of a FastAPI dependency, which becomes HTTP 500: the
server reporting itself broken for what is only a token it will not accept.
And PyJWT enforces an expiry only when one is PRESENT, so a token minted with
no `exp` was accepted for ever — a 30-day TTL that silently did not apply.
"""

import importlib
import sys
import time

import jwt
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient


@pytest.fixture()
def backend(tmp_path, monkeypatch):
    monkeypatch.setenv("SENSORDYME_AI_PROVIDER", "unavailable")
    monkeypatch.setenv("SENSORDYME_DB", str(tmp_path / "test.db"))
    for mod in ("cloud.app", "cloud", "cloud.ratelimit"):
        sys.modules.pop(mod, None)
    app_module = importlib.import_module("cloud.app")
    client = TestClient(app_module.app)
    token = client.post(
        "/v1/signup", json={"email": "auth@example.com", "password": "password1"}
    ).json()["token"]
    return app_module, client, {"Authorization": f"Bearer {token}"}


# ------------------------------------------------------------------ ntfy topic
BAD_TOPICS = [
    "../../admin",                     # climbs out of the topic path entirely
    "topic/../other",                  # ...and so does this, less obviously
    "topic?priority=max&tags=skull",   # appends query parameters to our request
    "topic#frag",
    "topic with spaces",
    "topic\nX-Injected: yes",          # a newline in a URL built by hand
    "a" * 65,                          # ntfy's own length limit
    "tópico",                          # non-ASCII: not a topic ntfy accepts
]


@pytest.mark.parametrize("topic", BAD_TOPICS)
def test_a_topic_that_is_not_a_topic_is_refused(backend, topic):
    _, client, auth = backend
    resp = client.post("/v1/account/ntfy", json={"ntfy_topic": topic}, headers=auth)
    assert resp.status_code == 400, f"{topic!r} was accepted as an ntfy topic"


@pytest.mark.parametrize("topic", BAD_TOPICS)
def test_a_refused_topic_is_not_stored_either(backend, topic):
    """Rejecting the request is only half of it — the row must be unchanged,
    or the next push goes to whatever was written before the check."""
    app_module, client, auth = backend
    client.post("/v1/account/ntfy", json={"ntfy_topic": "sensordyme-real"}, headers=auth)
    client.post("/v1/account/ntfy", json={"ntfy_topic": topic}, headers=auth)
    with app_module.db() as conn:
        stored = conn.execute("SELECT ntfy_topic FROM users WHERE id = 1").fetchone()["ntfy_topic"]
    assert stored == "sensordyme-real"


@pytest.mark.parametrize("topic", ["sensordyme-alerts", "a", "A_b-9", "x" * 64])
def test_a_real_topic_is_still_accepted(backend, topic):
    """The validation must not break the feature it protects."""
    app_module, client, auth = backend
    assert client.post(
        "/v1/account/ntfy", json={"ntfy_topic": topic}, headers=auth
    ).status_code == 200
    with app_module.db() as conn:
        assert conn.execute(
            "SELECT ntfy_topic FROM users WHERE id = 1"
        ).fetchone()["ntfy_topic"] == topic


def test_clearing_the_topic_still_turns_push_off(backend):
    """Empty is how the app disables push; it must not be caught by the check."""
    app_module, client, auth = backend
    client.post("/v1/account/ntfy", json={"ntfy_topic": "sensordyme-alerts"}, headers=auth)
    assert client.post("/v1/account/ntfy", json={"ntfy_topic": "  "}, headers=auth).status_code == 200
    with app_module.db() as conn:
        assert conn.execute(
            "SELECT ntfy_topic FROM users WHERE id = 1"
        ).fetchone()["ntfy_topic"] == ""


# ------------------------------------------------------------------ auth claims
def _signed(app_module, payload):
    return {"Authorization": "Bearer " + jwt.encode(payload, app_module.JWT_SECRET, algorithm="HS256")}


@pytest.mark.parametrize("uid", ["not-an-int", None, [1], {"id": 1}, True, 1.5])
def test_a_token_with_a_uid_that_is_not_a_user_id_is_401_not_500(backend, uid):
    """A token this server will not accept is 401. 500 says the server broke,
    which sends the app's Connection doctor hunting for an outage."""
    app_module, client, _ = backend
    headers = _signed(app_module, {"uid": uid, "exp": int(time.time()) + 60, "jti": "j1"})
    assert client.get("/v1/devices", headers=headers).status_code == 401


def test_a_boolean_uid_does_not_authenticate_as_user_one(backend):
    """`bool` is a subclass of `int`, so a plain isinstance check would let
    `True` through as user 1 — the first account on the backend."""
    app_module, client, _ = backend
    headers = _signed(app_module, {"uid": True, "exp": int(time.time()) + 60, "jti": "j2"})
    assert client.get("/v1/devices", headers=headers).status_code == 401


def test_a_token_with_no_uid_is_401_not_500(backend):
    app_module, client, _ = backend
    headers = _signed(app_module, {"exp": int(time.time()) + 60, "jti": "j3"})
    assert client.get("/v1/devices", headers=headers).status_code == 401


def test_a_token_with_no_expiry_is_rejected(backend):
    """PyJWT enforces an `exp` that is present and nothing at all when it is
    absent, so this token was valid for ever."""
    app_module, client, _ = backend
    headers = _signed(app_module, {"uid": 1, "jti": "j4"})
    assert client.get("/v1/devices", headers=headers).status_code == 401


def test_a_token_with_no_jti_is_rejected(backend):
    """A token with no jti cannot be revoked, so signing out is cosmetic for
    it. Deliberate behaviour change: tokens predating revocation are refused
    and their owners sign in once more."""
    app_module, client, _ = backend
    headers = _signed(app_module, {"uid": 1, "exp": int(time.time()) + 60})
    assert client.get("/v1/devices", headers=headers).status_code == 401


def test_an_expired_token_is_still_rejected(backend):
    app_module, client, _ = backend
    headers = _signed(app_module, {"uid": 1, "exp": int(time.time()) - 1, "jti": "j5"})
    assert client.get("/v1/devices", headers=headers).status_code == 401


def test_a_real_token_still_works_and_logout_still_revokes_it(backend):
    """The whole point: none of the above may cost a genuine session."""
    _, client, auth = backend
    assert client.get("/v1/devices", headers=auth).status_code == 200
    assert client.post("/v1/logout", headers=auth).status_code == 200
    assert client.get("/v1/devices", headers=auth).status_code == 401


# ------------------------------------------------------------- session refresh
def test_refresh_issues_a_new_token_without_costing_the_old_one(backend):
    """The 30-day expiry is fixed at issue and nothing renewed it, so an app
    opened on day 31 landed on the sign-in screen however healthy the setup
    (M8.5 reconnect audit). A valid session may trade itself for a fresh one;
    signing out stays /v1/logout's job, so the old token keeps its own expiry."""
    _, client, auth = backend
    reply = client.post("/v1/session/refresh", headers=auth)
    assert reply.status_code == 200
    fresh = reply.json()["token"]
    old = dict(auth)["Authorization"].split(" ", 1)[1]
    assert fresh != old
    old_claims = jwt.decode(old, options={"verify_signature": False})
    new_claims = jwt.decode(fresh, options={"verify_signature": False})
    assert new_claims["exp"] >= old_claims["exp"]  # same-second issue gives equality
    assert new_claims["jti"] != old_claims["jti"]
    assert new_claims["uid"] == old_claims["uid"]
    assert new_claims["email"] == old_claims["email"]
    # both sessions work: renewal is additive, not a rotation
    assert client.get("/v1/devices", headers=auth).status_code == 200
    assert client.get("/v1/devices", headers={"Authorization": f"Bearer {fresh}"}).status_code == 200


def test_refresh_refuses_what_every_authenticated_route_refuses(backend):
    """Refresh must never resurrect a session: no token, a signed-out token
    and a deleted account's token are all 401, exactly like /v1/devices."""
    _, client, auth = backend
    assert client.post("/v1/session/refresh").status_code == 401
    assert client.post(
        "/v1/session/refresh", headers={"Authorization": "Bearer not-a-token"}
    ).status_code == 401
    deleted = client.post(
        "/v1/account/delete", json={"password": "password1", "confirm": True}, headers=auth
    )
    assert deleted.status_code == 200, deleted.text
    assert client.post("/v1/session/refresh", headers=auth).status_code == 401


def test_a_signed_out_token_cannot_refresh_itself(backend):
    _, client, auth = backend
    assert client.post("/v1/logout", headers=auth).status_code == 200
    assert client.post("/v1/session/refresh", headers=auth).status_code == 401
