"""Getting the network list across the wire intact.

A value larger than one MTU is transferred by the central issuing Read Blob
requests for successive chunks. If the server ignores the offset, every chunk
returns the start of the value and the transfer truncates — the app receives
invalid JSON and shows NO networks, which is indistinguishable from a device
that cannot see any. It fails precisely where it matters: a flat with a dozen
networks in range.
"""

import json

from services.ble_gatt import _fit_networks


def networks(count, prefix="Brightspeed_ABCDEF"):
    return [{"ssid": f"{prefix}{i:04d}", "signal": 90 - i, "secured": True}
            for i in range(count)]


def test_a_long_list_is_still_valid_json():
    payload = _fit_networks(networks(60))
    parsed = json.loads(payload)  # would raise if truncated mid-object
    assert isinstance(parsed, list)


def test_the_strongest_networks_are_the_ones_kept():
    """Sorted strongest-first upstream, so trimming the tail drops the networks
    least likely to be the one someone is standing next to."""
    parsed = json.loads(_fit_networks(networks(60)))
    assert parsed[0]["signal"] == 90


def test_trimming_is_by_bytes_not_by_count():
    """The previous cap took the first twelve and assumed that fit. Twelve real
    SSIDs are roughly 700 bytes, so it never engaged."""
    long_names = [{"ssid": "X" * 60 + str(i), "signal": 50, "secured": True}
                  for i in range(40)]
    payload = _fit_networks(long_names)
    json.loads(payload)
    assert len(payload) <= 4096


def test_a_short_list_is_untouched():
    original = networks(3)
    assert json.loads(_fit_networks(original)) == original


def test_an_empty_list_is_an_empty_array():
    assert _fit_networks([]) == b"[]"


def test_one_absurdly_long_ssid_does_not_produce_broken_json():
    payload = _fit_networks([{"ssid": "Y" * 9000, "signal": 50, "secured": True}])
    assert json.loads(payload) == []
