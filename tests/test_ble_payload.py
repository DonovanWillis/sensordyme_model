"""Getting the network list across the wire intact.

A value larger than one MTU is transferred by the central issuing Read Blob
requests for successive chunks. If the server ignores the offset, every chunk
returns the start of the value and the transfer truncates — the app receives
invalid JSON and shows NO networks, which is indistinguishable from a device
that cannot see any. It fails precisely where it matters: a flat with a dozen
networks in range.
"""

import json

from services.ble_gatt import _GATT_MAX_ATTRIBUTE_BYTES, _fit_networks


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
    assert len(payload) <= _GATT_MAX_ATTRIBUTE_BYTES


def test_the_payload_never_exceeds_what_gatt_can_carry():
    """512 octets is the ceiling in the GATT specification, not a tuning knob.
    BlueZ truncates there with no error, so anything larger reaches the app as
    JSON cut mid-string — which it cannot parse, so it shows NO networks. That
    reads as "this device can't see any Wi-Fi" while the device sees twelve.

    Observed on hardware: a read returned exactly 512 bytes ending mid-SSID.
    """
    for count in (1, 5, 12, 40, 200):
        payload = _fit_networks(networks(count))
        assert len(payload) <= _GATT_MAX_ATTRIBUTE_BYTES, f"{count} networks -> {len(payload)} bytes"
        json.loads(payload)  # raises if truncated


def test_a_realistic_neighbourhood_still_yields_a_usable_list():
    """The real scan from the test flat: twelve networks, ordinary SSID lengths.
    Trimming to fit must not leave so few that the list is useless."""
    real = [
        {"ssid": "MyAltice 54be1b", "signal": 94, "secured": True},
        {"ssid": "ATTBcBKJJA", "signal": 79, "secured": True},
        {"ssid": "Brightspeed_0112F425", "signal": 57, "secured": True},
        {"ssid": "LEWIS", "signal": 57, "secured": True},
        {"ssid": "Brightspeed_01879288", "signal": 55, "secured": True},
        {"ssid": "Samsung Fridge_E50AJTRE004E", "signal": 54, "secured": True},
        {"ssid": "Brightspeed_010CBF10", "signal": 49, "secured": True},
        {"ssid": "MyOptimum d9127f", "signal": 39, "secured": True},
        {"ssid": "MyAltice 35a84d", "signal": 22, "secured": True},
    ]
    parsed = json.loads(_fit_networks(real))
    assert len(parsed) >= 5, f"only {len(parsed)} networks survived the cap"
    assert parsed[0]["ssid"] == "MyAltice 54be1b", "the strongest network must survive"


def test_a_short_list_is_untouched():
    original = networks(3)
    assert json.loads(_fit_networks(original)) == original


def test_an_empty_list_is_an_empty_array():
    assert _fit_networks([]) == b"[]"


def test_one_absurdly_long_ssid_does_not_produce_broken_json():
    payload = _fit_networks([{"ssid": "Y" * 9000, "signal": 50, "secured": True}])
    assert json.loads(payload) == []
