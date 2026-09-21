#!/usr/bin/env python3
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tiny Govee Developer-API CLI.

Reads the key from $GOVEE_API_KEY (already exported in ~/.zshrc).

In Muse-auth mode (the default) the key lives in the Muse
vault as a header-kind credential ($GOVEE_API_KEY is enrolled once, on
first use): this process only holds a surrogate, and the daemon swaps
it into the real ``Govee-API-Key`` header at the network boundary.
Device-state queries classify as reads; ``/device/control`` calls are
writes and follow the Sentinel write policy (grants).

Usage:
    ./govee.py list                       # show all devices
    ./govee.py state "Living room lamp"   # query current state
    ./govee.py on  "Living room lamp"
    ./govee.py off "Living room lamp"
    ./govee.py brightness "Living room lamp" 40   # 1..100
    ./govee.py color "Living room lamp" ff8800    # hex RGB
    ./govee.py kelvin "Living room lamp" 4000     # color temperature
"""

import json
import os
import sys
import uuid
from typing import Any
from urllib.request import Request, urlopen

API = "https://openapi.api.govee.com/router/api/v1"

EXCLUDED_NAMES = {"permanent outdoor lights", "string lights"}

# Muse-mode boundary session and surrogate, created once per process.
_MUSE_SESSION: tuple[Any, str] | None = None


def _api_key() -> str:
    """Return the Govee API key from $GOVEE_API_KEY, exiting if it is not set."""
    key = os.environ.get("GOVEE_API_KEY", "")
    if not key:
        sys.exit("error: GOVEE_API_KEY not set")
    return key


def _muse_session() -> tuple[Any, str]:
    """Return the Muse boundary session and surrogate, enrolling on first use.

    A ``govee`` credential already in the vault wins; otherwise
    ``$GOVEE_API_KEY`` is enrolled once as a header-kind credential
    (``Govee-API-Key``), after which the env var is no longer needed —
    it is dropped from this process's environment either way (remove
    the persistent shell export yourself; a child process cannot).

    Returns:
        ``(MuseBoundarySession, surrogate_token)``.
    """
    global _MUSE_SESSION
    if _MUSE_SESSION is None:
        from kiss.agents.third_party_agents.muse_auth.client import (
            MuseBoundarySession,
            bearer_surrogate,
        )

        surrogate = bearer_surrogate(
            "govee", os.environ.get("GOVEE_API_KEY", ""), header="Govee-API-Key"
        )
        if not surrogate:
            sys.exit("error: GOVEE_API_KEY not set and no 'govee' Muse vault credential")
        os.environ.pop("GOVEE_API_KEY", None)
        _MUSE_SESSION = (MuseBoundarySession("govee"), surrogate)
    return _MUSE_SESSION


# The only boundary failure that provably happens BEFORE any bytes
# reach the API host: the daemon rejecting a surrogate that died with a
# restarted/rotated vault (a restarted daemon is re-spawned and rejects
# the stale surrogate before forwarding).  A lost daemon socket reply
# is NOT pre-egress — authd may already have forwarded the write — so
# it is surfaced, never replayed.
_PRE_EGRESS_ERRORS = ("stale surrogate",)


def _muse_request(method: str, url: str, payload: dict | None) -> Any:
    """Execute one request at the Muse boundary, surviving a daemon restart.

    Surrogates die with the daemon, so a stale-surrogate rejection (or
    a dead daemon socket) resets the cached session and retries once
    with a freshly minted surrogate.  Only failures that occur before
    network egress are retried: an ambiguous failure after the request
    may have reached Govee is surfaced instead of replayed, so a
    device-control write can never fire twice.

    Args:
        method: HTTP method.
        url: Absolute request URL.
        payload: Optional JSON body.

    Returns:
        The boundary :class:`requests.Response`.
    """
    from kiss.agents.third_party_agents.muse_auth.client import MuseAuthError

    global _MUSE_SESSION
    for attempt in (0, 1):
        session, surrogate = _muse_session()
        # The surrogate travels as a bearer; the daemon swaps it into
        # the real Govee-API-Key header at the network boundary.
        headers = {"Authorization": f"Bearer {surrogate}", "Content-Type": "application/json"}
        try:
            return session.request(method, url, headers=headers, json=payload, timeout=60)
        except MuseAuthError as e:
            retriable = any(marker in str(e) for marker in _PRE_EGRESS_ERRORS)
            if attempt or not retriable:
                raise
            _MUSE_SESSION = None
    raise AssertionError("unreachable")  # pragma: no cover


def _request(path: str, payload: dict | None = None) -> dict:
    url = f"{API}{path}"
    from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

    if muse_auth_enabled():
        resp = _muse_request("POST" if payload else "GET", url, payload)
        if resp.status_code >= 400:
            sys.exit(f"error: HTTP {resp.status_code}: {resp.text[:500]}")
        return dict(resp.json())
    headers = {"Govee-API-Key": _api_key(), "Content-Type": "application/json"}
    data = json.dumps(payload).encode() if payload else None
    req = Request(url, data=data, headers=headers, method="POST" if data else "GET")
    with urlopen(req, timeout=60) as resp:
        result: dict[str, Any] = json.loads(resp.read())
        return result


def list_devices() -> list[dict]:
    """Return device list from /user/devices, excluding EXCLUDED_NAMES."""
    devs = _request("/user/devices")["data"]
    return [d for d in devs if d["deviceName"].lower() not in EXCLUDED_NAMES]


def find_device(name: str) -> dict:
    """Find a device by case-insensitive name substring or exact MAC."""
    devices = list_devices()
    for d in devices:
        if d["device"] == name or d["deviceName"].lower() == name.lower():
            return d
    matches = [d for d in devices if name.lower() in d["deviceName"].lower()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        sys.exit(f"error: no device matches {name!r}")
    sys.exit("error: ambiguous: " + ", ".join(m["deviceName"] for m in matches))


def control(
    device: dict[str, Any],
    capability_type: str,
    instance: str,
    value: Any,
) -> dict[str, Any]:
    """Send a /device/control command for one capability."""
    payload = {
        "requestId": str(uuid.uuid4()),
        "payload": {
            "sku": device["sku"],
            "device": device["device"],
            "capability": {"type": capability_type, "instance": instance, "value": value},
        },
    }
    return _request("/device/control", payload)


def state(device: dict) -> dict:
    """Query current state for a device."""
    payload = {
        "requestId": str(uuid.uuid4()),
        "payload": {"sku": device["sku"], "device": device["device"]},
    }
    return _request("/device/state", payload)


def cmd_list() -> None:
    """Print all devices."""
    devs = list_devices()
    print(f"{len(devs)} device(s):")
    for d in devs:
        print(f"  {d['sku']:8s}  {d['device']}  {d['deviceName']}")


def cmd_state(name: str) -> None:
    """Print state of one device."""
    d = find_device(name)
    print(json.dumps(state(d), indent=2))


def cmd_power(name: str, on: bool) -> None:
    """Power on/off a device."""
    d = find_device(name)
    r = control(d, "devices.capabilities.on_off", "powerSwitch", 1 if on else 0)
    print(json.dumps(r, indent=2))


def cmd_brightness(name: str, pct: int) -> None:
    """Set brightness 1..100."""
    d = find_device(name)
    r = control(d, "devices.capabilities.range", "brightness", int(pct))
    print(json.dumps(r, indent=2))


def cmd_color(name: str, hex_rgb: str) -> None:
    """Set color from hex RGB string like ff8800."""
    d = find_device(name)
    rgb = int(hex_rgb.lstrip("#"), 16)
    r = control(d, "devices.capabilities.color_setting", "colorRgb", rgb)
    print(json.dumps(r, indent=2))


def cmd_kelvin(name: str, k: int) -> None:
    """Set color temperature in kelvin."""
    d = find_device(name)
    r = control(d, "devices.capabilities.color_setting", "colorTemperatureK", int(k))
    print(json.dumps(r, indent=2))


def main(argv: list[str]) -> None:
    """CLI entry."""
    # A direct CLI run does not inherit the kiss-web daemon's
    # environment: import the canonical ``$KISS_HOME/api_keys.env``
    # (the Muse-auth ``KISS_MUSE_AUTH`` opt-out, API keys) before any
    # Muse-mode check or credential migration, exactly like
    # ``channel_main()``.
    from kiss.core.vscode_config import load_api_keys, load_api_keys_readonly

    # A read-only $KISS_HOME (the store's lock file cannot be
    # created) must neither stop the CLI nor drop a canonical
    # KISS_MUSE_AUTH=0 opt-out: fall back to the lock-free,
    # write-free import.
    try:
        load_api_keys()
    except OSError:
        load_api_keys_readonly()
    if len(argv) < 2:
        print(__doc__)
        return
    cmd = argv[1]
    args = argv[2:]
    if cmd == "list":
        cmd_list()
    elif cmd == "state":
        cmd_state(args[0])
    elif cmd == "on":
        cmd_power(args[0], True)
    elif cmd == "off":
        cmd_power(args[0], False)
    elif cmd == "brightness":
        cmd_brightness(args[0], int(args[1]))
    elif cmd == "color":
        cmd_color(args[0], args[1])
    elif cmd == "kelvin":
        cmd_kelvin(args[0], int(args[1]))
    else:
        sys.exit(f"unknown command: {cmd}")


if __name__ == "__main__":
    main(sys.argv)
