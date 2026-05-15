#!/usr/bin/env python3
"""Tiny Govee Developer-API CLI.

Reads the key from $GOVEE_API_KEY (already exported in ~/.zshrc).

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
from urllib.request import Request, urlopen

API = "https://openapi.api.govee.com/router/api/v1"
KEY = os.environ.get("GOVEE_API_KEY")
if not KEY:
    sys.exit("error: GOVEE_API_KEY not set")

# Devices to hide from `list` and from name lookups.
EXCLUDED_NAMES = {"permanent outdoor lights", "string lights"}


def _request(path: str, payload: dict | None = None) -> dict:
    url = f"{API}{path}"
    headers = {"Govee-API-Key": KEY, "Content-Type": "application/json"}
    data = json.dumps(payload).encode() if payload else None
    req = Request(url, data=data, headers=headers, method="POST" if data else "GET")
    with urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


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


def control(device: dict, capability_type: str, instance: str, value) -> dict:
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
