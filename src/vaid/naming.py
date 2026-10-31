"""Device-key normalisation.

data/samples/README.md trap 1: the flow file is
``WithingsSmartScale_0024e41b6f96_flows.csv`` (one underscore) while the
parameter directory is ``WithingsSmartScale__0024e41b6f96`` (two). A naive join
on the directory name silently drops that device's entire handshake view, which
would look like a genuine "this device has no TLS" observation.

`device_key` collapses both spellings onto the same key and `class_label` strips
the MAC suffix, because the MAC is a forbidden identifier (§4.3) and must not
survive into the label either.
"""
from __future__ import annotations

import re

_MAC_SUFFIX = re.compile(r"_+([0-9A-Fa-f]{12})$")


def device_key(name: str) -> str:
    """Canonical key for a device directory or flow-file stem."""
    name = name.replace("_flows.csv", "").replace("_flows", "")
    m = _MAC_SUFFIX.search(name)
    if m:
        return f"{name[: m.start()].rstrip('_')}_{m.group(1).lower()}"
    return name.rstrip("_")


def class_label(name: str) -> str:
    """Human-readable device-type label with the MAC removed."""
    key = device_key(name)
    m = _MAC_SUFFIX.search(key)
    return key[: m.start()].rstrip("_") if m else key


def device_mac(name: str) -> str | None:
    """The MAC embedded in the file name. Used ONLY by the audit, to search for
    the device's own OUI leaking into feature columns. Never a feature."""
    m = _MAC_SUFFIX.search(device_key(name))
    return m.group(1).lower() if m else None
