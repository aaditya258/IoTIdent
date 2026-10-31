"""Reading UNSW-IoTraffic flow and protocol-parameter CSVs.

Design rules that come from the data, not from taste:

* Timestamps are parsed with an explicit format and anything that fails becomes
  NaT rather than a guess. The eight epoch-artifact rows described in
  data/samples/README.md (1970/1981/1990/2010-2012) are counted, not silently
  dropped -- §4.1 asks for that count.
* `flowDuration` and every interarrival column are microseconds
  (data/schema/flow_schema.md). Conversion happens once, here.
* Parameter rows are joined to flows on the six-field key including
  `flowSeqNum`; the join rate is measured and reported, because without that key
  the handshake view cannot be aligned to flows at all.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import FLOW_NUMERIC, JOIN_KEY, Paths
from .naming import class_label, device_key

TIME_FMT = "%Y-%m-%d %H:%M:%S.%f"
#: Rows outside this range are capture artifacts, not observations.
PLAUSIBLE_START = pd.Timestamp("2016-01-01")
PLAUSIBLE_END = pd.Timestamp("2018-12-31")


def to_epoch_seconds(t: pd.Series) -> np.ndarray:
    """Epoch seconds from a datetime column.

    pandas 3 infers the resolution from the data, so a column parsed from
    microsecond-precision timestamps comes back as datetime64[us] and a naive
    ``astype("int64") / 1e9`` is off by a factor of a thousand -- which silently
    collapses every observation window into a handful. Convert explicitly.
    """
    return (t.astype("datetime64[ns]").astype("int64").to_numpy() / 1e9)


def _parse_time(s: pd.Series) -> pd.Series:
    t = pd.to_datetime(s, format=TIME_FMT, errors="coerce")
    if t.isna().any():  # a minority of rows carry whole seconds
        alt = pd.to_datetime(s[t.isna()], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        t = t.copy()
        t[t.isna()] = alt
    return t


@dataclass
class FlowTable:
    device: str          # canonical device key
    label: str           # class label (MAC stripped)
    df: pd.DataFrame
    n_raw_rows: int
    n_unparsed_time: int
    n_epoch_artifact: int


def read_flows(path: str) -> FlowTable:
    dev = device_key(os.path.basename(path))
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""],
                     engine="c", on_bad_lines="warn")
    n_raw = len(df)
    df["time"] = _parse_time(df["time"])
    n_unparsed = int(df["time"].isna().sum())
    df = df[df["time"].notna()]
    bad = (df["time"] < PLAUSIBLE_START) | (df["time"] > PLAUSIBLE_END)
    n_artifact = int(bad.sum())
    df = df[~bad]

    for c in FLOW_NUMERIC + ["ipProto", "srcPort", "dstPort", "flowSeqNum"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # microseconds -> seconds, once and only once
    for c in ("flowDuration", "srcAvgInterarrivalTime", "dstAvgInterarrivalTime",
              "avgInterarrivalTime", "srcStdDevInterarrivalTime",
              "dstStdDevInterarrivalTime", "stdDevInterarrivalTime"):
        if c in df.columns:
            df[c] = df[c] / 1e6

    df = df.sort_values("time", kind="stable").reset_index(drop=True)
    return FlowTable(device=dev, label=class_label(dev), df=df, n_raw_rows=n_raw,
                     n_unparsed_time=n_unparsed, n_epoch_artifact=n_artifact)


def list_devices(paths: Paths) -> list[str]:
    files = sorted(glob.glob(os.path.join(paths.flows, "*_flows.csv")))
    if not files:
        raise FileNotFoundError(
            f"no flow files under {paths.flows} -- point --root at the repository "
            "that contains data/samples/flows")
    return [device_key(os.path.basename(f)) for f in files]


def flow_path(paths: Paths, device: str) -> str:
    hits = glob.glob(os.path.join(paths.flows, "*_flows.csv"))
    for h in hits:
        if device_key(os.path.basename(h)) == device:
            return h
    raise FileNotFoundError(device)


def param_dir(paths: Paths, device: str) -> str | None:
    """Locate a device's parameter directory, tolerating the double-underscore
    spelling of WithingsSmartScale."""
    for d in sorted(glob.glob(os.path.join(paths.parameters, "*"))):
        if os.path.isdir(d) and device_key(os.path.basename(d)) == device:
            return d
    return None


def read_parameter(paths: Paths, device: str, side: str, proto: str) -> pd.DataFrame:
    """Read one protocol-parameter CSV. Returns an empty frame when absent --
    absence is an observation (the view is unavailable), not an error."""
    d = param_dir(paths, device)
    if d is None:
        return pd.DataFrame()
    p = os.path.join(d, side, f"{proto}attributes.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    try:
        df = pd.read_csv(p, dtype=str, keep_default_na=False, na_values=[""],
                         engine="python", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()
    if df.empty or "time" not in df.columns:
        return pd.DataFrame()
    df["time"] = _parse_time(df["time"])
    df = df[df["time"].notna()]
    for c in ("ipProto", "srcPort", "dstPort", "flowSeqNum"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("time", kind="stable").reset_index(drop=True)


def join_rate(flows: pd.DataFrame, params: pd.DataFrame) -> float:
    """Fraction of parameter rows that resolve into a flow row on the six-field
    key. Reported by the audit; a low rate invalidates the handshake view."""
    if params.empty or flows.empty:
        return float("nan")
    key = [k for k in JOIN_KEY if k in flows.columns and k in params.columns]
    if len(key) < len(JOIN_KEY):
        return float("nan")
    def canon(v):
        # pandas may hold the same port as 443 in one frame and 443.0 in the
        # other, depending on whether that column ever saw a NaN. Normalise.
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return ""
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, float):
            return str(int(v)) if float(v).is_integer() else str(v)
        return str(v)

    def keys(df: pd.DataFrame) -> np.ndarray:
        cols = [df[c].map(canon).to_numpy() for c in key]
        out = cols[0]
        for c in cols[1:]:
            out = np.char.add(np.char.add(out.astype(str), "|"), c.astype(str))
        return out

    left = keys(params)
    right = set(keys(flows).tolist())
    return float(np.mean([k in right for k in left])) if len(left) else float("nan")
