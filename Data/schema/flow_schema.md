# Flow CSV schema — UNSW-IoTraffic

Derived from the AmazonEcho sample row. 29 columns, one row per bidirectional
5-tuple flow. Files are per-device (27 devices).

## Units — resolved

The sample is internally consistent and fixes the units:

```
flowDuration / srcNumPackets = 3.11e8 / 11517 = 27,004
srcAvgInterarrivalTime       = 27,012
```

Therefore **`flowDuration` and all interarrival columns are microseconds**.
The sampled flow ran 311 seconds. Payload-size columns are bytes and exclude
headers.

## Columns

| Column | Type | Role |
|---|---|---|
| `time` | timestamp | Flow start. **Join/ordering key for E1, E2, Exp 5.** |
| `srcMac`, `dstMac` | hex | FORBIDDEN. Null in the sample, but see `chaddr` in protocol files. |
| `ethType` | hex | Constant 0x0800 in sample; drop if zero-variance. |
| `srcIp` | ip | FORBIDDEN (local IP = label proxy). |
| `dstIp` | ip | Abstracted only (view C). Never raw. |
| `ipProto` | int | Feature (view F). |
| `srcPort`, `dstPort` | int | Feature via service bucketing (view C). |
| `flowSeqNum` | int | **JOIN KEY to protocol-parameter files.** Must be preserved. |
| `srcNumPackets`, `dstNumPackets` | int | View F |
| `srcPayloadSize`, `dstPayloadSize` | bytes | View F |
| `srcAvgPayloadSize`, `dstAvgPayloadSize` | bytes | View F |
| `srcMaxPayloadSize`, `dstMaxPayloadSize` | bytes | View F |
| `srcStdDevPayloadSize`, `dstStdDevPayloadSize` | bytes | View F |
| `flowDuration` | µs | View F |
| `srcAvgInterarrivalTime`, `dstAvgInterarrivalTime`, `avgInterarrivalTime` | µs | View F |
| `srcStdDevInterarrivalTime`, `dstStdDevInterarrivalTime`, `stdDevInterarrivalTime` | µs | View F |
| `allMatchedProtocols` | pipe-delimited multi-label | **Audit key** — gives TLS/QUIC/DTLS prevalence for §4.1 |
| `protocol` | string | Primary protocol label |

## Consequences for the proposal

**No per-packet sequences and no TCP flags exist at flow level.** §4.5 lists
"TCP flag statistics where available" — they are not available here. The temporal
encoder (§5.2) must therefore operate over a **sequence of flow events** within
window T, not a packet sequence. This must be stated explicitly in §4.5 rather
than left implied.

Feature count for view F: 17 numeric aggregates + `ipProto` + port-derived
service bucket ≈ 20 features.

## Join to the handshake view

```
(srcIp, dstIp, ipProto, srcPort, dstPort, flowSeqNum)
```

Both the flow CSVs and the protocol-parameter CSVs carry all six fields. This is
what makes the multi-view design implementable — **without `flowSeqNum` the
handshake view cannot be aligned to flows at all.**

## Known data-quality flag

The sample row has `ipProto`=17 (UDP), `dstPort`=33434 (traceroute range),
11,517 outbound packets against 1 inbound — yet `protocol`=`icmpv6`. Count the
rate of label/protocol disagreement during the §4.1 audit; if widespread it
affects how far `allMatchedProtocols` can be trusted as the handshake-availability
signal.

## Ingest warning

Every sample received so far was opened in Excel, which rendered `time` as
`30:02.8` (MM:SS.S) and `flowDuration` as `3.11E+08`. **Timestamps are
unrecoverable from those files.** Raw exports are required.
