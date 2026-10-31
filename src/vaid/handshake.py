"""TLS ClientHello and DHCP parsing, implementing `allowlist_handshake` from
data/schema/forbidden_fields.yaml.

The critical rule, and the reason this module exists rather than the raw column
being fed to a model: TLS extension 0x0000 is server_name, so the raw SNI sits
inside the allowlisted `client-hello-extensions` blob. The blob is parsed into
(type, length) pairs; the ORDERED TYPE LIST and the LENGTHS survive, extension
VALUES are discarded except for the four types the spec permits, and the SNI
value never leaves this module.

`parse_client_hello_extensions` returns the SNI length but not one byte of the
name itself, so a downstream feature can say "a 28-byte server name was
requested" without saying which one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Extension types, per data/schema/forbidden_fields.yaml
EXT_SERVER_NAME = 0x0000
EXT_SUPPORTED_GROUPS = 0x000A
EXT_EC_POINT_FORMATS = 0x000B
EXT_SIGNATURE_ALGORITHMS = 0x000D
EXT_ALPN = 0x0010
EXT_SESSION_TICKET = 0x0023
EXT_KEY_SHARE = 0x0033
EXT_SUPPORTED_VERSIONS = 0x002B
EXT_PRE_SHARED_KEY = 0x0029
EXT_ECH = 0xFE0D

VALUE_USE_PERMITTED = {EXT_SUPPORTED_GROUPS, EXT_EC_POINT_FORMATS,
                       EXT_SIGNATURE_ALGORITHMS, EXT_KEY_SHARE}
PRESENCE_ONLY = {EXT_ALPN, EXT_SESSION_TICKET}
VALUE_FORBIDDEN = {EXT_SERVER_NAME}

TLS_VERSIONS = ["0300", "0301", "0302", "0303", "0304"]


@dataclass
class ClientHello:
    version: str = ""
    cipher_suites: list[int] = field(default_factory=list)
    ext_types: list[int] = field(default_factory=list)     # ordered, per spec
    ext_lengths: list[int] = field(default_factory=list)
    supported_groups: list[int] = field(default_factory=list)
    supported_versions: list[int] = field(default_factory=list)
    ec_point_formats: list[int] = field(default_factory=list)
    n_signature_algorithms: int = 0
    has_alpn: bool = False
    has_session_ticket: bool = False
    has_key_share: bool = False
    has_sni: bool = False
    sni_len: int = 0          # LENGTH ONLY -- the name itself is discarded here
    has_psk: bool = False
    has_ech: bool = False
    complete: bool = True     # was the whole record captured?
    parse_error: bool = False
    parse_ok: bool = False

    @property
    def tls13(self) -> bool:
        """TLS 1.3 is negotiated through supported_versions. RFC 8446 pins
        legacy_version at 0x0303, so reading the version field instead would
        report 1.2 for every 1.3 handshake."""
        return 0x0304 in self.supported_versions

    def fingerprint(self) -> int:
        """JA3-shaped fingerprint over allowlisted parsed content only.

        Kept because it is measurable and because §7.2 needs to discuss it, but
        note what data/samples shows: a device emits 2-3 distinct values across
        thousands of handshakes, so this feature behaves like a per-device
        constant and will dominate any in-domain score it is given.
        """
        key = (self.version, tuple(self.cipher_suites), tuple(self.ext_types))
        return hash(key) & 0xFFFFFFFF


def _u16(b, i: int) -> int:
    return (b[i] << 8) | b[i + 1]


def parse_hex(s: str | None) -> bytes:
    if not s or not isinstance(s, str):
        return b""
    s = s.strip()
    if len(s) % 2:
        s = s[:-1]
    try:
        return bytes.fromhex(s)
    except ValueError:
        return b""


def parse_cipher_suites(hexstr: str | None) -> list[int]:
    b = parse_hex(hexstr)
    return [_u16(b, i) for i in range(0, len(b) - 1, 2)]


def parse_client_hello_extensions(hexstr: str | None) -> ClientHello:
    """Parse the extensions blob into types, lengths and permitted values.

    The blob in UNSW-IoTraffic starts at the first extension record, i.e. it is
    the extension_data list itself, not the whole ClientHello.
    """
    ch = ClientHello()
    b = parse_hex(hexstr)
    i, n = 0, len(b)
    if n < 4:
        return ch
    while i + 4 <= n:
        etype = _u16(b, i)
        elen = _u16(b, i + 2)
        i += 4
        if elen < 0 or i + elen > n:
            break  # truncated record: stop, keep what was parsed
        val = b[i: i + elen]
        i += elen
        ch.ext_types.append(etype)
        ch.ext_lengths.append(elen)

        if etype in VALUE_FORBIDDEN:
            # server_name: keep presence and length, discard the name.
            ch.has_sni = True
            if elen >= 5:
                ch.sni_len = _u16(val, 3)
            del val
        elif etype == EXT_SUPPORTED_GROUPS and elen >= 2:
            ln = _u16(val, 0)
            ch.supported_groups = [_u16(val, 2 + k) for k in range(0, min(ln, elen - 2) - 1, 2)]
        elif etype == EXT_EC_POINT_FORMATS and elen >= 1:
            ln = val[0]
            ch.ec_point_formats = list(val[1: 1 + ln])
        elif etype == EXT_SIGNATURE_ALGORITHMS and elen >= 2:
            ch.n_signature_algorithms = _u16(val, 0) // 2
        elif etype == EXT_ALPN:
            ch.has_alpn = True          # presence only, never the protocol list
        elif etype == EXT_SESSION_TICKET:
            ch.has_session_ticket = True
        elif etype == EXT_KEY_SHARE:
            ch.has_key_share = True
    ch.parse_ok = bool(ch.ext_types)
    return ch


def parse_dhcp_parameter_request_list(hexstr: str | None) -> list[int]:
    """DHCP option 55 in request order.

    Allowlisted: it fingerprints the networking stack, not the device. In
    data/samples three devices share 0103060c0f1c2a, which is the evidence for
    that claim -- it groups by stack.
    """
    b = parse_hex(hexstr)
    return list(b)


def client_hello_from_row(row) -> ClientHello:
    ch = parse_client_hello_extensions(row.get("client-hello-extensions"))
    ch.cipher_suites = parse_cipher_suites(row.get("client-hello-cipher-suites"))
    v = (row.get("client-hello-version") or "").strip().lower()
    ch.version = v if v in TLS_VERSIONS else ("other" if v else "")
    return ch

# --------------------------------------------------------------------------
# Custom Flow payload decoding (DATA2016 / DATA2025)
# --------------------------------------------------------------------------
# The Custom Flow representation encodes the bytes of the first 10 packets into
# C_000..C_2999, with -4 opening a packet, -8 closing it and -255 padding unused
# positions. Handshakes therefore come out of the FLOW record and need no pcap.

PKT_START, PKT_END, PAD = -4.0, -8.0, -255.0


def is_grease(v: int) -> bool:
    """RFC 8701 reserved values: 0x0a0a, 0x1a1a, ... 0xfafa.

    They are present in this data and must be excluded from every count, or
    cipher-suite and extension vocabularies fill up with noise that differs
    between captures.
    """
    return (v & 0x0F0F) == 0x0A0A and (v >> 8) == (v & 0xFF)


def split_payload_packets(row) -> list[list[int]]:
    """Decode one flow's C_* columns into per-packet byte lists."""
    out, cur, inside = [], [], False
    for v in row:
        if v == PKT_START:
            cur, inside = [], True
        elif v == PKT_END:
            if inside:
                out.append(cur)
            inside = False
        elif v == PAD:
            continue
        elif inside and 0.0 <= v <= 255.0:
            cur.append(int(v))
    if inside and cur:                       # capture ended mid-packet
        out.append(cur)
    return out


def parse_client_hello_record(p: list[int]) -> ClientHello | None:
    """Parse a TLS record that opens a ClientHello, from captured bytes.

    Sets `complete` from the record's own length field: the payload capture is
    bounded, so a long ClientHello is cut and its extension block then looks
    ABSENT rather than missing. Presence of an extension is sound evidence from
    a truncated record; absence is not, which is why `complete` travels with it.
    """
    ch = ClientHello()
    if len(p) < 6 or p[0] != 0x16 or p[1] != 0x03 or p[5] != 0x01:
        return None
    rec_len = _u16(p, 3)
    ch.complete = len(p) >= 5 + rec_len
    try:
        i = 9                                 # 5 record header + 4 handshake header
        v = f"{p[i]:02x}{p[i+1]:02x}"
        ch.version = v if v in TLS_VERSIONS else "other"
        i += 2 + 32                           # legacy_version + random
        i += 1 + p[i]                         # legacy_session_id
        cs_len = _u16(p, i)
        i += 2
        ch.cipher_suites = [c for c in (_u16(p, i + k) for k in range(0, cs_len - 1, 2))
                            if not is_grease(c)]
        i += cs_len
        i += 1 + p[i]                         # compression methods
        ext_total = _u16(p, i)
        i += 2
        end = min(len(p), i + ext_total)
        while i + 4 <= end:
            et, el = _u16(p, i), _u16(p, i + 2)
            i += 4
            if i + el > len(p):
                ch.complete = False
                break
            body = p[i: i + el]
            i += el
            if is_grease(et):
                continue
            ch.ext_types.append(et)
            ch.ext_lengths.append(el)
            if et == EXT_SUPPORTED_VERSIONS and el >= 1:
                n = body[0]
                ch.supported_versions = [
                    v for v in (_u16(body, 1 + k)
                                for k in range(0, min(n, el - 1) - 1, 2))
                    if not is_grease(v)]
            elif et == EXT_SERVER_NAME and el >= 5:
                ch.has_sni = True
                ch.sni_len = _u16(body, 3)    # LENGTH ONLY; the name stops here
            elif et == EXT_SUPPORTED_GROUPS and el >= 2:
                ln = _u16(body, 0)
                ch.supported_groups = [
                    g for g in (_u16(body, 2 + k)
                                for k in range(0, min(ln, el - 2) - 1, 2))
                    if not is_grease(g)]
            elif et == EXT_EC_POINT_FORMATS and el >= 1:
                ch.ec_point_formats = list(body[1: 1 + body[0]])
            elif et == EXT_SIGNATURE_ALGORITHMS and el >= 2:
                ch.n_signature_algorithms = _u16(body, 0) // 2
            elif et == EXT_ALPN:
                ch.has_alpn = True            # presence only
            elif et == EXT_SESSION_TICKET:
                ch.has_session_ticket = True
            elif et == EXT_KEY_SHARE:
                ch.has_key_share = True
            elif et == EXT_PRE_SHARED_KEY:
                ch.has_psk = True
            elif et == EXT_ECH:
                ch.has_ech = True
    except (IndexError, ValueError):
        ch.parse_error = True
    ch.parse_ok = bool(ch.ext_types)
    return ch


#: DHCP option 55. §4.6 of the proposal allowlists it as a stack fingerprint.
def parse_dhcp_parameter_request_list_from_payload(p: list[int]) -> list[int]:
    """Pull option 55 out of a captured DHCP packet's bytes.

    Layout: 236-byte BOOTP header, 4-byte magic cookie 63 82 53 63, then
    type-length-value options. Returns [] when the capture stops short.
    """
    try:
        i = p.index(0x63)
        while i + 3 < len(p):
            if p[i:i + 4] == [0x63, 0x82, 0x53, 0x63]:
                break
            i = p.index(0x63, i + 1)
        else:
            return []
        i += 4
        while i + 1 < len(p):
            code, ln = p[i], p[i + 1]
            if code == 255:
                break
            if code == 0:
                i += 1
                continue
            if i + 2 + ln > len(p):
                break
            if code == 55:
                return list(p[i + 2: i + 2 + ln])
            i += 2 + ln
    except (ValueError, IndexError):
        pass
    return []
