from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from scapy.all import IP, IPv6, PcapReader, TCP, UDP


@dataclass(frozen=True)
class Flow:
    key: tuple[str, str, int, int, str]
    packets: tuple[bytes, ...]
    label: str
    session: str


def _endpoint(packet):
    if IP in packet:
        return packet[IP].src, packet[IP].dst, packet[IP].proto
    if IPv6 in packet:
        return packet[IPv6].src, packet[IPv6].dst, packet[IPv6].nh
    return None


def _flow_key(packet):
    endpoint = _endpoint(packet)
    if endpoint is None:
        return None
    source, destination, protocol = endpoint
    source_port = int(getattr(packet.payload, "sport", 0))
    destination_port = int(getattr(packet.payload, "dport", 0))
    left = (source, source_port)
    right = (destination, destination_port)
    first, second = sorted((left, right))
    return (*first, *second, str(protocol))


def _packet_bytes(packet, include_payload: bool = True) -> bytes:
    if IP in packet:
        layer = packet[IP]
    elif IPv6 in packet:
        layer = packet[IPv6]
    else:
        return b""
    raw = bytearray(bytes(layer))
    if IPv6 in packet:
        address_size = 16
        address_offsets = (8, 24)
    else:
        address_size = 4
        address_offsets = (12, 16)
    for offset in address_offsets:
        raw[offset:offset + address_size] = b"\0" * address_size
    transport_offset = int(getattr(layer, "ihl", 5) or 5) * 4 if IP in packet else 40
    raw[transport_offset:transport_offset + 4] = b"\0" * 4
    if include_payload:
        return bytes(raw)
    header_length = int(getattr(layer, "ihl", 5) or 5) * 4 if IP in packet else 40
    if TCP in packet:
        header_length += int(getattr(packet[TCP], "dataofs", 5) or 5) * 4
    elif UDP in packet:
        header_length += 8
    return bytes(raw[:header_length])


def read_flows(
    path: str | Path,
    label: str,
    session: str | None = None,
    include_payload: bool = True,
) -> list[Flow]:
    """Read one PCAP and return bidirectional, anonymized flows."""
    path = Path(path)
    packets_by_key: dict[tuple, list[bytes]] = {}
    with PcapReader(str(path)) as reader:
        for packet in reader:
            key = _flow_key(packet)
            if key is None:
                continue
            payload = _packet_bytes(packet, include_payload=include_payload)
            if payload:
                packets_by_key.setdefault(key, []).append(payload)
    flow_session = session or path.stem
    return [
        Flow(key=key, packets=tuple(packets), label=label, session=flow_session)
        for key, packets in packets_by_key.items()
    ]


def iter_manifest(rows: Iterable[dict], include_payload: bool = True):
    for row in rows:
        yield from read_flows(
            row["pcap"],
            row["label"],
            row.get("session"),
            include_payload=include_payload,
        )


def payload_bytes(flow: Flow) -> bytes:
    return b"".join(flow.packets)
