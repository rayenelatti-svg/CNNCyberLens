import numpy as np
import pytest

from packetlens.flows import Flow
from packetlens.preprocess import flow_to_image


def test_flow_is_padded_and_normalized():
    flow = Flow(("a", "b", 1, 2, "6"), (bytes([0, 255]),), "chat", "s1")
    image = flow_to_image(flow)
    assert image.shape == (28, 28)
    assert image[0, 0] == 0
    assert image[0, 1] == 1
    assert np.all(image.flat[2:] == 0)


def test_window_must_be_square():
    flow = Flow(("a", "b", 1, 2, "6"), (b"x",), "chat", "s1")
    with pytest.raises(ValueError):
        flow_to_image(flow, 101)


def test_header_only_bytes_mask_transport_ports():
    from scapy.layers.inet import IP, TCP

    from packetlens.flows import _packet_bytes

    packet = (
        IP(src="192.0.2.1", dst="198.51.100.2")
        / TCP(sport=1234, dport=443)
        / b"payload"
    )
    encoded = _packet_bytes(packet, include_payload=False)
    assert encoded[12:20] == b"\0" * 8
    assert encoded[20:24] == b"\0" * 4
    assert len(encoded) == 40
    assert b"payload" not in encoded
