from __future__ import annotations

from math import isqrt

import numpy as np

from .flows import Flow, payload_bytes


def flow_to_image(flow: Flow, window: int = 784) -> np.ndarray:
    if window < 1:
        raise ValueError("window must be positive")
    side = isqrt(window)
    if side * side != window:
        raise ValueError("window must be a perfect square")
    values = np.frombuffer(payload_bytes(flow)[:window], dtype=np.uint8)
    values = np.pad(values, (0, window - len(values)))
    return (values.astype(np.float32) / 255.0).reshape(side, side)
