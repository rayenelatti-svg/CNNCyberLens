from __future__ import annotations

import argparse

import numpy as np
import torch
from scapy.all import sniff

from .flows import _flow_key, _packet_bytes
from .model import create_model
from .preprocess import flow_to_image
from .flows import Flow


def classify_packets(packets, model, classes):
    grouped = {}
    for packet in packets:
        key = _flow_key(packet)
        if key:
            grouped.setdefault(key, []).append(_packet_bytes(packet))
    model.eval()
    results = []
    with torch.no_grad():
        for key, raw_packets in grouped.items():
            flow = Flow(key, tuple(raw_packets), "", "live")
            image = torch.from_numpy(flow_to_image(flow)[None, None])
            probabilities = model(image).softmax(1)[0]
            index = int(probabilities.argmax())
            results.append({"flow": key, "category": classes[index],
                            "confidence": float(probabilities[index])})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seconds", type=int, default=10)
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location="cpu")
    model = create_model(
        saved.get("model_name", "native"),
        len(saved["classes"]),
        pretrained=False,
    )
    model.load_state_dict(saved["state_dict"])
    packets = sniff(iface=args.interface, timeout=args.seconds, store=True)
    for result in classify_packets(packets, model, saved["classes"]):
        print(result)


if __name__ == "__main__":
    main()
