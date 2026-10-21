from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .flows import iter_manifest
from .preprocess import flow_to_image


def build_dataset(
    manifest: str | Path,
    output: str | Path,
    window: int = 784,
    include_payload: bool = True,
):
    rows = pd.read_csv(manifest).fillna("")
    images, labels, sessions = [], [], []
    for row in rows.to_dict("records"):
        flows = list(iter_manifest([row], include_payload=include_payload))
        for flow in flows:
            images.append(flow_to_image(flow, window))
            labels.append(flow.label)
            sessions.append(flow.session)
        print(
            f"PROCESSED {row['label']} {row['pcap']} | "
            f"flows: {len(flows)} | total flows: {len(images)}",
            flush=True,
        )
    if not images:
        raise ValueError("No IP flows found in the manifest PCAP files")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        images=np.asarray(images, dtype=np.float32)[:, None, :, :],
        labels=np.asarray(labels),
        sessions=np.asarray(sessions),
    )


def main():
    parser = argparse.ArgumentParser(description="Convert PCAP flows to images")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("manifest")
    build.add_argument("output")
    build.add_argument("--window", type=int, default=784)
    build.add_argument(
        "--headers-only",
        action="store_true",
        help="exclude transport payload bytes for an ablation",
    )
    args = parser.parse_args()
    if args.command == "build":
        build_dataset(
            args.manifest,
            args.output,
            args.window,
            include_payload=not args.headers_only,
        )


if __name__ == "__main__":
    main()
