from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_samples(dataset: str, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(dataset, allow_pickle=False)
    if count < 1:
        raise ValueError("count must be positive")
    count = min(count, len(data["images"]))
    if count == 0:
        raise ValueError("dataset contains no images")
    return data["images"][:count, 0], data["labels"][:count], data["sessions"][:count]


def save_grid(
    dataset: str,
    output: str = "reports/sample_grid.png",
    count: int = 25,
    show: bool = False,
):
    images, labels, _ = _load_samples(dataset, count)
    columns = 5
    rows = int(np.ceil(len(images) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(10, 2 * rows))
    axes = np.asarray(axes).reshape(-1)
    for index, axis in enumerate(axes):
        axis.axis("off")
        if index >= len(images):
            continue
        axis.imshow(images[index], cmap="viridis", vmin=0, vmax=1)
        axis.set_title(f"{index}: {labels[index]}", fontsize=8)
    figure.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    if show:
        plt.show()
    plt.close(figure)


def save_matrices(
    dataset: str,
    output: str = "reports/matrices.txt",
    count: int = 25,
) -> None:
    """Write each byte-map matrix with its sample category and session."""
    images, labels, sessions = _load_samples(dataset, count)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for index, (image, label, session) in enumerate(
            zip(images, labels, sessions)
        ):
            stream.write(f"matrix {index} | category={label} | session={session}\n")
            np.savetxt(stream, image, fmt="%.4f", delimiter=" ")
            stream.write("\n")


def save_category_report(
    dataset: str,
    output: str = "reports/category_patterns.png",
    per_category: int = 6,
) -> None:
    """Render several examples and mean byte-map patterns for every category."""
    if per_category < 1:
        raise ValueError("per_category must be positive")
    data = np.load(dataset, allow_pickle=False)
    images = data["images"][:, 0]
    labels = data["labels"]
    categories = list(dict.fromkeys(str(label) for label in labels))
    columns = max(1, min(per_category, 6))
    rows = len(categories) * 2
    figure, axes = plt.subplots(
        rows, columns, figsize=(2.5 * columns, 2.8 * rows), squeeze=False
    )
    for category_index, category in enumerate(categories):
        indices = np.flatnonzero(labels.astype(str) == category)[:per_category]
        examples = images[indices]
        mean = examples.mean(axis=0)
        standard_deviation = examples.std(axis=0)
        for column in range(columns):
            axis = axes[category_index * 2, column]
            axis.axis("off")
            if column < len(examples):
                axis.imshow(examples[column], cmap="viridis", vmin=0, vmax=1)
                axis.set_title(f"{category}\nflow {column + 1}", fontsize=8)
        mean_axis = axes[category_index * 2 + 1, 0]
        mean_axis.imshow(mean, cmap="viridis", vmin=0, vmax=1)
        mean_axis.set_title(f"{category}\nmean ({len(examples)} flows)", fontsize=8)
        mean_axis.axis("off")
        for column in range(1, columns):
            axis = axes[category_index * 2 + 1, column]
            axis.axis("off")
        figure.text(
            0.01,
            1 - (category_index * 2 + 1.5) / rows,
            f"{category}: mean intensity={mean.mean():.4f}, "
            f"variation={standard_deviation.mean():.4f}",
            va="center",
            fontsize=9,
        )
    figure.suptitle("PacketLens: examples and mean pattern by category")
    figure.tight_layout()
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Render PacketLens byte-map samples")
    parser.add_argument("dataset")
    parser.add_argument("--output", default="reports/sample_grid.png")
    parser.add_argument(
        "--matrix-output",
        default="reports/matrices.txt",
        help="write matrix values with category and session metadata",
    )
    parser.add_argument("--count", type=int, default=25)
    parser.add_argument(
        "--category-output",
        default="reports/category_patterns.png",
        help="write examples and mean pattern for each category",
    )
    parser.add_argument(
        "--per-category",
        type=int,
        default=6,
        help="number of flow images shown for each category",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="display the colored Matplotlib diagram interactively",
    )
    args = parser.parse_args()
    save_grid(args.dataset, args.output, args.count, show=args.show)
    save_matrices(args.dataset, args.matrix_output, args.count)
    save_category_report(args.dataset, args.category_output, args.per_category)


if __name__ == "__main__":
    main()
