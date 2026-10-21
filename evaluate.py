from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .model import create_model


def evaluate(dataset, checkpoint, report_dir="reports"):
    data = np.load(dataset, allow_pickle=False)
    saved = torch.load(checkpoint, map_location="cpu")
    model = create_model(
        saved.get("model_name", "native"), len(saved["classes"]), pretrained=False
    )
    model.load_state_dict(saved["state_dict"])
    model.eval()
    indices = saved["test_indices"]
    with torch.no_grad():
        predictions = model(torch.from_numpy(data["images"][indices])).argmax(1).numpy()
    truth = np.array([saved["classes"].index(x) for x in data["labels"][indices]])
    class_indices = np.arange(len(saved["classes"]))
    results = {
        "accuracy": accuracy_score(truth, predictions),
        "macro_f1": f1_score(
            truth,
            predictions,
            labels=class_indices,
            average="macro",
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            truth, predictions, labels=class_indices
        ).tolist(),
        "classification_report": classification_report(
            truth,
            predictions,
            labels=class_indices,
            target_names=saved["classes"],
            zero_division=0,
        ),
    }
    report_path = __import__("pathlib").Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)
    with (report_path / "evaluation.json").open("w") as report:
        json.dump(results, report, indent=2, default=str)
    plt.figure(figsize=(8, 6))
    matrix = np.asarray(results["confusion_matrix"])
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar()
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            plt.text(column, row, matrix[row, column], ha="center", va="center")
    plt.xticks(range(len(saved["classes"])), saved["classes"], rotation=45, ha="right")
    plt.yticks(range(len(saved["classes"])), saved["classes"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(report_path / "confusion_matrix.png", dpi=160)
    plt.close()
    np.savetxt(
        report_path / "predictions.csv",
        np.column_stack((data["labels"][indices], [saved["classes"][x] for x in predictions])),
        fmt="%s", delimiter=",", header="actual,predicted", comments="",
    )
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate PacketLens CNN")
    parser.add_argument("dataset")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()
    print(json.dumps(
        evaluate(args.dataset, args.checkpoint, args.report_dir),
        indent=2, default=str,
    ))


if __name__ == "__main__":
    main()
