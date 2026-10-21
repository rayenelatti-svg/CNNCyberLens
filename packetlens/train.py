from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .model import create_model


def train_dataset(
    dataset,
    output,
    epochs=10,
    seed=42,
    model_name="mobilenet_v3_small",
    pretrained=True,
    report_dir="reports",
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    data = np.load(dataset, allow_pickle=False)
    encoder = LabelEncoder()
    labels = encoder.fit_transform(data["labels"])
    folds = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=seed
    ).split(data["images"], labels, data["sessions"])
    fold_indices = [test_indices for _, test_indices in folds]
    test_idx = fold_indices[0]
    validation_idx = fold_indices[1]
    train_idx = np.concatenate(fold_indices[2:])
    x_train = torch.from_numpy(data["images"][train_idx])
    y_train = torch.from_numpy(labels[train_idx]).long()
    model = create_model(model_name, len(encoder.classes_), pretrained=pretrained)
    counts = np.bincount(labels[train_idx], minlength=len(encoder.classes_))
    sample_weights = torch.tensor(
        1.0 / np.maximum(counts[labels[train_idx]], 1), dtype=torch.double
    )
    sampler = WeightedRandomSampler(
        sample_weights, len(sample_weights), replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=64, sampler=sampler
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = []
    best_state = None
    best_validation_f1 = -1.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, targets in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(images), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(targets)
        model.eval()
        with torch.no_grad():
            validation_predictions = model(
                torch.from_numpy(data["images"][validation_idx])
            ).argmax(1).numpy()
        validation_f1 = f1_score(
            labels[validation_idx],
            validation_predictions,
            average="macro",
            labels=np.arange(len(encoder.classes_)),
            zero_division=0,
        )
        if validation_f1 > best_validation_f1:
            best_validation_f1 = validation_f1
            best_state = copy.deepcopy(model.state_dict())
        scheduler.step()
        history.append({
            "epoch": epoch + 1,
            "loss": running_loss / len(y_train),
            "validation_accuracy": accuracy_score(
                labels[validation_idx], validation_predictions
            ),
            "validation_macro_f1": validation_f1,
        })
    if best_state is None:
        raise RuntimeError("Training did not produce a model checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_predictions = model(
            torch.from_numpy(data["images"][test_idx])
        ).argmax(1).numpy()
    test_accuracy = accuracy_score(labels[test_idx], test_predictions)
    test_macro_f1 = f1_score(
        labels[test_idx], test_predictions, average="macro", zero_division=0
    )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    with (Path(report_dir) / "training_history.json").open("w") as report:
        json.dump(history, report, indent=2)
    plt.plot([row["epoch"] for row in history], [row["loss"] for row in history], label="loss")
    plt.plot(
        [row["epoch"] for row in history],
        [row["validation_macro_f1"] for row in history],
        label="validation macro-F1",
    )
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(report_dir) / "training_curves.png", dpi=160)
    plt.close()
    torch.save({
        "state_dict": model.state_dict(),
        "classes": encoder.classes_.tolist(),
        "test_indices": test_idx.tolist(),
        "validation_indices": validation_idx.tolist(),
        "model_name": model_name,
        "window": int(data["images"].shape[-1] ** 2),
    }, output)
    return {
        "train_flows": len(train_idx),
        "validation_flows": len(validation_idx),
        "test_flows": len(test_idx),
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,
        "best_validation_macro_f1": best_validation_f1,
        "classes": encoder.classes_.tolist(),
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="Train PacketLens CNN")
    parser.add_argument("dataset")
    parser.add_argument("--output", default="models/cnn.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--model",
        choices=("mobilenet_v3_small", "native", "native_enhanced"),
        default="mobilenet_v3_small",
    )
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()
    print(json.dumps(train_dataset(
        args.dataset, args.output, args.epochs, model_name=args.model,
        pretrained=not args.no_pretrained, report_dir=args.report_dir,
    ), indent=2))


if __name__ == "__main__":
    main()
