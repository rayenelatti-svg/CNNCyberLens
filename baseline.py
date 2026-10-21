from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder


def run_baseline(dataset, seed=42):
    data = np.load(dataset, allow_pickle=False)
    features = np.stack([
        data["images"].mean(axis=(1, 2, 3)),
        data["images"].std(axis=(1, 2, 3)),
        (data["images"] > 0).mean(axis=(1, 2, 3)),
    ], axis=1)
    encoder = LabelEncoder()
    labels = encoder.fit_transform(data["labels"])
    train, test = next(GroupShuffleSplit(
        n_splits=1, test_size=.2, random_state=seed
    ).split(features, labels, data["sessions"]))
    model = RandomForestClassifier(n_estimators=200, random_state=seed, class_weight="balanced")
    model.fit(features[train], labels[train])
    prediction = model.predict(features[test])
    return {"accuracy": accuracy_score(labels[test], prediction),
            "macro_f1": f1_score(labels[test], prediction, average="macro", zero_division=0)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()
    print(json.dumps(run_baseline(args.dataset), indent=2))


if __name__ == "__main__":
    main()
