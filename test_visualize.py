import numpy as np

from packetlens.visualize import save_category_report, save_grid, save_matrices


def test_visualization_writes_colored_grid_and_labeled_matrices(tmp_path):
    dataset = tmp_path / "flows.npz"
    grid = tmp_path / "grid.png"
    matrices = tmp_path / "matrices.txt"
    categories = tmp_path / "categories.png"
    np.savez(
        dataset,
        images=np.array(
            [
                [[[0.0, 1.0], [0.5, 0.25]]],
                [[[1.0, 0.0], [0.25, 0.75]]],
            ],
            dtype=np.float32,
        ),
        labels=np.array(["chat", "streaming"]),
        sessions=np.array(["s1", "s2"]),
    )

    save_grid(str(dataset), str(grid), count=2, show=False)
    save_matrices(str(dataset), str(matrices), count=2)
    save_category_report(str(dataset), str(categories), per_category=1)

    assert grid.exists()
    assert categories.exists()
    text = matrices.read_text(encoding="utf-8")
    assert "matrix 0 | category=chat | session=s1" in text
    assert "matrix 1 | category=streaming | session=s2" in text
    assert "0.0000 1.0000" in text
