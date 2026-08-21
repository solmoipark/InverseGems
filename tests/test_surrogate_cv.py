import numpy as np
import pandas as pd
import pytest
import yaml

from inverse_gems.surrogate_cv import evaluate_surrogate_repeated_cv


def _cv_config(tmp_path):
    config = tmp_path / "surrogate_cv.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "estimator": {"kind": "extra_trees", "n_estimators": 20, "random_state": 0},
                "split": {"strategy": "group_shuffle", "test_size": 0.25, "random_state": 11},
            }
        ),
        encoding="utf-8",
    )
    return config


def test_repeated_cv_reports_spread(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for index in range(120):
        x1 = rng.uniform(0, 1)
        x2 = rng.uniform(0, 1)
        rows.append(
            {
                "meta__recipe_id": f"g{index % 40}",
                "x__a": x1,
                "x__b": x2,
                "y__dense": 2.0 * x1 + x2 + rng.normal(0, 0.05),
                "y__sparse": (x1 if x1 > 0.9 else 0.0),
            }
        )
    table = tmp_path / "table.csv"
    pd.DataFrame(rows).to_csv(table, index=False)

    out = evaluate_surrogate_repeated_cv(
        model_table=table, out=tmp_path / "cv", config=_cv_config(tmp_path), n_repeats=3
    )
    aggregated = pd.read_csv(out / "cv_target_metrics.csv")
    assert set(aggregated["target"]) == {"y__dense", "y__sparse"}
    dense = aggregated[aggregated.target == "y__dense"].iloc[0]
    assert dense["r2_mean"] > 0.8
    assert dense["n_repeats"] == 3
    assert not pd.isna(dense["r2_std"])
    per_repeat = pd.read_csv(out / "cv_per_repeat_metrics.csv")
    assert len(per_repeat) == 6  # 2 targets x 3 repeats
    assert per_repeat["random_state"].nunique() == 3
    assert (out / "cv_report.md").exists()


def test_repeated_cv_target_filter_and_min_repeats(tmp_path):
    rows = [
        {"meta__recipe_id": f"g{i}", "x__a": i * 0.1, "y__dense": i * 0.2, "y__other": 1.0}
        for i in range(40)
    ]
    table = tmp_path / "t.csv"
    pd.DataFrame(rows).to_csv(table, index=False)
    out = evaluate_surrogate_repeated_cv(
        model_table=table, out=tmp_path / "cv", config=_cv_config(tmp_path),
        n_repeats=2, targets=["dense"],
    )
    aggregated = pd.read_csv(out / "cv_target_metrics.csv")
    assert list(aggregated["target"]) == ["y__dense"]

    with pytest.raises(ValueError, match="at least 2"):
        evaluate_surrogate_repeated_cv(
            model_table=table, out=tmp_path / "cv2", config=_cv_config(tmp_path), n_repeats=1
        )
