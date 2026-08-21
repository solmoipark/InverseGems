import json

import joblib
import pandas as pd

from inverse_gems.chemistry_domain import write_chemistry_domain_report


class DomainBundleEstimator:
    def predict(self, x):
        return x[["x__a"]].to_numpy()


def test_chemistry_domain_report_flags_out_of_range_candidate(tmp_path):
    reference = tmp_path / "reference.csv"
    pd.DataFrame(
        {
            "meta__recipe_id": ["r1", "r2"],
            "meta__chem_hash": ["h1", "h2"],
            "x__a": [0.0, 1.0],
            "x__b": [0.0, 1.0],
        }
    ).to_csv(reference, index=False)
    candidates = tmp_path / "candidates.csv"
    pd.DataFrame(
        {
            "meta__recipe_id": ["inside", "outside"],
            "meta__chem_hash": ["c1", "c2"],
            "x__a": [0.5, 2.0],
            "x__b": [0.5, 0.5],
        }
    ).to_csv(candidates, index=False)
    bundle = tmp_path / "model.joblib"
    joblib.dump({"estimator": DomainBundleEstimator(), "inputs": ["x__a", "x__b"], "targets": ["y__a"]}, bundle)

    out = write_chemistry_domain_report(
        reference_model_table=reference,
        candidate_table=candidates,
        model_bundle=bundle,
        out=tmp_path / "domain",
        nearest_distance_warn=1.0,
    )

    report = pd.read_csv(out / "chemistry_domain_report.csv")
    summary = json.loads((out / "chemistry_domain_summary.json").read_text(encoding="utf-8"))
    assert report["out_of_domain"].tolist() == [False, True]
    assert report.loc[1, "outside_input_range_columns"] == "x__a"
    assert summary["out_of_domain_count"] == 1
