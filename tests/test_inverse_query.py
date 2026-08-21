import csv
import yaml
import pandas as pd

from inverse_gems.batch import run_batch_cached
from inverse_gems.feature_table import build_feature_table
from inverse_gems.inverse_query import run_inverse_query


def test_inverse_query_filters_and_writes_candidates(tmp_path):
    recipes = tmp_path / "recipes.csv"
    with recipes.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["recipe_id", "OPC", "fly_ash", "w_b", "age_days", "age_bin"])
        writer.writeheader()
        writer.writerow({"recipe_id": "a", "OPC": 30, "fly_ash": 70, "w_b": 0.4, "age_days": 1, "age_bin": "around_1d"})
        writer.writerow({"recipe_id": "b", "OPC": 90, "fly_ash": 10, "w_b": 0.7, "age_days": 28, "age_bin": "standard"})
    db = tmp_path / "db"
    run_batch_cached(recipes_csv=recipes, db=db, use_mock=True)
    selection = tmp_path / "selection.yaml"
    selection.write_text(
        yaml.safe_dump(
            {
                "phase_amounts": {"include": ["Mock Portlandite", "Missing Phase"]},
                "phase_volumes": {"include": []},
                "aqueous_species": {"include": []},
                "scalars": {"include": ["pH", "porosity"]},
            }
        ),
        encoding="utf-8",
    )
    feature = tmp_path / "features.csv"
    build_feature_table(db=db, selection=selection, out=feature, output_format="csv")
    query = tmp_path / "query.yaml"
    query.write_text(
        yaml.safe_dump(
            {
                "targets": {"phase_amounts": {"Missing Phase": {"min": 0, "max": 0}}, "scalars": {"pH": {"min": 0, "max": 14}}},
                "recipe_constraints": {"OPC": {"min": 20, "max": 60}, "age_bin": {"include": ["around_1d"]}},
                "objectives": {"minimize": {"OPC": 1.0}, "maximize": {"Missing Phase": 1.0}},
                "top_k": 5,
                "missing_phase_policy": "zero",
            }
        ),
        encoding="utf-8",
    )
    out = run_inverse_query(feature_table=feature, query=query, out=tmp_path / "query_out")
    candidates = pd.read_csv(out / "candidates.csv")
    assert len(candidates) == 1
    assert candidates.loc[0, "recipe_id"] == "a"
    assert {"rank", "score", "recipe_id", "chem_hash", "age_bin", "porosity"}.issubset(candidates.columns)
