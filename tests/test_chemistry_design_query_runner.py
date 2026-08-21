import json

import joblib
import pandas as pd
import yaml

from inverse_gems.chemistry_design_query_runner import run_chemistry_design_query


class ToyChemistryEstimator:
    def predict(self, x):
        cash = 0.05 * x["x__chem_oxide_equiv_mol_CaO"].to_numpy() + 0.10 * x["x__chem_oxide_equiv_mol_SiO2"].to_numpy()
        porosity = 0.5 - 0.05 * x["x__chem_oxide_equiv_mol_CaO"].to_numpy()
        return pd.DataFrame({"y__amount_C_A_S_H": cash, "y__porosity": porosity}).to_numpy()


def test_run_chemistry_design_query_generates_candidates_and_searches(tmp_path):
    bundle = tmp_path / "model.joblib"
    inputs = [
        "x__chem_oxide_equiv_mol_CaO",
        "x__chem_oxide_equiv_mol_SiO2",
        "x__xgems_water_g",
        "x__temperature_celsius",
    ]
    joblib.dump(
        {
            "estimator": ToyChemistryEstimator(),
            "inputs": inputs,
            "targets": ["y__amount_C_A_S_H", "y__porosity"],
            "reaction_provenance": {},
        },
        bundle,
    )
    pd.DataFrame(
        {
            "target": ["y__amount_C_A_S_H", "y__porosity"],
            "r2": [0.9, 0.9],
            "mae": [0.01, 0.01],
        }
    ).to_csv(tmp_path / "target_metrics.csv", index=False)
    reference = tmp_path / "reference.csv"
    pd.DataFrame(
        {
            "meta__recipe_id": ["r1", "r2"],
            "meta__chem_hash": ["h1", "h2"],
            **{column: [0.0, 10.0] for column in inputs},
        }
    ).to_csv(reference, index=False)
    query = tmp_path / "query.yaml"
    query.write_text(
        yaml.safe_dump(
            {
                "name": "toy_chemistry_query",
                "material_system": "OPC_slag",
                "design_space": {"strict_materials": True},
                "age_days": 56.0,
                "inputs": {"w_b": {"equals": 0.40}},
                "targets": {"porosity": {"max": 1.0}},
                "ranking": {"mode": "lexicographic"},
                "preferences": [{"target": "C-A-S-H", "direction": "maximize"}],
                "search_top_k": 3,
            }
        ),
        encoding="utf-8",
    )

    out = run_chemistry_design_query(
        query=query,
        out=tmp_path / "run",
        model_bundle=bundle,
        reference_model_table=reference,
        n_candidates=8,
        seed=7,
        skip_validation=True,
        target_availability_policy="ignore",
    )

    candidates = pd.read_csv(out / "final_candidates.csv")
    summary = json.loads((out / "chemistry_design_query_summary.json").read_text(encoding="utf-8"))
    assert len(candidates) == 3
    assert (out / "generated_candidates" / "candidate_recipes.csv").exists()
    assert (out / "chemistry_candidate_table.csv").exists()
    assert (out / "domain_report" / "chemistry_domain_report.csv").exists()
    assert summary["material_system"] == "OPC_slag"
    assert summary["age_days"] == 56.0
    assert summary["strict_materials"] is True
    assert summary["final_stage"] == "surrogate_search"
    assert candidates["x__w_b"].round(8).eq(0.40).all()
    assert candidates["x__gypsum"].round(8).eq(0.0).all()
    assert "out_of_domain" in candidates.columns
    assert "nearest_scaled_distance" in candidates.columns
    stages = {stage["name"]: stage for stage in summary["inverse_design_flow_summary"]["stages"]}
    assert stages["candidate_generation"]["rows_out"] == 8
    assert stages["reactive_chemistry_table"]["status"] == "complete"
    assert stages["xgems_validation"]["status"] == "skipped"
    assert (out / "inverse_design_flow_summary.md").exists()
