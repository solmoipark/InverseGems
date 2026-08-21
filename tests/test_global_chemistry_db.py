import json

import joblib
import numpy as np
import pandas as pd

from inverse_gems.global_chemistry_db import (
    acquire_global_chemistry_candidates,
    import_cached_db_into_global_db,
    initialize_global_chemistry_db,
    lookup_global_chemistry,
)
from inverse_gems.database import InverseGemsDatabase


class TinyEstimator:
    def predict(self, x):
        return x[["x__chem_oxide_equiv_mol_CaO"]].to_numpy()


class MultiTargetEstimator:
    def predict(self, x):
        base = x[["x__chem_oxide_equiv_mol_CaO"]].to_numpy()
        return np.hstack([base, base * 0.0 + 10.0, base * 0.0 + 1.0])


class OpposingTargetEstimator:
    def predict(self, x):
        base = x[["x__chem_oxide_equiv_mol_CaO"]].to_numpy()
        return np.hstack([base, 10.0 - base])


def _chem_row(recipe_id: str, chem_hash: str, cao: float, sio2: float, opc: float, slag: float, age: float) -> dict:
    row = {
        "meta__recipe_id": recipe_id,
        "meta__chem_hash": chem_hash,
        "meta__template_name": "toy",
        "meta__material_system": "OPC_slag",
        "meta__target_profile": "toy_target",
        "x__OPC": opc,
        "x__slag": slag,
        "x__fly_ash": 0.0,
        "x__metakaolin": 0.0,
        "x__silica_fume": 0.0,
        "x__limestone": 0.0,
        "x__gypsum": 0.0,
        "x__w_b": 0.4,
        "x__water_g": 40.0,
        "x__age_days": age,
        "x__temperature_celsius": 20.0,
        "x__xgems_water_g": 40.0,
    }
    for oxide in ["CaO", "SiO2", "Al2O3", "Fe2O3", "MgO", "SO3", "Na2O", "K2O", "CO2", "H2O"]:
        row[f"x__chem_oxide_equiv_mol_{oxide}"] = 0.0
    row["x__chem_oxide_equiv_mol_CaO"] = cao
    row["x__chem_oxide_equiv_mol_SiO2"] = sio2
    return row


def test_global_lookup_and_acquisition_write_user_testable_outputs(tmp_path):
    db = tmp_path / "global_db"
    manifest = initialize_global_chemistry_db(db=db)
    assert manifest["reactive_chemistry_centered"] is True
    assert (db / "inverse_gems.sqlite").exists()

    reference = tmp_path / "reference.csv"
    pd.DataFrame(
        [
            _chem_row("ref_1", "h1", 1.0, 0.5, 70.0, 30.0, 28.0),
            _chem_row("ref_2", "h2", 2.0, 1.0, 60.0, 40.0, 90.0),
        ]
    ).to_csv(reference, index=False)
    candidate = tmp_path / "candidate.csv"
    pd.DataFrame(
        [
            _chem_row("hit", "h1", 1.0, 0.5, 70.0, 30.0, 28.0),
            _chem_row("novel", "h3", 3.0, 1.5, 50.0, 50.0, 56.0),
        ]
    ).to_csv(candidate, index=False)
    bundle = tmp_path / "model.joblib"
    joblib.dump(
        {
            "estimator": TinyEstimator(),
            "inputs": ["x__chem_oxide_equiv_mol_CaO", "x__chem_oxide_equiv_mol_SiO2", "x__xgems_water_g", "x__temperature_celsius"],
            "targets": ["y__amount_C_A_S_H"],
        },
        bundle,
    )

    lookup_dir = lookup_global_chemistry(
        db=db,
        out=tmp_path / "lookup",
        candidate_table=candidate,
        reference_model_table=reference,
        model_bundle=bundle,
        nearest_distance_warn=1.0,
    )
    lookup = pd.read_csv(lookup_dir / "global_chemistry_lookup.csv")
    assert lookup["exact_chem_hash_hit"].tolist() == [True, False]
    assert bool(lookup.loc[1, "out_of_domain"]) is True

    acq_dir = acquire_global_chemistry_candidates(
        db=db,
        out=tmp_path / "acq",
        candidate_table=candidate,
        reference_model_table=reference,
        model_bundle=bundle,
        max_candidates=1,
        nearest_distance_warn=1.0,
    )
    selected = pd.read_csv(acq_dir / "acquisition_candidates.csv")
    recipes = pd.read_csv(acq_dir / "acquisition_recipes.csv")
    summary = json.loads((acq_dir / "acquisition_summary.json").read_text(encoding="utf-8"))
    assert selected.iloc[0]["meta__recipe_id"] == "novel"
    assert selected.iloc[0]["acquisition_bucket"] == "novel_out_of_domain"
    assert "new reactive chemistry" in selected.iloc[0]["acquisition_reason"]
    assert recipes.iloc[0]["recipe_id"] == "novel"
    assert recipes.iloc[0]["target_profile"] == "toy_target"
    assert summary["selected_rows"] == 1
    assert summary["selection_bucket_counts"] == {"novel_out_of_domain": 1}
    assert summary["top_candidates"][0]["meta__recipe_id"] == "novel"
    assert (acq_dir / "acquisition_candidates.md").exists()


def test_global_lookup_warns_when_recipe_projection_has_no_dat_lst(tmp_path):
    db = tmp_path / "global_db"
    initialize_global_chemistry_db(db=db)
    recipes = tmp_path / "recipes.csv"
    pd.DataFrame(
        [
            {
                "recipe_id": "r1",
                "OPC": 60.0,
                "slag": 40.0,
                "w_b": 0.4,
                "water_g": 40.0,
                "age_days": 28.0,
            }
        ]
    ).to_csv(recipes, index=False)

    lookup_dir = lookup_global_chemistry(db=db, out=tmp_path / "lookup_no_dat", recipes_csv=recipes)

    summary = json.loads((lookup_dir / "global_chemistry_lookup_summary.json").read_text(encoding="utf-8"))
    assert summary["dat_lst"] is None
    assert "dat_lst was not provided" in summary["warnings"][0]


def test_global_acquisition_penalizes_near_exact_chemistry_vectors(tmp_path):
    db = tmp_path / "global_db"
    initialize_global_chemistry_db(db=db)
    reference = tmp_path / "reference.csv"
    pd.DataFrame([_chem_row("ref", "known_hash", 1.0, 0.5, 70.0, 30.0, 28.0)]).to_csv(reference, index=False)
    candidate = tmp_path / "candidate.csv"
    pd.DataFrame(
        [
            _chem_row("near_known", "different_hash", 1.0 + 1.0e-12, 0.5, 70.0, 30.0, 28.0),
            _chem_row("novel", "novel_hash", 3.0, 1.5, 50.0, 50.0, 56.0),
        ]
    ).to_csv(candidate, index=False)

    lookup_dir = lookup_global_chemistry(
        db=db,
        out=tmp_path / "lookup_near",
        candidate_table=candidate,
        reference_model_table=reference,
    )
    lookup = pd.read_csv(lookup_dir / "global_chemistry_lookup.csv")
    by_id = lookup.set_index("meta__recipe_id")
    assert bool(by_id.loc["near_known", "exact_chem_hash_hit"]) is False
    assert bool(by_id.loc["near_known", "near_exact_chemistry_hit"]) is True

    acq_dir = acquire_global_chemistry_candidates(
        db=db,
        out=tmp_path / "acq_near",
        candidate_table=candidate,
        reference_model_table=reference,
        max_candidates=1,
    )
    selected = pd.read_csv(acq_dir / "acquisition_candidates.csv")
    assert selected.iloc[0]["meta__recipe_id"] == "novel"


def test_global_acquisition_can_prioritize_low_confidence_targets(tmp_path):
    db = tmp_path / "global_db"
    initialize_global_chemistry_db(db=db)
    candidate = tmp_path / "candidate.csv"
    pd.DataFrame(
        [
            _chem_row("low_target", "h_low", 0.1, 0.5, 60.0, 40.0, 28.0),
            _chem_row("high_target", "h_high", 10.0, 0.5, 60.0, 40.0, 28.0),
        ]
    ).to_csv(candidate, index=False)
    bundle = tmp_path / "model.joblib"
    joblib.dump(
        {
            "estimator": TinyEstimator(),
            "inputs": ["x__chem_oxide_equiv_mol_CaO", "x__chem_oxide_equiv_mol_SiO2", "x__xgems_water_g", "x__temperature_celsius"],
            "targets": ["y__amount_C_A_S_H"],
        },
        bundle,
    )

    acq_dir = acquire_global_chemistry_candidates(
        db=db,
        out=tmp_path / "targeted_acq",
        candidate_table=candidate,
        model_bundle=bundle,
        max_candidates=1,
        priority_targets=["C-A-S-H"],
        target_priority_weight=50.0,
    )

    selected = pd.read_csv(acq_dir / "acquisition_candidates.csv")
    summary = json.loads((acq_dir / "acquisition_summary.json").read_text(encoding="utf-8"))
    assert selected.iloc[0]["meta__recipe_id"] == "high_target"
    assert selected.iloc[0]["priority_target_score"] == 1.0
    assert summary["priority_targets"]["resolved_targets"] == ["y__amount_C_A_S_H"]
    assert summary["priority_targets"]["enabled"] is True


def test_global_acquisition_filters_diagnostic_priority_targets_by_kind(tmp_path):
    db = tmp_path / "global_db"
    initialize_global_chemistry_db(db=db)
    candidate = tmp_path / "candidate.csv"
    pd.DataFrame(
        [
            _chem_row("low_phase", "h_low", 0.1, 0.5, 60.0, 40.0, 28.0),
            _chem_row("high_phase", "h_high", 10.0, 0.5, 60.0, 40.0, 28.0),
        ]
    ).to_csv(candidate, index=False)
    diagnostics = tmp_path / "diagnostics.csv"
    pd.DataFrame(
        [
            {"target_column": "y__amount_ettringite", "status": "usable_with_caution"},
            {"target_column": "y__pH", "status": "usable_with_caution"},
            {"target_column": "y__porosity", "status": "not_recommended"},
        ]
    ).to_csv(diagnostics, index=False)
    bundle = tmp_path / "model.joblib"
    joblib.dump(
        {
            "estimator": MultiTargetEstimator(),
            "inputs": ["x__chem_oxide_equiv_mol_CaO", "x__chem_oxide_equiv_mol_SiO2", "x__xgems_water_g", "x__temperature_celsius"],
            "targets": ["y__amount_ettringite", "y__pH", "y__porosity"],
        },
        bundle,
    )

    acq_dir = acquire_global_chemistry_candidates(
        db=db,
        out=tmp_path / "phase_acq",
        candidate_table=candidate,
        model_bundle=bundle,
        max_candidates=1,
        priority_targets_from_diagnostics=diagnostics,
        priority_target_kinds=["phase"],
        target_priority_weight=50.0,
    )

    selected = pd.read_csv(acq_dir / "acquisition_candidates.csv")
    summary = json.loads((acq_dir / "acquisition_summary.json").read_text(encoding="utf-8"))
    assert selected.iloc[0]["meta__recipe_id"] == "high_phase"
    assert summary["priority_targets"]["requested_targets"] == ["y__amount_ettringite"]
    assert summary["priority_targets"]["resolved_targets"] == ["y__amount_ettringite"]
    assert summary["priority_target_diagnostics_filter"]["kinds"] == ["phase"]


def test_global_acquisition_uses_ranked_priority_report_limit(tmp_path):
    db = tmp_path / "global_db"
    initialize_global_chemistry_db(db=db)
    candidate = tmp_path / "candidate.csv"
    pd.DataFrame(
        [
            _chem_row("high_a", "h_a", 9.0, 0.5, 60.0, 40.0, 28.0),
            _chem_row("high_b", "h_b", 1.0, 0.5, 60.0, 40.0, 28.0),
        ]
    ).to_csv(candidate, index=False)
    diagnostics = tmp_path / "active_learning_target_priorities.csv"
    pd.DataFrame(
        [
            {
                "target_column": "y__amount_target_b",
                "status": "usable_with_caution",
                "evaluation_reliability": "unstable_low_test_nonzero",
                "active_learning_priority_score": 100.0,
            },
            {
                "target_column": "y__amount_target_a",
                "status": "usable_with_caution",
                "evaluation_reliability": "sparse_full_distribution",
                "active_learning_priority_score": 10.0,
            },
        ]
    ).to_csv(diagnostics, index=False)
    bundle = tmp_path / "model.joblib"
    joblib.dump(
        {
            "estimator": OpposingTargetEstimator(),
            "inputs": ["x__chem_oxide_equiv_mol_CaO", "x__chem_oxide_equiv_mol_SiO2", "x__xgems_water_g", "x__temperature_celsius"],
            "targets": ["y__amount_target_a", "y__amount_target_b"],
        },
        bundle,
    )

    acq_dir = acquire_global_chemistry_candidates(
        db=db,
        out=tmp_path / "ranked_acq",
        candidate_table=candidate,
        model_bundle=bundle,
        max_candidates=1,
        priority_targets_from_diagnostics=diagnostics,
        priority_target_limit=1,
        target_priority_weight=50.0,
    )

    selected = pd.read_csv(acq_dir / "acquisition_candidates.csv")
    summary = json.loads((acq_dir / "acquisition_summary.json").read_text(encoding="utf-8"))
    assert summary["priority_targets"]["requested_targets"] == ["y__amount_target_b"]
    assert summary["priority_targets"]["resolved_targets"] == ["y__amount_target_b"]
    assert summary["priority_target_diagnostics_filter"]["limit"] == 1
    assert selected.iloc[0]["meta__recipe_id"] == "high_b"


def test_global_acquisition_can_prioritize_target_region_neighbors(tmp_path):
    db = tmp_path / "global_db"
    initialize_global_chemistry_db(db=db)
    candidate = tmp_path / "candidate.csv"
    pd.DataFrame(
        [
            _chem_row("near_region", "h_near", 2.05, 1.02, 60.0, 40.0, 28.0),
            _chem_row("far_region", "h_far", 20.0, 10.0, 60.0, 40.0, 28.0),
        ]
    ).to_csv(candidate, index=False)
    target_region = tmp_path / "target_region_nonzero_rows.csv"
    pd.DataFrame([_chem_row("known_nonzero", "h_known", 2.0, 1.0, 60.0, 40.0, 28.0)]).to_csv(
        target_region,
        index=False,
    )

    acq_dir = acquire_global_chemistry_candidates(
        db=db,
        out=tmp_path / "region_acq",
        candidate_table=candidate,
        target_region_table=target_region,
        target_region_weight=100.0,
        target_region_distance_scale=0.25,
        max_candidates=1,
    )

    selected = pd.read_csv(acq_dir / "acquisition_candidates.csv")
    summary = json.loads((acq_dir / "acquisition_summary.json").read_text(encoding="utf-8"))
    assert selected.iloc[0]["meta__recipe_id"] == "near_region"
    assert selected.iloc[0]["target_region_nearest_recipe_id"] == "known_nonzero"
    assert selected.iloc[0]["target_region_score"] > 0.9
    assert "target-region score" in selected.iloc[0]["acquisition_reason"]
    assert summary["target_region"]["enabled"] is True
    assert summary["target_region"]["reference_rows"] == 1


def test_import_cached_db_into_global_db_merges_sqlite_rows(tmp_path):
    source = InverseGemsDatabase(tmp_path / "source")
    source.upsert_chemistry_run(
        {
            "chem_hash": "chem_a",
            "chem_hash_version": 1,
            "created_at": "now",
            "status": "complete",
            "temperature_celsius": 20.0,
            "water_mol": 1.0,
            "canonical_vector_json": "{}",
            "oxide_equivalent_vector_json": "{}",
            "xgems_run_dir": str(tmp_path / "source" / "chemistry_runs" / "chem_a"),
            "warnings_json": "{}",
        }
    )
    source.upsert_prepared_chemistry_run(
        {
            "prepared_id": "prep_a",
            "recipe_id": "recipe_a",
            "chem_hash": "chem_a",
            "created_at": "now",
            "reaction_model_id": "local",
            "reaction_model_signature": "sig",
            "water_mol": 1.0,
            "temperature_celsius": 20.0,
        }
    )
    source.insert_recipe_run(
        {
            "recipe_id": "recipe_a",
            "chem_hash": "chem_a",
            "prepared_id": "prep_a",
            "created_at": "now",
            "reaction_model_id": "local",
            "reaction_model_signature": "sig",
            "material_system": "OPC",
            "recipe_json": '{"binder_masses_g":{"OPC":100},"water_g":40,"w_b":0.4,"age_days":28}',
            "water_g": 40.0,
            "w_b": 0.4,
            "age_days": 28.0,
            "temperature_celsius": 20.0,
            "porosity": 0.3,
        }
    )
    source.insert_source_contributions(
        [
            {
                "recipe_id": "recipe_a",
                "chem_hash": "chem_a",
                "source_material": "OPC",
                "source_phase_or_oxide": "C3S",
                "source_mass_g_initial": 50.0,
                "reaction_degree": 0.5,
                "reacted_mass_g": 25.0,
                "unreacted_mass_g": 25.0,
                "component": "Ca",
                "component_mol": 1.0,
            }
        ]
    )

    manifest = import_cached_db_into_global_db(db=tmp_path / "target", source_db=tmp_path / "source", copy_run_dirs=False)
    target = InverseGemsDatabase(tmp_path / "target")
    assert manifest["imported_rows"]["chemistry_runs"] == 1
    assert target.get_chemistry_run("chem_a")["status"] == "complete"
    assert target.get_recipe_run("recipe_a")["chem_hash"] == "chem_a"
    assert len(target.source_rows_for_recipe("recipe_a")) == 1
