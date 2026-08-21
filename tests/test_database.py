import json

from inverse_gems.cached_forward import run_forward_cached
from inverse_gems.database import InverseGemsDatabase


def test_two_recipe_runs_can_share_one_chem_hash(tmp_path):
    db_dir = tmp_path / "db"
    first = run_forward_cached(recipe_text="OPC 100, w/b 0.4, age 28", db=db_dir, use_mock=True)
    second = run_forward_cached(recipe_text="OPC 100, w/b 0.4, age 28", db=db_dir, use_mock=True)
    db = InverseGemsDatabase(db_dir)
    assert first["chem_hash"] == second["chem_hash"]
    assert first["recipe_id"] != second["recipe_id"]
    assert len(db.linked_recipe_ids(first["chem_hash"])) == 2


def test_prepared_chemistry_projection_links_recipes_to_one_chemistry(tmp_path):
    db_dir = tmp_path / "db"
    first = run_forward_cached(recipe_text="OPC 30, fly ash 70, w/b 0.4, age 28", db=db_dir, use_mock=True)
    second = run_forward_cached(recipe_text="OPC 30, fly ash 70, w/b 0.4, age 28", db=db_dir, use_mock=True)
    db = InverseGemsDatabase(db_dir)

    assert first["chem_hash"] == second["chem_hash"]
    assert first["prepared_id"] == second["prepared_id"]
    assert first["reaction_model_signature"] == second["reaction_model_signature"]

    prepared_rows = db.prepared_rows()
    assert len(prepared_rows) == 1
    prepared = db.get_prepared_chemistry_run(first["prepared_id"])
    assert prepared is not None
    assert prepared["chem_hash"] == first["chem_hash"]
    assert prepared["reaction_model_id"] == first["reaction_model_id"]
    assert prepared["reaction_model_signature"] == first["reaction_model_signature"]
    assert "H2O@" in prepared["xgems_species_amounts_json"]

    first_recipe = db.get_recipe_run(first["recipe_id"])
    second_recipe = db.get_recipe_run(second["recipe_id"])
    assert first_recipe["prepared_id"] == first["prepared_id"]
    assert second_recipe["prepared_id"] == first["prepared_id"]
    assert (db.prepared_chemistry_runs_dir / first["prepared_id"] / "prepared_chemistry.json").exists()


def test_cached_forward_writes_parameter_provenance_to_db_and_run_dirs(tmp_path):
    db_dir = tmp_path / "db"
    result = run_forward_cached(
        recipe_text="OPC 30, metakaolin 70, w/b 0.4, age 28",
        db=db_dir,
        use_mock=True,
        recipe_metadata={"target_profile": "hemi_test", "target_profile_description": "test profile"},
    )
    db = InverseGemsDatabase(db_dir)

    recipe_row = db.get_recipe_run(result["recipe_id"])
    assert recipe_row["target_profile"] == "hemi_test"
    assert recipe_row["target_profile_description"] == "test profile"
    assert recipe_row["reaction_model_signature_version"] == result["reaction_model_signature_version"]
    assert recipe_row["reaction_parameter_set_id"] == result["reaction_parameter_set_id"]
    assert recipe_row["reaction_parameter_config_hash"] == result["reaction_parameter_config_hash"]
    assert json.loads(recipe_row["reaction_model_json"])["reaction_model_signature_version"] == result["reaction_model_signature_version"]

    recipe_provenance_path = db.recipe_runs_dir / result["recipe_id"] / "run_provenance.json"
    prepared_provenance_path = db.prepared_chemistry_runs_dir / result["prepared_id"] / "run_provenance.json"
    chemistry_provenance_path = db.chemistry_runs_dir / result["chem_hash"] / "chemistry_provenance.json"
    for path in [recipe_provenance_path, prepared_provenance_path, chemistry_provenance_path]:
        assert path.exists()

    run_provenance = json.loads(recipe_provenance_path.read_text(encoding="utf-8"))
    chemistry_provenance = json.loads(chemistry_provenance_path.read_text(encoding="utf-8"))
    assert run_provenance["reaction_model"]["reaction_model_signature"] == result["reaction_model_signature"]
    assert run_provenance["reaction_parameter_set"]["id"] == result["reaction_parameter_set_id"]
    assert chemistry_provenance["chem_hash"] == result["chem_hash"]
    assert chemistry_provenance["species_map_hash"]


def test_database_backfills_target_profile_from_recipe_json(tmp_path):
    db_dir = tmp_path / "db"
    result = run_forward_cached(
        recipe_text="OPC 30, metakaolin 70, w/b 0.4, age 28",
        db=db_dir,
        use_mock=True,
        recipe_metadata={"target_profile": "hemi_test"},
    )
    db = InverseGemsDatabase(db_dir)
    with db.connect() as conn:
        conn.execute("UPDATE recipe_runs SET target_profile = NULL WHERE recipe_id = ?", (result["recipe_id"],))

    refreshed = InverseGemsDatabase(db_dir)
    recipe_row = refreshed.get_recipe_run(result["recipe_id"])
    assert recipe_row["target_profile"] == "hemi_test"
