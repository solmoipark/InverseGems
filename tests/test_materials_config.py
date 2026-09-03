import csv
import json

import yaml

from inverse_gems.api import run_forward_request
from inverse_gems.materials import load_materials
from inverse_gems.utils import config_path


def _forward_query_yaml(path):
    path.write_text(
        yaml.safe_dump(
            {
                "name": "materials_config_forward",
                "recipe": {"binders": {"OPC": 60, "slag": 40}, "w_b": 0.45},
                "age_grid": {"values": [28.0]},
                "plots": [],
                "response_summary": {"phases": ["Mock C-S-H raw phase"], "scalars": ["pH", "porosity"]},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _custom_materials_yaml(path, slag_cao):
    data = yaml.safe_load(config_path("materials.yaml").read_text(encoding="utf-8"))
    data["slag"]["oxide_mass_percent"]["CaO"] = slag_cao
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _slag_cao_initial_mass(db_dir):
    ledgers = sorted((db_dir / "recipe_runs").glob("*/source_contribution_ledger.csv"))
    assert ledgers, "expected at least one recipe run ledger"
    masses = set()
    for ledger in ledgers:
        with ledger.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["source_material"] == "slag" and row["source_phase_or_oxide"] == "CaO":
                    masses.add(float(row["source_mass_g_initial"]))
    assert len(masses) == 1, masses
    return masses.pop()


def test_run_forward_request_threads_materials_config(tmp_path):
    default_cao = load_materials()["slag"].oxide_mass_percent["CaO"]
    edited_cao = round(default_cao + 5.0, 2)
    custom = _custom_materials_yaml(tmp_path / "materials_custom.yaml", edited_cao)
    assert load_materials(custom)["slag"].oxide_mass_percent["CaO"] == edited_cao

    query = _forward_query_yaml(tmp_path / "forward.yaml")
    default_db = tmp_path / "db_default"
    custom_db = tmp_path / "db_custom"

    baseline = run_forward_request(forward_query=query, out=tmp_path / "run_default", db=default_db, use_mock=True)
    result = run_forward_request(
        forward_query=query,
        out=tmp_path / "run_custom",
        db=custom_db,
        use_mock=True,
        materials_config=custom,
    )
    assert baseline.status == "complete"
    assert result.status == "complete"

    default_recipes = sorted((default_db / "recipe_runs").glob("*/recipe.json"))
    custom_recipes = sorted((custom_db / "recipe_runs").glob("*/recipe.json"))
    assert default_recipes and custom_recipes
    for path in default_recipes:
        assert json.loads(path.read_text(encoding="utf-8"))["metadata"]["materials_config"] is None
    for path in custom_recipes:
        assert json.loads(path.read_text(encoding="utf-8"))["metadata"]["materials_config"] == str(custom)

    default_mass = _slag_cao_initial_mass(default_db)
    custom_mass = _slag_cao_initial_mass(custom_db)
    assert custom_mass > default_mass
