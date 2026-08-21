from inverse_gems.sampling import generate_recipe_rows
from inverse_gems.utils import load_yaml


def test_sampling_bounds_seed_and_age_metadata():
    rows_a = generate_recipe_rows(config_path="configs/sampling.yaml", n=5, mode="mixed", age_preset="early_dense_v1", seed=42)
    rows_b = generate_recipe_rows(config_path="configs/sampling.yaml", n=5, mode="mixed", age_preset="early_dense_v1", seed=42)
    assert rows_a == rows_b
    assert len(rows_a) == 5 * 28
    for row in rows_a:
        total = sum(float(row[k]) for k in ["OPC", "slag", "fly_ash", "metakaolin", "silica_fume", "limestone", "gypsum"])
        assert abs(total - 100.0) < 1.0e-8
        assert row["template_name"]
        assert row["age_bin"]
        assert float(row["water_g"]) > 0


def test_lc3_ratio_constraint_when_generated():
    rows = generate_recipe_rows(config_path="configs/sampling.yaml", n=60, mode="recipe_templates", age_preset="standard_sparse", seed=7)
    lc3 = [row for row in rows if row["template_name"] == "LC3_like"]
    assert lc3
    for row in lc3:
        ratio = float(row["metakaolin"]) / float(row["limestone"])
        assert 1.2 <= ratio <= 3.5


def test_material_system_generation_constrains_components_and_explicit_age():
    rows = generate_recipe_rows(
        config_path="configs/sampling.yaml",
        n=8,
        mode="mixed",
        ages="28",
        seed=11,
        material_system="OPC_slag",
        recipe_id_prefix="OPC_slag_age28",
    )
    assert len(rows) == 8
    for row in rows:
        assert str(row["recipe_id"]).startswith("OPC_slag_age28_")
        assert row["template_name"] == "OPC_slag"
        assert row["material_system"] == "OPC_slag"
        assert float(row["age_days"]) == 28.0
        assert 30.0 <= float(row["OPC"]) <= 85.0
        assert 15.0 <= float(row["slag"]) <= 70.0
        assert 0.0 <= float(row["gypsum"]) <= 8.0
        assert 0.30 <= float(row["w_b"]) <= 0.55
        assert float(row["fly_ash"]) == 0.0
        assert float(row["metakaolin"]) == 0.0
        assert float(row["silica_fume"]) == 0.0
        assert float(row["limestone"]) == 0.0
        total = sum(float(row[k]) for k in ["OPC", "slag", "fly_ash", "metakaolin", "silica_fume", "limestone", "gypsum"])
        assert abs(total - 100.0) < 1.0e-8


def test_strict_material_system_generation_excludes_optional_gypsum():
    rows = generate_recipe_rows(
        config_path="configs/sampling.yaml",
        n=8,
        mode="mixed",
        ages="56",
        seed=17,
        material_system="OPC_slag_fly_ash",
        strict_materials=True,
        recipe_id_prefix="strict_osf",
    )
    assert len(rows) == 8
    for row in rows:
        assert row["material_system"] == "OPC_slag_fly_ash"
        assert float(row["OPC"]) > 0.0
        assert float(row["slag"]) > 0.0
        assert float(row["fly_ash"]) > 0.0
        assert float(row["gypsum"]) == 0.0
        assert float(row["limestone"]) == 0.0
        assert float(row["metakaolin"]) == 0.0
        assert float(row["silica_fume"]) == 0.0
        total = sum(float(row[k]) for k in ["OPC", "slag", "fly_ash", "metakaolin", "silica_fume", "limestone", "gypsum"])
        assert abs(total - 100.0) < 1.0e-8


def test_all_material_system_profiles_generate_and_respect_fixed_zero():
    systems = load_yaml("configs/material_systems.yaml")
    for name, profile in systems.items():
        rows = generate_recipe_rows(
            config_path="configs/sampling.yaml",
            n=3,
            mode="mixed",
            ages="28",
            seed=13,
            material_system=name,
            recipe_id_prefix=name,
        )
        assert len(rows) == 3
        for row in rows:
            assert row["material_system"] == name
            assert row["template_name"] == name
            for component in profile.get("fixed_zero", []):
                assert float(row[component]) == 0.0
            total = sum(float(row[k]) for k in ["OPC", "slag", "fly_ash", "metakaolin", "silica_fume", "limestone", "gypsum"])
            assert abs(total - 100.0) < 1.0e-8


def test_continuous_age_mixed_material_system_sampling():
    rows = generate_recipe_rows(
        config_path="configs/sampling.yaml",
        n=6,
        mode="mixed",
        seed=21,
        material_systems=["OPC_only", "OPC_slag", "OPC_fly_ash"],
        strict_materials=True,
        age_sampling="log_uniform",
        age_min=0.1,
        age_max=365.0,
        age_count=2,
        recipe_id_prefix="global",
    )
    assert len(rows) == 12
    systems = {row["material_system"] for row in rows}
    assert systems.issubset({"OPC_only", "OPC_slag", "OPC_fly_ash"})
    assert len(systems) >= 2
    for row in rows:
        assert row["age_sampling_mode"] == "log_uniform"
        assert row["material_system_sampling_mode"] == "mixed"
        assert 0.1 <= float(row["age_days"]) <= 365.0
        assert float(row["gypsum"]) == 0.0
        total = sum(float(row[k]) for k in ["OPC", "slag", "fly_ash", "metakaolin", "silica_fume", "limestone", "gypsum"])
        assert abs(total - 100.0) < 1.0e-8


def test_balanced_mixed_material_system_sampling_covers_each_system_evenly():
    rows = generate_recipe_rows(
        config_path="configs/sampling.yaml",
        n=6,
        mode="mixed",
        seed=21,
        material_systems=["OPC_only", "OPC_slag", "OPC_fly_ash"],
        material_systems_sampling="balanced",
        ages="28",
        recipe_id_prefix="balanced_global",
    )
    counts = {}
    for row in rows:
        counts[row["material_system"]] = counts.get(row["material_system"], 0) + 1
        assert row["material_systems_sampling"] == "balanced"
    assert counts == {"OPC_fly_ash": 2, "OPC_only": 2, "OPC_slag": 2}


def test_target_profile_generates_biased_rare_phase_candidates():
    rows = generate_recipe_rows(
        config_path="configs/sampling.yaml",
        n=9,
        mode="mixed",
        seed=33,
        target_profile="hemicarbonate",
        recipe_id_prefix="hemi_target",
    )
    assert len(rows) == 18
    systems = {row["material_system"] for row in rows}
    assert systems == {"OPC_fly_ash_limestone", "OPC_slag_limestone", "quaternary_low_clinker"}
    for row in rows:
        assert row["target_profile"] == "hemicarbonate"
        assert row["material_systems_sampling"] == "balanced"
        assert 7.0 <= float(row["age_days"]) <= 365.0
        assert 0.0 <= float(row["limestone"]) <= 3.0
        assert 0.0 <= float(row["gypsum"]) <= 6.0
        assert 0.32 <= float(row["w_b"]) <= 0.55
        total = sum(float(row[k]) for k in ["OPC", "slag", "fly_ash", "metakaolin", "silica_fume", "limestone", "gypsum"])
        assert abs(total - 100.0) < 1.0e-8


def test_target_profile_respects_explicit_material_system_subset():
    rows = generate_recipe_rows(
        config_path="configs/sampling.yaml",
        n=4,
        mode="mixed",
        seed=34,
        target_profile="carbonate_afm",
        material_systems=["LC3_like"],
        ages="28",
        recipe_id_prefix="lc3_carbonate_target",
    )
    assert len(rows) == 4
    for row in rows:
        assert row["material_system"] == "LC3_like"
        assert row["target_profile"] == "carbonate_afm"
        assert 10.0 <= float(row["limestone"]) <= 25.0
        assert 18.0 <= float(row["metakaolin"]) <= 40.0


def test_aluminosilicate_gel_target_profile_focuses_early_mk_systems():
    rows = generate_recipe_rows(
        config_path="configs/sampling.yaml",
        n=6,
        mode="mixed",
        seed=35,
        target_profile="aluminosilicate_gel",
        recipe_id_prefix="al_gel_target",
    )
    assert len(rows) == 12
    systems = {row["material_system"] for row in rows}
    assert systems == {"LC3_like", "OPC_metakaolin"}
    for row in rows:
        assert row["target_profile"] == "aluminosilicate_gel"
        assert 0.1 <= float(row["age_days"]) <= 3.0
        assert 20.0 <= float(row["metakaolin"]) <= 40.0
        assert 3.0 <= float(row["gypsum"]) <= 8.0
        assert 0.30 <= float(row["w_b"]) <= 0.55
        if row["material_system"] == "LC3_like":
            assert 8.0 <= float(row["limestone"]) <= 24.0
        else:
            assert float(row["limestone"]) == 0.0
