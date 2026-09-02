"""OPC minor oxides (SO3 as CaSO4, MgO, Na2O, K2O) reach the xGEMS input.

Before this policy the OPC was reduced to its Bogue phases only, so the system had no
sulfur (no ettringite/monosulfate) and no alkalis (pH pinned at the portlandite buffer).
"""

from __future__ import annotations

import pytest

from inverse_gems.materials import load_materials
from inverse_gems.reaction_parameters import DEFAULT_OPC_MINOR_OXIDES, ReactionParameterSet, load_reaction_parameters
from inverse_gems.recipe import parse_recipe
from inverse_gems.xgems_input_builder import build_xgems_input


def _inputs(enabled: bool):
    materials = load_materials()
    recipe = parse_recipe("OPC 100, w/b 0.5, age 28", materials=materials)
    params = load_reaction_parameters()
    if not enabled:
        params = ReactionParameterSet(**{**params.__dict__, "opc_minor_oxides": {**DEFAULT_OPC_MINOR_OXIDES, "enabled": False}})
    return build_xgems_input(recipe, materials=materials, reaction_parameters=params), materials


def test_minor_oxides_present_by_default():
    xi, materials = _inputs(True)
    species = xi.species_amounts_kg
    for name in ("SO3", "MgO", "Na2O", "K2O", "CaO"):
        assert species.get(name, 0.0) > 0.0, (name, species)
    meta = xi.reaction_degrees["opc_minor_oxides"]
    assert meta["enabled"] and set(meta["per_oxide"]) == {"MgO", "SO3", "Na2O", "K2O"}
    ox = materials["OPC"].oxide_mass_percent
    opc_g = 100.0  # recipe basis
    # SO3, Na2O, K2O fully reacted; MgO follows the clinker mean degree
    assert species["SO3"] == pytest.approx(opc_g * ox["SO3"] / 100.0 * 1.0e-3)
    assert species["K2O"] == pytest.approx(opc_g * ox["K2O"] / 100.0 * 1.0e-3)
    assert 0.0 < meta["per_oxide"]["MgO"]["degree"] < 1.0
    assert meta["per_oxide"]["MgO"]["degree"] == pytest.approx(meta["clinker_mean_degree"])
    # calcium of the calcium sulfate re-enters (56.08/80.06 g CaO per g SO3)
    assert species["CaO"] == pytest.approx(opc_g * ox["SO3"] / 100.0 * 56.0774 / 80.0632 * 1.0e-3)


def test_mass_closure_of_opc():
    xi, materials = _inputs(True)
    reacted_kg = sum(v for k, v in xi.species_amounts_kg.items() if k != "H2O@")
    unreacted_g = xi.unreacted_masses_g["OPC"]
    total_g = reacted_kg * 1000.0 + unreacted_g
    # Bogue phases (92.6 %) + minor oxides (5.8 %) + companion CaO (1.8 %) ≈ 100 g of OPC
    assert 99.0 <= total_g <= 101.0, total_g


def test_disabled_reproduces_legacy_vector():
    xi_on, _ = _inputs(True)
    xi_off, _ = _inputs(False)
    assert "SO3" not in xi_off.species_amounts_kg and "Na2O" not in xi_off.species_amounts_kg
    assert xi_off.reaction_degrees["opc_minor_oxides"] == {"enabled": False}
    for phase in ("C3S", "C2S", "C3A", "C4AF"):
        assert xi_on.species_amounts_kg[phase] == pytest.approx(xi_off.species_amounts_kg[phase])


def test_policy_is_part_of_the_signature_payload():
    params = load_reaction_parameters()
    payload = params.signature_payload()
    assert payload["opc_minor_oxides"]["enabled"] is True


def test_yaml_override(tmp_path):
    cfg = tmp_path / "rp.yaml"
    cfg.write_text("id: t\nopc_minor_oxides: {degrees: {MgO: 0.0, K2O: 0.5}}\n", encoding="utf-8")
    params = load_reaction_parameters(cfg)
    materials = load_materials()
    recipe = parse_recipe("OPC 100, w/b 0.5, age 28", materials=materials)
    xi = build_xgems_input(recipe, materials=materials, reaction_parameters=params)
    meta = xi.reaction_degrees["opc_minor_oxides"]["per_oxide"]
    assert meta["MgO"]["degree"] == 0.0 and meta["K2O"]["degree"] == 0.5
    assert "MgO" not in xi.species_amounts_kg or xi.species_amounts_kg["MgO"] == 0.0
