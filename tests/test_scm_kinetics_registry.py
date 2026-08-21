import math

import numpy as np
import pytest

from inverse_gems.availability_modifier import C3S_C2SAvailabilityModifier
from inverse_gems.reaction_parameters import load_reaction_parameters
from inverse_gems.scm_reaction import (
    DEFAULT_KINETICS_MODEL,
    SCMKineticsParameters,
    SCMLogisticParameters,
    _KINETICS_REGISTRY,
    register_scm_kinetics,
    registered_scm_kinetics_models,
    scm_alpha,
    scm_kinetics_from_mapping,
)


@pytest.fixture()
def exponential_model():
    """Temporarily register a simple alternative kinetics model."""
    name = "test_exponential_saturation"

    @register_scm_kinetics(name, required=("alpha_max", "k"), asymptote_key="alpha_max")
    def _exponential(t, p):
        return p["alpha_max"] * (1.0 - np.exp(-p["k"] * t))

    yield name
    _KINETICS_REGISTRY.pop(name, None)


def test_default_model_unchanged_flat_mapping_and_to_dict():
    params = scm_kinetics_from_mapping({"A": 0.0, "B": 1.2, "C": 30.0, "D": 0.8, "G": 1.0})
    assert isinstance(params, SCMLogisticParameters)
    assert params.model == DEFAULT_KINETICS_MODEL
    # legacy provenance format stays flat (hash/signature stability)
    assert params.to_dict() == {"A": 0.0, "B": 1.2, "C": 30.0, "D": 0.8, "G": 1.0}


def test_registered_model_evaluates_and_clips(exponential_model):
    params = scm_kinetics_from_mapping({"model": exponential_model, "alpha_max": 0.7, "k": 0.1})
    assert isinstance(params, SCMKineticsParameters)
    alpha_28 = scm_alpha(28.0, params)
    assert alpha_28 == pytest.approx(0.7 * (1.0 - math.exp(-2.8)))
    # asymptote respected and clipped to [0, 1]
    assert scm_alpha(10000.0, params) == pytest.approx(0.7, abs=1e-6)

    series = scm_alpha([1.0, 7.0, 28.0], params)
    assert series == sorted(series)


def test_generic_model_to_dict_records_model_name(exponential_model):
    params = scm_kinetics_from_mapping({"model": exponential_model, "alpha_max": 0.7, "k": 0.1})
    provenance = params.to_dict()
    assert provenance["model"] == exponential_model
    assert provenance["alpha_max"] == 0.7


def test_unknown_model_and_missing_params_fail_loudly(exponential_model):
    with pytest.raises(KeyError):
        scm_kinetics_from_mapping({"model": "no_such_model", "x": 1.0})
    with pytest.raises(ValueError):
        scm_kinetics_from_mapping({"model": exponential_model, "alpha_max": 0.7})


def test_availability_modifier_caps_generic_model_asymptote(exponential_model):
    slag = scm_kinetics_from_mapping({"model": exponential_model, "alpha_max": 0.9, "k": 0.05})
    base = {"slag": slag}
    modifier = C3S_C2SAvailabilityModifier()
    modification = modifier.modify_parameters(
        binder_masses_g={"OPC": 20.0, "slag": 80.0},
        opc_phase_mass_percent={"C3S": 60.0, "C2S": 15.0},
        scm_params=base,
    )
    assert modification.metadata["applied"] is True
    effective = modification.parameters["slag"]
    # low clinker -> asymptote capped below the configured alpha_max
    assert effective.D < 0.9
    assert effective.values["alpha_max"] == effective.D


def test_reaction_parameter_config_can_switch_models(tmp_path, exponential_model):
    config = tmp_path / "reaction.yaml"
    config.write_text(
        "id: exp_trial_v0\n"
        "scm_reaction:\n"
        "  slag:\n"
        f"    model: {exponential_model}\n"
        "    alpha_max: 0.75\n"
        "    k: 0.08\n",
        encoding="utf-8",
    )
    params = load_reaction_parameters(config)
    slag = params.scm_parameters["slag"]
    assert isinstance(slag, SCMKineticsParameters)
    assert slag.model == exponential_model
    # other SCMs keep the default logistic form
    assert isinstance(params.scm_parameters["fly_ash"], SCMLogisticParameters)
    assert params.to_dict()["scm_reaction"]["slag"]["model"] == exponential_model


def test_registry_lists_default_model():
    assert DEFAULT_KINETICS_MODEL in registered_scm_kinetics_models()
