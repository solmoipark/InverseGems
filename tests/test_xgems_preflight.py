import json

import pandas as pd

from inverse_gems.cli import main
from inverse_gems.xgems_preflight import run_xgems_input_preflight


def test_xgems_input_preflight_writes_formula_report(tmp_path):
    dat = tmp_path / "Test-dat.lst"
    dat.write_text("dummy\n", encoding="utf-8")
    out = tmp_path / "preflight"

    report = run_xgems_input_preflight(
        recipe_text="OPC 30, fly ash 70, w/b 0.4, age 28",
        dat_lst=dat,
        out=out,
        xgems_input_mode="formula",
    )

    assert report["ok"] is True
    assert report["dat_lst"]["ok"] is True
    assert "Fe" in report["input_compatibility"]["used_elements"]
    assert report["water"]["matches_recipe_water"] is True
    assert (out / "xgems_input_preflight.json").exists()
    assert (out / "xgems_input_preflight.md").exists()
    amounts = pd.read_csv(out / "xgems_input_amounts.csv")
    assert {"input_name", "amount_kg", "amount_g"}.issubset(amounts.columns)
    assert "H2O@" in set(amounts["input_name"])


def test_xgems_input_preflight_flags_adjusted_water_and_ph_uncertainty(tmp_path):
    dat = tmp_path / "Test-dat.lst"
    dat.write_text("dummy\n", encoding="utf-8")

    report = run_xgems_input_preflight(
        recipe_text="OPC 100, w/b 0.6, age 28",
        dat_lst=dat,
        out=tmp_path / "preflight_adjusted_water",
        xgems_input_mode="formula",
        xgems_water_mode="fixed_w_b",
        xgems_water_w_b=0.4,
    )

    assert report["ok"] is True
    assert report["water"]["matches_recipe_water"] is False
    assert "xgems_water_adjusted" in report["flags"]
    assert "pH_uncertain_if_run" in report["flags"]


def test_preflight_xgems_input_cli(tmp_path):
    dat = tmp_path / "Test-dat.lst"
    out = tmp_path / "preflight_cli"
    dat.write_text("dummy\n", encoding="utf-8")

    code = main(
        [
            "preflight-xgems-input",
            "--dat-lst",
            str(dat),
            "--recipe",
            "OPC 40, metakaolin 60, w/b 0.45, age 28",
            "--out",
            str(out),
        ]
    )

    assert code == 0
    payload = json.loads((out / "xgems_input_preflight.json").read_text(encoding="utf-8"))
    assert payload["recipe"]["binder_masses_g"]["metakaolin"] == 60
    assert payload["input_compatibility"]["input_mode"] == "formula"
