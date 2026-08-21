import json

from inverse_gems.cli import main
from inverse_gems.env_check import check_environment


def test_check_environment_core_with_dat_lst(tmp_path):
    dat = tmp_path / "Test-dat.lst"
    dat.write_text("dummy\n", encoding="utf-8")

    report = check_environment(dat_lst=dat, require_xgems=False)

    assert report["ok"] is True
    assert report["dat_lst"]["ok"] is True
    assert report["imports"]["pandas"]["ok"] is True
    assert "xgems" in report


def test_check_environment_missing_dat_lst_is_not_ok(tmp_path):
    report = check_environment(dat_lst=tmp_path / "missing-dat.lst", require_xgems=False)

    assert report["ok"] is False
    assert report["dat_lst"]["ok"] is False


def test_check_env_cli_writes_report(tmp_path):
    dat = tmp_path / "Test-dat.lst"
    out = tmp_path / "env_report.json"
    dat.write_text("dummy\n", encoding="utf-8")

    code = main(["check-env", "--dat-lst", str(dat), "--out", str(out)])

    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["dat_lst"]["path"] == str(dat)


def test_check_env_cli_returns_nonzero_for_missing_dat(tmp_path):
    code = main(["check-env", "--dat-lst", str(tmp_path / "missing-dat.lst")])

    assert code == 1
