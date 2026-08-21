import shutil
from pathlib import Path

from inverse_gems.global_chemistry_db import initialize_global_chemistry_db, load_global_manifest


def test_copied_global_db_manifest_rebases_paths(tmp_path):
    original = tmp_path / "db_a"
    initialize_global_chemistry_db(db=original)
    manifest = load_global_manifest(original)
    assert str(original) in manifest["paths"]["model_table"]

    copied = tmp_path / "db_b"
    shutil.copytree(original, copied)

    rebased = load_global_manifest(copied)
    for key, value in rebased["paths"].items():
        assert str(copied) in value, f"path {key!r} still points outside the copied DB: {value}"
    assert rebased["rebased_from"] == str(original)
    assert str(copied) in rebased["sqlite_path"]

    # the fix is persisted: a second load needs no rebase and keeps new paths
    again = load_global_manifest(copied)
    assert again["paths"] == rebased["paths"]
    # the original DB manifest is untouched
    assert str(original) in load_global_manifest(original)["paths"]["model_table"]
