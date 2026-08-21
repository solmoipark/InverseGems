import json

from inverse_gems.cli import run_forward_recipe
from inverse_gems.xgems_output_capture import EXPECTED_OUTPUT_FILES


def test_forward_mock_produces_complete_run_directory(tmp_path):
    run_dir = run_forward_recipe(
        recipe_text="OPC 30, fly ash 70, w/b 0.4, age 28",
        out=tmp_path,
        use_mock=True,
        command_name="forward-mock",
    )
    for filename in EXPECTED_OUTPUT_FILES:
        assert (run_dir / filename).exists()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    provenance = json.loads((run_dir / "run_provenance.json").read_text(encoding="utf-8"))
    assert manifest["reaction_model_signature"]
    assert provenance["reaction_model"]["reaction_model_signature"] == manifest["reaction_model_signature"]
