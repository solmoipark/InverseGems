from inverse_gems.reaction_model import current_reaction_model_metadata


def test_reaction_model_metadata_has_stable_signature_for_same_inputs():
    first = current_reaction_model_metadata()
    second = current_reaction_model_metadata()

    assert first["reaction_model_id"]
    assert first["reaction_model_signature"] == second["reaction_model_signature"]
    assert first["reaction_model_signature_version"] == 1
    assert "file_hashes" in first["reaction_model_payload"]
