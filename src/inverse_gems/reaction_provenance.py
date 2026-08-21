from __future__ import annotations

from typing import Any

import pandas as pd


REACTION_METADATA_COLUMNS = ["prepared_id", "reaction_model_id", "reaction_model_signature"]


def reaction_metadata_columns(prefix: str = "") -> list[str]:
    return [f"{prefix}{column}" for column in REACTION_METADATA_COLUMNS]


def ensure_reaction_metadata_columns(columns: list[str]) -> list[str]:
    out = list(columns)
    for column in REACTION_METADATA_COLUMNS:
        if column not in out:
            out.append(column)
    return out


def _string_values(frame: pd.DataFrame, column: str, *, limit: int = 100) -> list[str]:
    if column not in frame.columns:
        return []
    values = frame[column].dropna().astype(str)
    values = values[values.str.len() > 0]
    values = values[~values.str.lower().isin({"nan", "none", "null"})]
    return sorted(values.unique().tolist())[:limit]


def _count_unique(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0
    values = frame[column].dropna().astype(str)
    values = values[values.str.len() > 0]
    values = values[~values.str.lower().isin({"nan", "none", "null"})]
    return int(values.nunique())


def reaction_provenance_from_frame(frame: pd.DataFrame, *, prefix: str = "") -> dict[str, Any]:
    prepared_column = f"{prefix}prepared_id"
    model_id_column = f"{prefix}reaction_model_id"
    signature_column = f"{prefix}reaction_model_signature"
    signatures = _string_values(frame, signature_column)
    model_ids = _string_values(frame, model_id_column)
    return {
        "row_count": int(len(frame)),
        "columns": {
            "prepared_id": prepared_column if prepared_column in frame.columns else None,
            "reaction_model_id": model_id_column if model_id_column in frame.columns else None,
            "reaction_model_signature": signature_column if signature_column in frame.columns else None,
        },
        "prepared_id_unique_count": _count_unique(frame, prepared_column),
        "reaction_model_ids": model_ids,
        "reaction_model_signatures": signatures,
        "reaction_model_id_count": len(model_ids),
        "reaction_model_signature_count": len(signatures),
        "is_single_reaction_model_signature": len(signatures) == 1,
    }


def _list_from_spec(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def expected_reaction_provenance_from_query(query: dict[str, Any]) -> dict[str, Any]:
    reaction = query.get("reaction_model") or {}
    constraints = (query.get("constraints") or {}).get("metadata") or {}
    signature_spec = constraints.get("reaction_model_signature") or constraints.get("meta__reaction_model_signature") or {}
    id_spec = constraints.get("reaction_model_id") or constraints.get("meta__reaction_model_id") or {}
    signatures = []
    signatures.extend(_list_from_spec(query.get("reaction_model_signature")))
    signatures.extend(_list_from_spec(query.get("reaction_model_signatures")))
    signatures.extend(_list_from_spec(reaction.get("signature")))
    signatures.extend(_list_from_spec(reaction.get("signatures")))
    if isinstance(signature_spec, dict):
        signatures.extend(_list_from_spec(signature_spec.get("include")))
        signatures.extend(_list_from_spec(signature_spec.get("equals")))
    else:
        signatures.extend(_list_from_spec(signature_spec))

    ids = []
    ids.extend(_list_from_spec(query.get("reaction_model_id")))
    ids.extend(_list_from_spec(query.get("reaction_model_ids")))
    ids.extend(_list_from_spec(reaction.get("id")))
    ids.extend(_list_from_spec(reaction.get("ids")))
    if isinstance(id_spec, dict):
        ids.extend(_list_from_spec(id_spec.get("include")))
        ids.extend(_list_from_spec(id_spec.get("equals")))
    else:
        ids.extend(_list_from_spec(id_spec))
    return {
        "reaction_model_ids": sorted(set(ids)),
        "reaction_model_signatures": sorted(set(signatures)),
        "mismatch_policy": str(reaction.get("mismatch_policy") or query.get("reaction_model_mismatch_policy") or "error"),
    }


def check_reaction_provenance_compatibility(
    *,
    table: dict[str, Any],
    bundle: dict[str, Any] | None = None,
    expected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bundle = bundle or {}
    expected = expected or {}
    warnings: list[str] = []
    errors: list[str] = []
    table_signatures = set(table.get("reaction_model_signatures") or [])
    bundle_signatures = set(bundle.get("reaction_model_signatures") or [])
    expected_signatures = set(expected.get("reaction_model_signatures") or [])
    table_ids = set(table.get("reaction_model_ids") or [])
    bundle_ids = set(bundle.get("reaction_model_ids") or [])
    expected_ids = set(expected.get("reaction_model_ids") or [])

    if not table_signatures:
        warnings.append("Model table has no reaction_model_signature metadata; compatibility cannot be verified.")
    if bundle and not bundle_signatures:
        warnings.append("Surrogate bundle has no reaction_model_signature metadata; compatibility cannot be verified.")
    if table_signatures and bundle_signatures and table_signatures != bundle_signatures:
        errors.append(
            "Surrogate bundle reaction_model_signatures do not match model table: "
            f"bundle={sorted(bundle_signatures)}, table={sorted(table_signatures)}"
        )
    if table_ids and bundle_ids and table_ids != bundle_ids:
        warnings.append(
            "Surrogate bundle reaction_model_ids differ from model table: "
            f"bundle={sorted(bundle_ids)}, table={sorted(table_ids)}"
        )
    if expected_signatures and table_signatures and not expected_signatures.issubset(table_signatures):
        errors.append(
            "Expected reaction_model_signatures are not present in model table: "
            f"expected={sorted(expected_signatures)}, table={sorted(table_signatures)}"
        )
    if expected_signatures and bundle_signatures and not expected_signatures.issubset(bundle_signatures):
        errors.append(
            "Expected reaction_model_signatures are not present in surrogate bundle: "
            f"expected={sorted(expected_signatures)}, bundle={sorted(bundle_signatures)}"
        )
    if expected_ids and table_ids and not expected_ids.issubset(table_ids):
        errors.append(f"Expected reaction_model_ids are not present in model table: expected={sorted(expected_ids)}, table={sorted(table_ids)}")
    if expected_ids and bundle_ids and not expected_ids.issubset(bundle_ids):
        errors.append(f"Expected reaction_model_ids are not present in surrogate bundle: expected={sorted(expected_ids)}, bundle={sorted(bundle_ids)}")

    policy = str(expected.get("mismatch_policy") or "error").lower()
    ok = not errors or policy in {"warn", "warning", "ignore", "off", "false"}
    if errors and policy in {"warn", "warning"}:
        warnings.extend(errors)
        errors = []
    if policy in {"ignore", "off", "false"}:
        warnings.extend(errors)
        errors = []
        ok = True
    return {"ok": ok, "warnings": warnings, "errors": errors, "policy": policy}
