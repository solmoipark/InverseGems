from __future__ import annotations

import csv
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .phase_volume_reconstruction import write_phase_volume_reconstruction_files
from .utils import to_jsonable, write_json


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class InverseGemsDatabase:
    def __init__(self, db_dir: str | Path = "data/inverse_gems_db") -> None:
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_path = self.db_dir / "inverse_gems.sqlite"
        self.chemistry_runs_dir = self.db_dir / "chemistry_runs"
        self.prepared_chemistry_runs_dir = self.db_dir / "prepared_chemistry_runs"
        self.recipe_runs_dir = self.db_dir / "recipe_runs"
        self.chemistry_runs_dir.mkdir(parents=True, exist_ok=True)
        self.prepared_chemistry_runs_dir.mkdir(parents=True, exist_ok=True)
        self.recipe_runs_dir.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS chemistry_runs (
                  chem_hash TEXT PRIMARY KEY,
                  chem_hash_version INTEGER,
                  created_at TEXT,
                  status TEXT,
                  dat_lst_hash TEXT,
                  species_map_hash TEXT,
                  temperature_celsius REAL,
                  pressure REAL NULL,
                  water_mol REAL,
                  canonical_vector_json TEXT,
                  oxide_equivalent_vector_json TEXT,
                  xgems_run_dir TEXT,
                  warnings_json TEXT
                );

                CREATE TABLE IF NOT EXISTS prepared_chemistry_runs (
                  prepared_id TEXT PRIMARY KEY,
                  recipe_id TEXT,
                  chem_hash TEXT,
                  created_at TEXT,
                  reaction_model_id TEXT,
                  reaction_model_signature TEXT,
                  reaction_model_signature_version INTEGER,
                  reaction_model_json TEXT,
                  recipe_json TEXT,
                  reaction_degrees_json TEXT,
                  xgems_species_amounts_json TEXT,
                  unreacted_masses_json TEXT,
                  canonical_vector_json TEXT,
                  oxide_equivalent_vector_json TEXT,
                  water_mol REAL,
                  temperature_celsius REAL,
                  pressure REAL NULL,
                  xgems_water_policy_json TEXT,
                  source_ledger_hash TEXT,
                  warnings_json TEXT
                );

                CREATE TABLE IF NOT EXISTS recipe_runs (
                  recipe_id TEXT PRIMARY KEY,
                  chem_hash TEXT,
                  prepared_id TEXT,
                  created_at TEXT,
                  reaction_model_id TEXT,
                  reaction_model_signature TEXT,
                  reaction_model_signature_version INTEGER,
                  reaction_parameter_set_id TEXT,
                  reaction_parameter_config_path TEXT,
                  reaction_parameter_config_hash TEXT,
                  reaction_model_json TEXT,
                  template_name TEXT,
                  material_system TEXT,
                  target_profile TEXT,
                  target_profile_description TEXT,
                  recipe_json TEXT,
                  reaction_degrees_json TEXT,
                  initial_masses_json TEXT,
                  reacted_masses_json TEXT,
                  unreacted_masses_json TEXT,
                  water_g REAL,
                  w_b REAL,
                  water_mode TEXT,
                  xgems_water_g REAL,
                  xgems_w_b REAL,
                  xgems_water_mode TEXT,
                  xgems_water_policy_json TEXT,
                  solver_rescued INTEGER,
                  xgems_retry_count INTEGER,
                  primary_chem_hash TEXT,
                  primary_solver_status TEXT,
                  retry_history_json TEXT,
                  age_days REAL,
                  age_hours REAL,
                  age_minutes REAL,
                  age_label TEXT,
                  age_bin TEXT,
                  temperature_celsius REAL,
                  initial_volume_cm3 REAL,
                  final_solid_volume_cm3 REAL,
                  porosity REAL,
                  run_dir TEXT,
                  warnings_json TEXT
                );

                CREATE TABLE IF NOT EXISTS source_contributions (
                  recipe_id TEXT,
                  chem_hash TEXT,
                  source_material TEXT,
                  source_phase_or_oxide TEXT,
                  source_mass_g_initial REAL,
                  reaction_degree REAL,
                  reacted_mass_g REAL,
                  unreacted_mass_g REAL,
                  component TEXT,
                  component_mol REAL,
                  oxide_equivalent TEXT,
                  oxide_equivalent_mol REAL
                );
                """
            )
            self._ensure_column(conn, "recipe_runs", "template_name", "TEXT")
            self._ensure_column(conn, "recipe_runs", "prepared_id", "TEXT")
            self._ensure_column(conn, "recipe_runs", "reaction_model_id", "TEXT")
            self._ensure_column(conn, "recipe_runs", "reaction_model_signature", "TEXT")
            self._ensure_column(conn, "recipe_runs", "reaction_model_signature_version", "INTEGER")
            self._ensure_column(conn, "recipe_runs", "reaction_parameter_set_id", "TEXT")
            self._ensure_column(conn, "recipe_runs", "reaction_parameter_config_path", "TEXT")
            self._ensure_column(conn, "recipe_runs", "reaction_parameter_config_hash", "TEXT")
            self._ensure_column(conn, "recipe_runs", "reaction_model_json", "TEXT")
            self._ensure_column(conn, "recipe_runs", "material_system", "TEXT")
            self._ensure_column(conn, "recipe_runs", "target_profile", "TEXT")
            self._ensure_column(conn, "recipe_runs", "target_profile_description", "TEXT")
            self._ensure_column(conn, "recipe_runs", "water_g", "REAL")
            self._ensure_column(conn, "recipe_runs", "w_b", "REAL")
            self._ensure_column(conn, "recipe_runs", "water_mode", "TEXT")
            self._ensure_column(conn, "recipe_runs", "xgems_water_g", "REAL")
            self._ensure_column(conn, "recipe_runs", "xgems_w_b", "REAL")
            self._ensure_column(conn, "recipe_runs", "xgems_water_mode", "TEXT")
            self._ensure_column(conn, "recipe_runs", "xgems_water_policy_json", "TEXT")
            self._ensure_column(conn, "recipe_runs", "solver_rescued", "INTEGER")
            self._ensure_column(conn, "recipe_runs", "xgems_retry_count", "INTEGER")
            self._ensure_column(conn, "recipe_runs", "primary_chem_hash", "TEXT")
            self._ensure_column(conn, "recipe_runs", "primary_solver_status", "TEXT")
            self._ensure_column(conn, "recipe_runs", "retry_history_json", "TEXT")
            self._ensure_column(conn, "recipe_runs", "age_days", "REAL")
            self._ensure_column(conn, "recipe_runs", "age_hours", "REAL")
            self._ensure_column(conn, "recipe_runs", "age_minutes", "REAL")
            self._ensure_column(conn, "recipe_runs", "age_label", "TEXT")
            self._ensure_column(conn, "recipe_runs", "age_bin", "TEXT")
            self._ensure_column(conn, "recipe_runs", "temperature_celsius", "REAL")
            for column, decl in {
                "reaction_model_signature_version": "INTEGER",
                "reaction_model_json": "TEXT",
                "recipe_json": "TEXT",
                "reaction_degrees_json": "TEXT",
                "xgems_species_amounts_json": "TEXT",
                "unreacted_masses_json": "TEXT",
                "canonical_vector_json": "TEXT",
                "oxide_equivalent_vector_json": "TEXT",
                "xgems_water_policy_json": "TEXT",
                "source_ledger_hash": "TEXT",
                "warnings_json": "TEXT",
            }.items():
                self._ensure_column(conn, "prepared_chemistry_runs", column, decl)
            self._backfill_recipe_metadata_columns(conn)

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str, column: str, decl: str) -> None:
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    @staticmethod
    def _backfill_recipe_metadata_columns(conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            """
            SELECT recipe_id, recipe_json
            FROM recipe_runs
            WHERE (target_profile IS NULL OR target_profile = '')
              AND recipe_json IS NOT NULL
              AND recipe_json != ''
            """
        ).fetchall()
        for row in rows:
            try:
                recipe = json.loads(row["recipe_json"])
            except Exception:
                continue
            metadata = recipe.get("metadata") or {}
            target_profile = metadata.get("target_profile")
            target_profile_description = metadata.get("target_profile_description")
            if target_profile in (None, "") and target_profile_description in (None, ""):
                continue
            conn.execute(
                """
                UPDATE recipe_runs
                SET target_profile = COALESCE(NULLIF(?, ''), target_profile),
                    target_profile_description = COALESCE(NULLIF(?, ''), target_profile_description)
                WHERE recipe_id = ?
                """,
                (target_profile or "", target_profile_description or "", row["recipe_id"]),
            )

    def get_chemistry_run(self, chem_hash: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM chemistry_runs WHERE chem_hash = ?", (chem_hash,)).fetchone()
        return dict(row) if row else None

    def chemistry_complete(self, chem_hash: str) -> bool:
        row = self.get_chemistry_run(chem_hash)
        return bool(row and row.get("status") == "complete")

    def upsert_chemistry_run(self, data: dict[str, Any]) -> None:
        fields = [
            "chem_hash",
            "chem_hash_version",
            "created_at",
            "status",
            "dat_lst_hash",
            "species_map_hash",
            "temperature_celsius",
            "pressure",
            "water_mol",
            "canonical_vector_json",
            "oxide_equivalent_vector_json",
            "xgems_run_dir",
            "warnings_json",
        ]
        values = [data.get(field) for field in fields]
        placeholders = ",".join("?" for _ in fields)
        updates = ",".join(f"{field}=excluded.{field}" for field in fields if field != "chem_hash")
        with self.connect() as conn:
            conn.execute(
                f"INSERT INTO chemistry_runs ({','.join(fields)}) VALUES ({placeholders}) "
                f"ON CONFLICT(chem_hash) DO UPDATE SET {updates}",
                values,
            )

    def get_prepared_chemistry_run(self, prepared_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM prepared_chemistry_runs WHERE prepared_id = ?",
                (prepared_id,),
            ).fetchone()
        return dict(row) if row else None

    def upsert_prepared_chemistry_run(self, data: dict[str, Any]) -> None:
        fields = [
            "prepared_id",
            "recipe_id",
            "chem_hash",
            "created_at",
            "reaction_model_id",
            "reaction_model_signature",
            "reaction_model_signature_version",
            "reaction_model_json",
            "recipe_json",
            "reaction_degrees_json",
            "xgems_species_amounts_json",
            "unreacted_masses_json",
            "canonical_vector_json",
            "oxide_equivalent_vector_json",
            "water_mol",
            "temperature_celsius",
            "pressure",
            "xgems_water_policy_json",
            "source_ledger_hash",
            "warnings_json",
        ]
        values = [data.get(field) for field in fields]
        placeholders = ",".join("?" for _ in fields)
        updates = ",".join(f"{field}=excluded.{field}" for field in fields if field != "prepared_id")
        with self.connect() as conn:
            conn.execute(
                f"INSERT INTO prepared_chemistry_runs ({','.join(fields)}) VALUES ({placeholders}) "
                f"ON CONFLICT(prepared_id) DO UPDATE SET {updates}",
                values,
            )

    def insert_recipe_run(self, data: dict[str, Any]) -> None:
        fields = [
            "recipe_id",
            "chem_hash",
            "prepared_id",
            "created_at",
            "reaction_model_id",
            "reaction_model_signature",
            "reaction_model_signature_version",
            "reaction_parameter_set_id",
            "reaction_parameter_config_path",
            "reaction_parameter_config_hash",
            "reaction_model_json",
            "template_name",
            "material_system",
            "target_profile",
            "target_profile_description",
            "recipe_json",
            "reaction_degrees_json",
            "initial_masses_json",
            "reacted_masses_json",
            "unreacted_masses_json",
            "water_g",
            "w_b",
            "water_mode",
            "xgems_water_g",
            "xgems_w_b",
            "xgems_water_mode",
            "xgems_water_policy_json",
            "solver_rescued",
            "xgems_retry_count",
            "primary_chem_hash",
            "primary_solver_status",
            "retry_history_json",
            "age_days",
            "age_hours",
            "age_minutes",
            "age_label",
            "age_bin",
            "temperature_celsius",
            "initial_volume_cm3",
            "final_solid_volume_cm3",
            "porosity",
            "run_dir",
            "warnings_json",
        ]
        values = [data.get(field) for field in fields]
        with self.connect() as conn:
            conn.execute(
                f"INSERT INTO recipe_runs ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})",
                values,
            )

    def insert_source_contributions(self, rows: Iterable[dict[str, Any]]) -> None:
        rows = list(rows)
        if not rows:
            return
        fields = [
            "recipe_id",
            "chem_hash",
            "source_material",
            "source_phase_or_oxide",
            "source_mass_g_initial",
            "reaction_degree",
            "reacted_mass_g",
            "unreacted_mass_g",
            "component",
            "component_mol",
            "oxide_equivalent",
            "oxide_equivalent_mol",
        ]
        with self.connect() as conn:
            conn.executemany(
                f"INSERT INTO source_contributions ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})",
                [[row.get(field) for field in fields] for row in rows],
            )

    def get_recipe_run(self, recipe_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM recipe_runs WHERE recipe_id = ?", (recipe_id,)).fetchone()
        return dict(row) if row else None

    def linked_recipe_ids(self, chem_hash: str) -> list[str]:
        with self.connect() as conn:
            rows = conn.execute("SELECT recipe_id FROM recipe_runs WHERE chem_hash = ? ORDER BY created_at", (chem_hash,))
            return [str(row["recipe_id"]) for row in rows]

    def recipe_rows(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM recipe_runs ORDER BY created_at, recipe_id").fetchall()
        return [dict(row) for row in rows]

    def prepared_rows(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM prepared_chemistry_runs ORDER BY created_at, prepared_id").fetchall()
        return [dict(row) for row in rows]

    def source_rows_for_recipe(self, recipe_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM source_contributions WHERE recipe_id = ? ORDER BY source_material, source_phase_or_oxide, component",
                (recipe_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def chemistry_dir(self, chem_hash: str) -> Path:
        path = self.chemistry_runs_dir / chem_hash
        path.mkdir(parents=True, exist_ok=True)
        return path

    def recipe_dir(self, recipe_id: str) -> Path:
        path = self.recipe_runs_dir / recipe_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def prepared_chemistry_dir(self, prepared_id: str) -> Path:
        path = self.prepared_chemistry_runs_dir / prepared_id
        path.mkdir(parents=True, exist_ok=True)
        return path


def write_name_value_csv(path: str | Path, data: Any) -> None:
    if isinstance(data, dict):
        rows = [{"name": str(name), "value": to_jsonable(value)} for name, value in data.items()]
    elif data is None:
        rows = []
    else:
        rows = [{"name": "raw", "value": to_jsonable(data)}]
    pd.DataFrame(rows, columns=["name", "value"]).to_csv(path, index=False)


def read_name_value_csv(path: str | Path) -> dict[str, float]:
    path = Path(path)
    if not path.exists():
        return {}
    out: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                out[str(row["name"])] = float(row["value"])
            except Exception:
                continue
    return out


def save_raw_xgems_state(raw_dir: str | Path, raw_state: dict[str, Any], species_amounts: dict[str, float]) -> None:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    write_json(raw_dir / "input_xgems_species_amounts.json", species_amounts)
    write_name_value_csv(raw_dir / "xgems_phase_amounts_raw.csv", raw_state.get("phase_masses"))
    write_name_value_csv(raw_dir / "xgems_phase_volumes_raw.csv", raw_state.get("phase_volumes"))
    write_phase_volume_reconstruction_files(raw_dir, raw_state)
    write_name_value_csv(raw_dir / "xgems_aqueous_species_raw.csv", raw_state.get("aqueous_species"))
    write_json(raw_dir / "xgems_scalars_raw.json", raw_state.get("scalars") or {})
    write_json(raw_dir / "xgems_attribute_report.json", raw_state.get("attribute_report") or {})


def copy_raw_outputs_to_chemistry_root(chem_dir: Path, raw_dir: Path) -> None:
    for name in [
        "input_xgems_species_amounts.json",
        "xgems_phase_amounts_raw.csv",
        "xgems_phase_volumes_raw.csv",
        "xgems_phase_volumes_reconstructed.csv",
        "xgems_phase_volume_reconstruction_report.csv",
        "xgems_phase_volume_reconstruction_summary.json",
        "xgems_aqueous_species_raw.csv",
        "xgems_scalars_raw.json",
        "xgems_attribute_report.json",
        "chemistry_provenance.json",
    ]:
        src = raw_dir / name
        if src.exists():
            shutil.copy2(src, chem_dir / name)
