# InverseGems

InverseGems (Python package `inverse_gems`) is a research framework for forward thermodynamic modeling and surrogate-assisted inverse design of blended cementitious binders with xGEMS/GEMS. It parses simple 100 g binder-basis recipes, computes OPC and SCM reaction degrees, builds xGEMS species inputs, runs either xGEMS/GEMS or a mock runner, saves raw outputs, and computes best-effort porosity. A reactive-chemistry database keyed by a deterministic chemistry fingerprint provides caching, provenance, surrogate training data, and active-learning expansion; an optional LLM layer translates natural-language requests into schema-validated task specifications while every scientific step stays deterministic.

InverseGems is built on [xGEMS](https://github.com/gemshub/xGEMS)/[GEMS3K](https://gems.web.psi.ch/) and the Cemdata thermodynamic database. It is an independent project, not affiliated with or endorsed by the GEMS development team.

**License**: BSD 3-Clause (see [LICENSE](LICENSE)).

## Related project: GemsPilot (agent layer)

The LLM agent layer that previously lived in this repository — the framework-neutral tool layer, the MCP server, observe–replan recovery loops, the autonomous coverage-growth campaign, session memory, and GEMS-Agent-Bench — now lives in the companion project **GemsPilot**. InverseGems is the deterministic scientific kernel; GemsPilot orchestrates it. The last snapshot containing the agent layer in this repository is tagged `v0.2.0-with-agent-layer`.

Kernel features that remain here: forward/inverse pipelines, the reactive-chemistry database, surrogate training and repeated-CV evaluation, pluggable SCM kinetics with the user-data calibration route, the OAT sensitivity workflow, per-request xGEMS call budgets, and diagnosis-driven solver water recovery.

## Installation

```bash
pip install -e .
```

## Mock Test

```bash
inverse-gems forward-mock --recipe "OPC 30, fly ash 70, w/b 0.4, age 28"
```

The mock command does not require xGEMS/GEMS. It creates a complete `runs/run_YYYYMMDD_HHMMSS_<hash>/` directory with fake raw phase/species outputs so the file-writing and porosity pipeline can be tested.

## Real xGEMS/GEMS Run

```bash
inverse-gems forward --dat-lst path/to/dat.lst --recipe "OPC 30, fly ash 70, w/b 0.4, age 28"
```

The real runner currently tries the import style used by the provided example:

```python
from run.GEMSCalc import GEMS
```

If your local xGEMS/GEMS module exposes a different API, update `XGEMSRunner` or pass an equivalent class path in code. The species names sent to xGEMS are configured in `configs/species_map.yaml`; edit that file when your `dat.lst` uses different names.

Some xGEMS databases do not expose clinker or oxide input species directly. For those, use formula mode to convert the prepared input amounts to element bulk composition:

```bash
inverse-gems forward --dat-lst Test-dat.lst --recipe "OPC 30, fly ash 70, w/b 0.4, age 28" --gems-class-path "xgems:ChemicalEngineDicts" --xgems-input-mode formula
```

## Output Files

Every run creates:

```text
manifest.json
input_user_request.txt
input_recipe.json
input_materials_used.json
input_reaction_degrees.json
input_xgems_species_amounts.json
xgems_phase_amounts_raw.csv
xgems_phase_volumes_raw.csv
xgems_aqueous_species_raw.csv
xgems_scalars_raw.json
xgems_attribute_report.json
xgems_stdout.txt
xgems_stderr.txt
porosity.json
warnings.json
```

Phase names are intentionally not cleaned, renamed, merged, aliased, or interpreted in this milestone. Raw xGEMS/GEMS names are preserved exactly as returned so they can be inspected before any later aggregation policy is designed.

## Early-Dense Cached Workflow

Recommended pilot database size:

```text
3,000 to 5,000 base recipes x 28 ages = 84,000 to 140,000 recipe rows
```

Recommended initial production database size:

```text
10,000 to 20,000 base recipes x 28 ages = 280,000 to 560,000 recipe rows
```

Start with:

```bash
inverse-gems generate-recipes \
  --config configs/sampling.yaml \
  --n 5000 \
  --mode mixed \
  --age-preset early_dense_v1 \
  --out data/recipes_early_dense_5k.csv \
  --seed 42
```

For practical material-system-specific studies, prefer generating recipes inside a named material system. The profiles live in `configs/material_systems.yaml`; disallowed materials are fixed to zero.

```bash
inverse-gems generate-recipes \
  --config configs/sampling.yaml \
  --material-system OPC_slag \
  --n 2000 \
  --ages 28 \
  --out data/recipes_OPC_slag_age28.csv \
  --seed 42
```

This creates 100 g binder-basis recipes using only OPC, slag, and gypsum for `OPC_slag`; fly ash, metakaolin, silica fume, and limestone are zero. Use `OPC_slag_limestone` when limestone should be allowed.

Available profiles include strict binary systems such as `OPC_slag`, `OPC_fly_ash`, `OPC_metakaolin`, `OPC_limestone`, and `OPC_silica_fume`, plus ternary systems such as `OPC_slag_limestone`, `OPC_fly_ash_limestone`, `LC3_like`, `OPC_slag_fly_ash`, and `OPC_slag_silica_fume`. Gypsum is treated as a controllable sulfate-regulator input rather than a reason to split the database.

The recommended long-term layout is one master cached xGEMS database with `material_system` and `age_days` stored as metadata. Build material-system or age-specific views at the model-table layer instead of physically splitting every combination into separate databases:

```bash
inverse-gems build-model-table \
  --feature-table data/feature_table_master.parquet \
  --config configs/model_dataset_OPC_slag_age28_stable_targets.yaml \
  --out data/model_table_OPC_slag_age28_stable_targets.parquet \
  --format parquet
```

Then run cached xGEMS:

```bash
inverse-gems run-batch-cached \
  --dat-lst path/to/dat.lst \
  --recipes data/recipes_early_dense_5k.csv \
  --db data/inverse_gems_db_early_dense/ \
  --resume \
  --progress-every 25
```

For a no-xGEMS mock workflow:

```bash
inverse-gems run-batch-cached-mock \
  --recipes data/recipes_early_dense_5k.csv \
  --db data/inverse_gems_db_mock_early_dense/ \
  --progress-every 25
```

Long cached batch runs write scale-up monitoring artifacts directly inside the DB directory:

- `batch_manifest.json`: run settings, recipe count, xGEMS mode, retry policy
- `batch_status.csv`: one row per attempted recipe
- `batch_progress.json`: current processed/complete/failed/cache-hit/rescued counts
- `batch_failures.csv`: failed or non-complete rows only
- `batch_summary.md`: human-readable progress and failure preview

Refresh these reports at any time without running new xGEMS calculations:

```bash
inverse-gems batch-status \
  --db data/inverse_gems_db_early_dense/ \
  --out reports/batch_status_early_dense/
```

Summarize and build features:

```bash
inverse-gems summarize-db \
  --db data/inverse_gems_db_early_dense/ \
  --out reports/db_summary_early_dense/

inverse-gems build-feature-table \
  --db data/inverse_gems_db_early_dense/ \
  --selection configs/output_selection.yaml \
  --out data/feature_table_early_dense.parquet \
  --format parquet
```

Before inverse querying or surrogate modeling, run feature diagnostics:

```bash
inverse-gems feature-diagnostics \
  --feature-table data/feature_table_early_dense.parquet \
  --out reports/feature_diagnostics_early_dense/
```

This writes numeric summaries, selected phase amount/volume mismatch checks, sparse-output flags, correlation tables, and optional PNG plots.

Build a model-ready table with fixed inputs and targets:

```bash
inverse-gems build-model-table \
  --feature-table data/feature_table_early_dense.parquet \
  --config configs/model_dataset.yaml \
  --out data/model_table_early_dense.parquet \
  --format parquet
```

The default model dataset keeps only complete chemistry rows, uses binder masses plus `w_b`, `age_days`, and `log_age_days` as inputs, and writes selected phase-group/scalar targets with `x__`, `y__`, and `meta__` column prefixes. It also keeps `prepared_id`, `reaction_model_id`, and `reaction_model_signature` as metadata.

For reaction-parameter-robust surrogate work, build a chemistry-input model table instead of a recipe-input model table:

```bash
inverse-gems build-model-table \
  --feature-table data/feature_table_early_dense.parquet \
  --config configs/model_dataset_chemistry_stable_targets.yaml \
  --out data/model_table_chemistry_stable_targets.parquet \
  --format parquet
```

This uses reactive oxide-equivalent moles, xGEMS water, and temperature as `x__` inputs. Recipe masses, age, and reaction-model provenance remain as metadata for tracing and later xGEMS validation. To screen new recipe candidates with such a surrogate, first project candidate recipes to the same chemistry feature space without running xGEMS:

```bash
inverse-gems build-chemistry-candidate-table \
  --recipes-csv data/candidate_recipes.csv \
  --dat-lst Test-dat.lst \
  --out data/candidate_chemistry_table.csv
```

Changing PK/SCM/availability parameters changes the recipe-to-chemistry projection and its `reaction_model_signature`; if the resulting chemistry stays inside the trained chemistry-space coverage, the chemistry surrogate can still be reused before final xGEMS validation.

## Global Reactive-Chemistry Workflow

Prefer a global reactive-chemistry DB over separate age/material-system DBs. In this layout, recipe, age, and material system are provenance; `chem_hash` and the reactive chemistry vector are the lookup/modeling keys.

Initialize the global DB wrapper:

```bash
inverse-gems init-global-chem-db \
  --db data/global_chem_db/
```

Generate a mixed material-system pool with continuous age sampling:

```bash
inverse-gems generate-recipes \
  --config configs/sampling.yaml \
  --n 500 \
  --mode mixed \
  --material-systems OPC_only,OPC_slag,OPC_fly_ash,OPC_metakaolin,OPC_silica_fume,OPC_slag_fly_ash,LC3_like \
  --material-systems-sampling balanced \
  --age-sampling log_uniform \
  --age-min 0.1 \
  --age-max 365 \
  --age-count 2 \
  --recipe-id-prefix global_pool \
  --out data/recipes_global_pool.csv
```

Run xGEMS or mock xGEMS into the same global DB, then refresh chemistry/model artifacts:

```bash
inverse-gems run-batch-cached-mock \
  --recipes data/recipes_global_pool.csv \
  --db data/global_chem_db/ \
  --resume

inverse-gems refresh-global-chem-db \
  --db data/global_chem_db/
```

Check DB coverage before treating the surrogate as ready for inverse-design routing:

```bash
inverse-gems global-chemistry-coverage \
  --db data/global_chem_db/ \
  --out reports/global_chemistry_coverage/
```

The coverage report writes material-system counts, age-milestone coverage, chemistry-input ranges, w/b range, target metric status when available, and warnings such as missing systems, missing ages, missing `target_metrics.csv`, or excessive water-adjusted pH rows.

Existing cached DBs can be merged into the same global DB:

```bash
inverse-gems import-global-chem-db \
  --db data/global_chem_db/ \
  --source-db data/older_cached_db/
```

Train the global chemistry surrogate:

```bash
inverse-gems train-global-chem-surrogate \
  --db data/global_chem_db/ \
  --surrogate-config configs/surrogate_baseline_xgems_env.yaml
```

Run `global-chemistry-coverage` again after training so target-level R2/MAE from `target_metrics.csv` are included.

Lookup or acquire new chemistry points before deciding which xGEMS runs to add:

```bash
inverse-gems lookup-global-chem-db \
  --db data/global_chem_db/ \
  --recipes-csv data/new_candidate_pool.csv \
  --out results/global_lookup/

inverse-gems acquire-global-chemistry \
  --db data/global_chem_db/ \
  --recipes-csv data/new_candidate_pool.csv \
  --out results/global_acquisition/ \
  --max-candidates 50
```

`acquire-global-chemistry` writes `acquisition_recipes.csv`, which can be sent directly to `run-batch-cached` or `run-batch-cached-mock`. The acquisition score prioritizes non-exact chem hashes, out-of-domain candidates, outside-range candidates, and candidates far from the nearest reference chemistry.

For active-learning rounds aimed at weak surrogate targets, add priority targets directly or load them from diagnostics:

```bash
inverse-gems acquire-global-chemistry \
  --db data/global_chem_db/ \
  --recipes-csv data/new_candidate_pool.csv \
  --out results/global_acquisition_targeted/ \
  --max-candidates 50 \
  --priority-target ettringite \
  --priority-target monocarbonate \
  --priority-targets-from-diagnostics reports/model_registry_diagnostics_current/model_registry_diagnostics.csv \
  --priority-target-kind phase
```

Priority targets add a normalized surrogate-predicted target score to the novelty/domain acquisition score. This is intended to propose additional xGEMS calculations near phases that are sparse or have low R2; final candidates still need xGEMS validation. Use `--priority-target-kind phase` when diagnostics include scalar targets such as pH or porosity but the active-learning round should focus only on phase amount/volume targets.

Forward and inverse-query wrappers can use the same global DB manifest:

```bash
inverse-gems run-global-forward-query-mock \
  --global-db data/global_chem_db/ \
  --query configs/forward_query.volume_vs_time.example.yaml \
  --out results/global_forward_smoke/

inverse-gems run-global-design-query \
  --global-db data/global_chem_db/ \
  --query configs/design_query.global_smoke.yaml \
  --out results/global_design_smoke/ \
  --n-candidates 100
```

For real validation, use `run-global-forward-query` or `run-global-design-query --validate` with `--dat-lst Test-dat.lst`, `--xgems-input-mode formula`, and the same xGEMS class path used by direct cached runs.

For user requests that mean "only these materials", enable strict material mode. This keeps optional materials such as gypsum at zero unless they are explicitly allowed:

```bash
inverse-gems run-chemistry-design-query \
  --query results/manual_age56_opc_slag_fly_ash_strict_chemistry_design_query.yaml \
  --model-bundle reports/baseline_surrogate_chemistry_age28_current_scale1000_retry_all_systems/model.joblib \
  --reference-model-table data/model_table_chemistry_age28_current_scale1000_retry_all_systems_stable_targets.csv \
  --dat-lst Test-dat.lst \
  --out results/manual_age56_opc_slag_fly_ash_strict_chemistry_design_run \
  --strict-materials \
  --validate \
  --validation-top-k 5
```

The run writes `candidate_review.csv` with flags such as `validated`, `surrogate_only`, `out_of_domain`, `solver_rescued`, and `pH_water_uncertain`. A candidate marked `out_of_domain` should be treated as a screening suggestion until it is validated with xGEMS/GEMS.

To start expanding chemistry coverage across ages, generate a strict multi-age pilot set and run it through the cached xGEMS workflow:

```bash
inverse-gems generate-recipes \
  --config configs/sampling.yaml \
  --n 8 \
  --age-preset standard_sparse \
  --material-system OPC_slag_fly_ash \
  --strict-materials \
  --recipe-id-prefix OPC_slag_fly_ash_strict_multiage_pilot \
  --out data/recipes_OPC_slag_fly_ash_strict_multiage_pilot.csv

inverse-gems run-batch-cached \
  --dat-lst Test-dat.lst \
  --recipes data/recipes_OPC_slag_fly_ash_strict_multiage_pilot.csv \
  --db data/inverse_gems_db_real_OPC_slag_fly_ash_strict_multiage_pilot \
  --xgems-input-mode formula \
  --retry-water-on-failure
```

If a master feature table contains multiple reaction parameter sets, build model tables by reaction signature before training a surrogate:

```bash
inverse-gems build-model-table \
  --feature-table data/feature_table_early_dense.parquet \
  --config configs/model_dataset.yaml \
  --out data/model_table_early_dense_trial_params.parquet \
  --format parquet \
  --reaction-model-id trial_parameters_v0
```

For inverse-design prototyping, prefer the stable target split first:

```bash
inverse-gems build-model-table \
  --feature-table data/feature_table_early_dense.parquet \
  --config configs/model_dataset_stable_targets.yaml \
  --out data/model_table_early_dense_stable_targets.parquet \
  --format parquet
```

Sparse targets such as monosulfate, hemicarbonate, and Al(OH)3mic are separated in `configs/model_dataset_sparse_targets.yaml` so they can later use occurrence classifiers or two-stage models.

Train a baseline surrogate before attempting inverse optimization:

```bash
inverse-gems train-baseline-surrogate \
  --model-table data/model_table_early_dense.parquet \
  --config configs/surrogate_baseline.yaml \
  --out reports/baseline_surrogate_early_dense/
```

The baseline command uses a group split by base recipe ID, writes target-wise metrics, test predictions, feature importance tables, plots, and a `model.joblib` bundle.
The bundle also records the `reaction_model_signature` values used for training in `surrogate_model_manifest.json`. Candidate search compares the model table and bundle provenance before ranking candidates.
For automatic model selection, add `reaction_model_id` and `reaction_model_signature` to the relevant entry in `configs/design_query_model_registry.yaml`. The design-query compiler first matches `material_system` and `age_days`, then narrows the match by `reaction_model.id` or `reaction_model.signature` when the user query or CLI provides one. Older registry entries without reaction provenance still work for unversioned diagnostic queries, but they are deliberately rejected when a specific reaction model is requested.

After registering trained models, generate a target-availability report before using those targets in inverse design:

```bash
inverse-gems model-registry-diagnostics \
  --model-registry configs/design_query_model_registry.yaml \
  --out reports/model_registry_diagnostics_current/ \
  --reaction-model-id local_default_parameters
```

This combines each model table schema with surrogate `target_metrics.csv` and flags all-zero, sparse, near-constant, missing-metric, and low-R2 targets. A high R2 alone is not enough; for example pH is marked as near-constant when its training range is too small to be a meaningful inverse-design target.
`compile-design-query` and `run-design-query` run the same check by default with `--target-availability-policy warn`, writing `target_availability_report.json` beside the compiled query. Use `--target-availability-policy error` when unavailable targets should block the workflow.

Search observed candidate recipes with the stable surrogate:

```bash
inverse-gems surrogate-candidate-search \
  --query configs/surrogate_candidate_search.age28_balanced.yaml \
  --out results/surrogate_candidate_search_age28_balanced/ \
  --reaction-model-id trial_parameters_v0
```

The first candidate-search mode is intentionally conservative: it ranks candidates drawn from the existing model table, using surrogate-predicted targets for filtering and scoring. It writes `candidates.csv`, `candidates.json`, a constraint summary, and `reaction_provenance_report.json`. By default, a reaction signature mismatch between the model table, surrogate bundle, and explicitly requested reaction model raises an error; use `--reaction-model-mismatch-policy warn` only for diagnostic work.

For composition-to-composition comparison, fix `age_days` in the candidate-search and selection configs, commonly to 28 days. Leaving age unconstrained is useful for time-window discovery, but it ranks maturity effects together with composition effects.

Revalidate the top surrogate candidates with real xGEMS/GEMS before trusting them as design suggestions:

```bash
inverse-gems validate-candidates \
  --candidates results/surrogate_candidate_search_age28_balanced/candidates.csv \
  --dat-lst Test-dat.lst \
  --db data/validation_candidate_search_age28_balanced/ \
  --out results/validate_candidates_age28_balanced/ \
  --top-k 10 \
  --retry-water-on-failure \
  --xgems-input-mode formula \
  --gems-class-path xgems:ChemicalEngineDicts
```

This writes `validation_runs.csv`, `validated_feature_table.csv`, and `validation_comparison.csv`, with each validated recipe kept on one row and raw/selected phase names still preserved by the existing output-selection configuration. Use `validate-candidates-mock` for a no-xGEMS smoke test.

Select validated candidates with editable YAML constraints and objectives:

```bash
inverse-gems select-candidates \
  --validation results/validate_candidates_age28_balanced/validation_comparison.csv \
  --config configs/candidate_selection.age28_balanced.yaml \
  --out results/selected_candidates_age28_balanced/
```

The selection command automatically uses `validated_feature_table.csv` beside the validation comparison when available. It writes `selected_candidates.csv`, `selected_candidates.json`, `selected_candidates.md`, `selection_summary.json`, and `rejected_by_selection_constraints.json`.

Selection constraints are intentionally optional and query-dependent. A user request may constrain only ettringite, Portlandite, C-A-S-H, gypsum, or any other available output without constraining porosity. For example configs, see `configs/candidate_selection.OPC_slag_age28.yaml` and `configs/candidate_selection.OPC_slag_age28_ettringite_cap_no_porosity.yaml`.

End-to-end design-query runs also write researcher-facing candidate review files beside the raw final candidate table:

```text
candidate_review.csv
candidate_review.json
candidate_review.md
candidate_review_summary.json
```

These files keep one candidate per row with recipe columns, predicted outputs, validated outputs when available, prediction deltas, validation status, and uncertainty flags such as `surrogate_only`, `solver_rescued`, or `pH_water_uncertain`. The raw `final_candidates.csv` or `final_selected_candidates.csv` is still preserved unchanged.

For future API/LLM use, do not let the model execute code or invent ranking logic directly. Have it emit the strict design-query schema in `configs/design_query.schema.json`, validate it, then compile it into executable search and selection configs.

The preferred inverse-design query is target-first: the user gives desired phase/porosity ranges and optional material/input constraints, not a fixed recipe. For example, `configs/design_query.target_first.example.yaml` uses `design_space`, `output_constraints`, ordered `objectives`, and `validation`:

```yaml
design_space:
  material_systems: [OPC_slag]
  allowed_materials: [OPC, slag, gypsum]
  input_constraints:
    OPC: {max: 40}
    w_b: {min: 0.30, max: 0.50}
  age_days: 28

output_constraints:
  porosity: {max: 0.38}
  ettringite: {max: 0.01}
  C-A-S-H: {min: 0.04}

objectives:
  - {target: C-A-S-H, direction: maximize}
  - {input: OPC, direction: minimize}
  - {target: porosity, direction: minimize}

validation:
  search_top_k: 60
  top_k_xgems: 12
```

During compilation, `allowed_materials` is converted into input constraints that force all other binder components to zero. Output constraints are applied to surrogate-predicted targets during search and validated targets after xGEMS/GEMS validation. Model paths are resolved from `configs/design_query_model_registry.yaml` using `material_system`/`age_days` or `design_space.material_systems`/`design_space.age_days`, with optional narrowing by `reaction_model.id` and `reaction_model.signature`. If more than one registered model still matches, set `model_id` explicitly.

```bash
inverse-gems design-query-schema --out configs/design_query.schema.json

inverse-gems validate-design-query \
  --query configs/design_query.target_first.example.yaml \
  --require-model-paths

inverse-gems compile-design-query \
  --query configs/design_query.target_first.example.yaml \
  --out results/compiled_target_first_design_query_example/
```

The compiled files are `surrogate_candidate_search.yaml`, `candidate_selection.yaml`, and `design_query_manifest.json`. The manifest records which registry entry supplied the model table and model bundle. Ordered user priorities are represented with `objectives` or `preferences`; the order is the priority order. Optional `tolerance` lets nearly equivalent first-priority values fall through to the next preference.

To run the compiled workflow in one command, use `run-design-query`. Add `--skip-validation` for a fast surrogate-only pass, or omit it and provide `--db` plus `--dat-lst` for xGEMS/GEMS validation and final selection:

```bash
inverse-gems run-design-query \
  --query configs/design_query.example.yaml \
  --out results/design_query_example_run/ \
  --skip-validation
```

General forward calculations use a separate forward-query schema. This is for requests such as "OPC 40% + slag 30% + fly ash 30%, age 0.1 days to 360 days, calculate and plot volume vs time." The LLM/API layer should emit `configs/forward_query.schema.json` instead of the inverse-design schema when the user asks for direct calculation, time series, or plotting:

```bash
inverse-gems forward-query-schema --out configs/forward_query.schema.json

inverse-gems validate-forward-query \
  --query configs/forward_query.volume_vs_time.example.yaml

inverse-gems run-forward-query-mock \
  --query configs/forward_query.volume_vs_time.example.yaml \
  --out results/forward_query_volume_vs_time_mock/ \
  --db data/forward_query_mock_db/
```

For a real xGEMS/GEMS time series, use `run-forward-query` with the usual `dat.lst` and xGEMS adapter options:

```bash
inverse-gems run-forward-query \
  --query configs/forward_query.volume_vs_time.example.yaml \
  --dat-lst Test-dat.lst \
  --out results/forward_query_volume_vs_time_real/ \
  --db data/inverse_gems_db_real_early_dense_2000_retry/ \
  --xgems-input-mode formula \
  --gems-class-path xgems:ChemicalEngineDicts \
  --retry-water-on-failure \
  --no-plots
```

Forward-query output keeps each requested age on one row in `time_series.csv`, with raw phase names preserved in columns such as `phase_volume__CNASH` or `phase_mass__ettringite`. It also writes `forward_query_summary.json`, the normalized manifest, the original query YAML, requested PNG plots, and diagnostics files:

- `forward_query_status.csv`
- `failed_ages.csv`
- `phase_nonzero_summary.csv`
- `phase_change_summary.csv`
- `scalar_timeseries.csv`
- `diagnostics.md`
- `response_summary.json`
- `response_summary.md`
- `response_summary.csv`
- `answer.json`
- `answer.md`
- `narrative_answer.json`
- `narrative_answer.md`

`response_summary.*` is generated automatically from the `forward_query.response_summary` block. This block selects exact raw phase names and scalar names to report after calculation:

```yaml
response_summary:
  phases: [CNASH, Portlandite]
  scalars: [pH, porosity]
  top_phases: 8
  table_limit: 20
  narrative_enabled: true
  narrative_language: ko
```

If `phases` is empty, the local code selects top nonzero raw phases from `phase_nonzero_summary.csv`. No phase aliases, grouping, cleaning, or aggregation are applied.

`answer.md` and `answer.json` are then generated from `response_summary.json`. They provide a short user-facing answer, a preview table, deterministic first/final/min/max numeric summaries, missing requested names, and links to the raw source files. The answer layer does not reinterpret phase names.

`narrative_answer.md` and `narrative_answer.json` are then generated from `answer.json`. By default this is a deterministic template, so no API key is required. It only rewrites already-selected values into prose and does not calculate, rename phases, group phases, or infer missing outputs.

To extract user-requested values from an existing forward-query run, use the response summary command:

```bash
inverse-gems summarize-forward-result \
  --run results/forward_query_volume_vs_time_real/ \
  --phase CNASH \
  --phase Portlandite \
  --scalar pH \
  --scalar porosity \
  --out results/forward_summary_volume_vs_time/
```

This writes `response_summary.json`, `response_summary.md`, and `response_summary.csv`. Phase names are exact raw xGEMS/GEMS names; the command does not alias, group, clean, or aggregate phases. Missing requested names are reported together with available raw phase and scalar names.

To regenerate the user-facing answer from an existing response summary:

```bash
inverse-gems write-forward-answer \
  --run results/forward_query_volume_vs_time_real/ \
  --table-limit 10
```

To regenerate the deterministic narrative:

```bash
inverse-gems write-forward-narrative \
  --run results/forward_query_volume_vs_time_real/ \
  --language ko
```

To use OpenAI only for wording, after `answer.json` already exists:

```bash
inverse-gems write-forward-narrative \
  --run results/forward_query_volume_vs_time_real/ \
  --language ko \
  --use-openai
```

The OpenAI narrative prompt instructs the model to use only `answer.json`, preserve raw phase names exactly, and avoid aliases, grouping, aggregation, or scientific reinterpretation.

## Unified Request API

For applications, use the facade API instead of calling each pipeline step manually:

```python
from inverse_gems import parse_request_preview, run_confirmed_request, run_request

result = run_request(
    forward_query="configs/forward_query.volume_vs_time.example.yaml",
    out="results/request_forward/",
    db="data/request_forward_db/",
    dat_lst="Test-dat.lst",
)

print(result.answer_text)
print(result.files["narrative_answer_md"])
```

Free-text requests can be split into a reviewable parse/preview response and a later confirmed execution:

```python
preview = parse_request_preview(
    request="Use OPC and slag at 28 days; find low porosity binders with low OPC.",
    out="results/api_preview/",
)

print(preview.files["parsed_query_preview_md"])
print(preview.summary["selected_model"])

confirmed = run_confirmed_request(
    confirmed_preview="results/api_preview/openai_parse/",
    out="results/api_confirmed/",
    db="data/api_confirmed_db/",
    confirm_preview=True,
    use_mock=True,
    skip_validation=True,
)

print(confirmed.files["final_candidates_csv"])
```

All facade calls return `RequestResult`. Its `files` map gives important artifact paths, while `summary` carries compact machine-readable details such as preview risk counts, selected model, final candidate locations, or forward response summary.

The equivalent mock CLI command is:

```bash
inverse-gems run-request-mock \
  --forward-query configs/forward_query.volume_vs_time.example.yaml \
  --out results/request_forward_mock/ \
  --db data/request_forward_mock_db/ \
  --no-plots
```

For a structured task query:

```bash
inverse-gems run-request-mock \
  --task-query configs/task_query.forward_volume_vs_time.example.yaml \
  --out results/request_task_mock/ \
  --db data/request_task_mock_db/ \
  --no-plots
```

For free-text requests, enable the OpenAI parser explicitly:

```bash
inverse-gems run-request \
  --request "OPC 40 + slag 30 + fly ash 30, age 0.1 to 360 days, show CNASH and porosity" \
  --use-openai \
  --dat-lst Test-dat.lst \
  --out results/request_openai_real/ \
  --db data/request_openai_real_db/ \
  --xgems-input-mode formula \
  --gems-class-path xgems:ChemicalEngineDicts \
  --no-plots
```

Without `--use-openai`, free-text requests are rejected rather than guessed. Use `--forward-query` or `--task-query` for deterministic execution.

## Environment Check

Before a real xGEMS/GEMS run, check the active Python environment:

```bash
inverse-gems check-env \
  --dat-lst Test-dat.lst \
  --gems-class-path xgems:ChemicalEngineDicts \
  --xgems-input-mode formula \
  --require-xgems \
  --instantiate-runner
```

On this Windows machine, the known-good xGEMS environment is usually more reliable when called by direct `python.exe` rather than `conda run`:

```powershell
$env:PYTHONPATH = (Resolve-Path 'src').Path
& C:\Users\solmo\miniforge3\envs\py313-xgems\python.exe -m inverse_gems.cli check-env `
  --dat-lst Test-dat.lst `
  --gems-class-path xgems:ChemicalEngineDicts `
  --xgems-input-mode formula `
  --require-xgems `
  --instantiate-runner `
  --out results/check_env_xgems.json
```

The same direct-env style was used for this known-good real facade smoke test:

```powershell
$env:PYTHONPATH = (Resolve-Path 'src').Path
& C:\Users\solmo\miniforge3\envs\py313-xgems\python.exe -m inverse_gems.cli run-request `
  --forward-query configs/forward_query.volume_vs_time.example.yaml `
  --dat-lst Test-dat.lst `
  --out results/request_forward_real_facade `
  --db data/request_forward_real_facade_db `
  --xgems-input-mode formula `
  --gems-class-path xgems:ChemicalEngineDicts `
  --retry-water-on-failure `
  --no-plots
```

Expected smoke-test outcome: `request_result.json` has `status: complete`, `row_count: 24`, `completed_count: 24`, `failed_count: 0`, and no missing requested phases or scalars.

## Acceptance Suite

Run the no-xGEMS acceptance suite after code changes:

```bash
inverse-gems acceptance-mock --out results/acceptance_mock/
```

Run the real xGEMS/GEMS acceptance suite from the xGEMS Conda environment:

```powershell
$env:PYTHONPATH = (Resolve-Path 'src').Path
& C:\Users\solmo\miniforge3\envs\py313-xgems\python.exe -m inverse_gems.cli acceptance-real `
  --dat-lst Test-dat.lst `
  --out results/acceptance_real/ `
  --gems-class-path xgems:ChemicalEngineDicts `
  --xgems-input-mode formula
```

The suite runs representative OPC, binary, ternary, LC3-like, single-age, and time-series cases. It writes `environment_report.json`, `acceptance_cases.json`, `acceptance_report.csv`, `acceptance_report.json`, and `acceptance_report.md`. Each recipe/query is kept on one row in the CSV; raw phase names in `phase_masses_top_json`, `phase_volumes_top_json`, and detail JSON files are preserved exactly.

Use `--case <case_id>` to run one case, `--no-retry-water-on-failure` to require the primary water policy to succeed, and `--with-plots` to include forward-query plots.

For real xGEMS/GEMS runs in the Conda xGEMS environment, `--no-plots` is recommended during calculation. It still writes all CSV and Markdown diagnostics. Plots can be generated later from `time_series.csv` in a regular plotting environment.

Single-age forward calculations use the same schema with `age_grid.values: [28]`. In that case, the runner also writes easy-to-read single-result artifacts:

```bash
inverse-gems run-forward-query-mock \
  --query configs/forward_query.single_age.example.yaml \
  --out results/forward_query_single_age_mock/ \
  --db data/forward_query_single_age_mock_db/
```

Additional single-age outputs include `single_result.json`, `single_result.csv`, `raw_phase_masses.csv`, `raw_phase_volumes.csv`, `raw_scalars.json`, and `calculation_summary.md`. Raw phase names are still preserved exactly.

For a single LLM/API entrypoint, use the task-query router schema. The router does not interpret chemistry or rank candidates itself; it only validates `task_type` and forwards the nested payload to either the forward-query runner or design-query runner:

```bash
inverse-gems task-query-schema --out configs/task_query.schema.json

inverse-gems validate-task-query \
  --query configs/task_query.forward_volume_vs_time.example.yaml

inverse-gems run-task-query-mock \
  --query configs/task_query.forward_volume_vs_time.example.yaml \
  --out results/task_query_forward_volume_vs_time_mock/ \
  --db data/task_query_mock_db/
```

Use `task_type: forward_time_series` or `task_type: forward_calculation` with `forward_query` for direct calculations and plotting. Use `task_type: inverse_design` with `design_query` for candidate search and optional xGEMS validation. For inverse-design surrogate-only routing, add `--skip-validation`; for real forward calculations or real validation, provide `--dat-lst`.

Inverse-design task queries may omit `material_system` when the user only names usable materials. In that case, local routing selects a registry model from `configs/design_query_model_registry.yaml` using `design_space.allowed_materials`, age, reaction-model provenance, and target diagnostics:

```bash
inverse-gems run-task-query-mock \
  --query configs/task_query.auto_inverse_design.example.yaml \
  --out results/task_query_auto_route_mock/ \
  --db data/task_query_auto_route_mock_db/ \
  --skip-validation
```

The same routing is available through `run-openai-task-query` and `run-request`. By default, routing only accepts targets marked `recommended` by model diagnostics. Use `--route-target-policy allow_caution` only when you intentionally want to allow targets such as pH that diagnostics marked usable with caution.

Before executing a task query, write a reviewable preview:

```bash
inverse-gems preview-task-query \
  --query configs/task_query.auto_inverse_design.example.yaml \
  --out results/preview_task_query_auto_example/
```

The preview writes `parsed_query_preview.json` and `parsed_query_preview.md`. For inverse design it includes the requested materials, age, targets, input constraints, route diagnostics, selected model, and any warnings such as automatic material-system selection or caution-only targets.

After reviewing the preview, execute exactly that reviewed task query with an explicit confirmation flag:

```bash
inverse-gems run-confirmed-task-query-mock \
  --preview results/preview_task_query_auto_example/ \
  --out results/confirmed_task_query_auto_example/ \
  --db data/confirmed_task_query_auto_example_db/ \
  --skip-validation \
  --confirm-preview
```

Confirmed runs refuse to start unless `--confirm-preview` is present. The run directory keeps a copy of the reviewed `task_query.yaml`, `parsed_query_preview.json`, and `parsed_query_preview.md` under `confirmed_preview/`.

The same confirmed execution is available through the unified request facade:

```bash
inverse-gems run-request-mock \
  --confirmed-preview results/preview_task_query_auto_example/ \
  --out results/request_confirmed_task_query_auto_example/ \
  --db data/request_confirmed_task_query_auto_example_db/ \
  --skip-validation \
  --confirm-preview
```

The LLM-facing prompt and examples are kept in editable config files. Render the complete prompt, then validate the LLM's YAML/JSON output before running anything:

```bash
inverse-gems render-task-router-prompt \
  --out results/rendered_task_router_prompt.md

inverse-gems validate-llm-task-output \
  --input path/to/llm_output.yaml \
  --out results/llm_task_output_validation_report.json \
  --fail-on-invalid
```

If validation fails, the report includes a `repair_prompt` that can be sent back to the LLM once. This repair loop is still parser-only: the LLM corrects the schema object, while all calculations, model selection, ranking, and xGEMS/GEMS execution remain local and deterministic.

With the optional OpenAI wrapper, the same flow can call the model directly. Set `OPENAI_API_KEY`, install the optional dependency if needed, and keep the output validation step in the loop:

```bash
pip install -e ".[llm]"

inverse-gems parse-task-query-openai \
  --request "OPC 40 + slag 30 + fly ash 30, age 0.1 to 360 days, calculate and plot volume vs time" \
  --out results/openai_parse_forward_volume/
```

The OpenAI commands automatically load `.env` from the current working directory or project root without overriding already-set environment variables. A local `.env` can contain `OPENAI_API_KEY=...`; optionally set `INVERSE_GEMS_OPENAI_MODEL=...` to override the default model.

Successful OpenAI parse commands also write `parsed_query_preview.json` and `parsed_query_preview.md` next to `task_query.yaml`, so the parsed request can be reviewed before execution.

For a confirm-and-run workflow, first parse, review `parsed_query_preview.md`, then execute the reviewed parse directory:

```bash
inverse-gems run-confirmed-task-query-mock \
  --preview results/openai_parse_forward_volume/ \
  --out results/openai_confirmed_forward_volume_mock/ \
  --db data/openai_confirmed_forward_volume_mock_db/ \
  --confirm-preview
```

To parse and execute in one command with the mock runner:

```bash
inverse-gems run-openai-task-query-mock \
  --request "OPC 40 + slag 30 + fly ash 30, age 0.1 to 360 days, calculate and plot volume vs time" \
  --out results/openai_forward_volume_mock/ \
  --db data/openai_task_mock_db/
```

Example inverse-design request with automatic material-system selection:

```bash
inverse-gems run-openai-task-query-mock \
  --request "Use OPC, slag, and fly ash at 28 days; find low-porosity, high C-A-S-H binders with low OPC." \
  --out results/openai_auto_route_inverse_mock/ \
  --db data/openai_auto_route_inverse_mock_db/ \
  --skip-validation
```

For real xGEMS/GEMS forward calculations or inverse-design validation, use `run-openai-task-query` and provide the same `--dat-lst`, `--gems-class-path`, `--xgems-input-mode`, and retry options used by the deterministic commands. Model defaults live in `configs/openai_task_router.yaml` and can be overridden with `--model`.

Run deterministic inverse query:

```bash
inverse-gems inverse-query \
  --feature-table data/feature_table_early_dense.parquet \
  --query configs/inverse_query.example.yaml \
  --out results/inverse_query_example/
```
