# Running the DeLaC experiments

Run commands from the repository root unless shown otherwise. The order below is shared setup, **Table 1**, **Table 4**, other main-text experiments, then supplementary experiments. Weights and pre-fitted model calibration files are not included. Published experimental outputs are confined to `data/generated/`; `data/dpa_validation/` contains reference inputs only. Use new output directories outside the release checkout and preserve historical runs. The relative `result/...` paths below are example working paths for a local experiment copy, not files to upload.

## 1. Shared setup

### Environments

Use Linux, Python 3.10, an NVIDIA GPU, and separate environments. Run only the matching requirements command in each environment:

```bash
# DeLaC, latent samplers, FTCP, and offline aggregation
python -m pip install -r requirements.txt
# DPA inference/relaxation
python -m pip install -r requirements/dpa.txt
# One external-baseline environment per file
python -m pip install -r requirements/cdvae.txt
python -m pip install -r requirements/diffcsp.txt
python -m pip install -r requirements/mattergen.txt
python -m pip install -r requirements/symat.txt
```

Requirements specify CUDA 12.8 PyTorch/PyG wheel indexes. Do not combine baseline files in one environment. Latent DDPM uses the DeLaC environment. The launchers import the retained baseline source directly; additional upstream editable installations are unnecessary.

```bash
export DELAC_ROOT="$PWD"
export PYTHONPATH="$DELAC_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export DELAC_PYTHON=/path/to/delac/bin/python
export DPA_PYTHON=/path/to/dpa/bin/python
export DDPM_PYTHON="$DELAC_PYTHON"
export CDVAE_PYTHON=/path/to/cdvae/bin/python
export DIFFCSP_PYTHON=/path/to/diffcsp/bin/python
export MATTERGEN_PYTHON=/path/to/mattergen/bin/python
export SYMAT_PYTHON=/path/to/symat/bin/python
export MPLCONFIGDIR="${TMPDIR:-/tmp}/delac-matplotlib"
```

### MP-20 and evaluation resources

Obtain MP-20 from the [MatterGen data release](https://github.com/microsoft/mattergen/tree/main/data-release/mp-20); the source revision is in `data/mp20/SOURCE.json`. In a separate MatterGen checkout with Git LFS:

```bash
git lfs pull -I data-release/mp-20/ --exclude=""
unzip data-release/mp-20/mp_20.zip -d datasets
git lfs pull -I data-release/alex-mp/reference_MP2020correction.gz --exclude=""
```

Back in this repository, retain the published split assignments:

```bash
"$DELAC_PYTHON" tools/prepare_mp20.py --csv-dir /path/to/mattergen/datasets/mp_20
"$DELAC_PYTHON" compute_score_norm.py \
  --train_path data/mp_20/train.pt --val_path data/mp_20/val.pt \
  --test_path data/mp_20/test.pt --output data/mp_20/new_score_norm.txt
export DELAC_REFERENCE=/path/to/mattergen/data-release/alex-mp/reference_MP2020correction.gz
```

Preparation checks the processed split sizes and stores original CSVs and graph caches in `data/mp_20/`. Obtain DPA-3.1-MPtrj separately and place it at `mlff_choosing/dpa-3.1-mptrj.pth`. SUN evaluation requires the full reference hull; the small validation dataset is not a substitute. `configs/external_artifacts.json` lists the historical checkpoint paths and checksums. Restore matching checkpoints for their historical outputs, or train the local models below.

## 2. Table 1: generator training and Raw quality evaluation

### 2.1 DeLaC training

Train only the ScaledRMSNorm configuration for the main experiments, export train+validation latents, then fit Gaussian and MAF:

```bash
"$DELAC_PYTHON" script/runner_for_flow_prep.py --mode train --config_suffix h_rmsnorm --seed 42
"$DELAC_PYTHON" script/runner_for_flow_prep.py --mode build_flow_data \
  --config_suffix h_rmsnorm --checkpoint result/FlowPrep/mp_20/large/h_rmsnorm/model_899.pth \
  --build_flow_out data/mp_20/norm_flow_feature_h_rmsnorm_ep899.pt
"$DELAC_PYTHON" final_experiments/experiment_2/sampler.py \
  --sampler empirical_gaussian --mode train \
  --model result/final_experiments/latent_sampler_h_rmsnorm/models/empirical_gaussian/model.joblib
"$DELAC_PYTHON" final_experiments/experiment_2/sampler.py \
  --sampler maf --mode train --model result/MAF/no_cond/mp_20/h_rmsnorm/0.0003/512/model_best.pt
```

The AE uses 900 epochs and batch size 256. Latent extraction and density fitting exclude the test split. Sampler settings are in `final_experiments/experiment_2/config.yaml`. `--mode train` does not sample candidates. Existing density artifacts are reused unless `--force` is specified.

### 2.2 External generators

Source revisions are in `final_experiments/experiment_3/source_manifest.json`.

Before using CDVAE or MatterGen, copy its upstream GemNet `gemnet-dT.json` from the recorded source revision into the same relative path under its retained vendor directory. These pre-fitted scaling parameters are external artifacts, alongside checkpoints. Exact source paths, target paths and checksums are listed in `configs/external_artifacts.json`; the benchmark preflight checks for them.

**SyMat:** train with `(cd SyMat && "$SYMAT_PYTHON" train.py --dataset mp_20)`. The expected checkpoint is `SyMat/result/mp_20/model_299.pth`.

**CDVAE:** restore the archived 145-epoch checkpoint or train with the retained MP-20 source. The recorded configuration is `configs/cdvae_mp20_training.yaml`.

```bash
cd final_experiments/experiment_3/vendor/cdvae-official
export PROJECT_ROOT="$PWD"
export HYDRA_JOBS="$DELAC_ROOT/result/cdvae_training_jobs"
export WABDB_DIR="$DELAC_ROOT/result/cdvae_training_logs"
"$CDVAE_PYTHON" cdvae/run.py data=mp_20 expname=cdvae_mp20 \
  data.root_path="$DELAC_ROOT/data/mp_20" train.pl_trainer.max_epochs=145 \
  train.model_checkpoints.save_top_k=-1 logging.wandb.mode=disabled
cd "$DELAC_ROOT"
```

Copy the chosen `epoch=144-step=15370.ckpt` and its `hparams.yaml`, `lattice_scaler.pt`, and `prop_scaler.pt` from the same Hydra training run into `result/final_experiments/experiment_3/training/cdvae_mp20/` before benchmarking.

**DiffCSP:** obtain the official MP-20 generation directory with `last.ckpt` and `hparams.yaml`, and place it at `final_experiments/experiment_3/vendor/diffcsp-official/pretrained/mp_gen/`. Its recorded configuration is `configs/diffcsp_mp20_training.yaml`.

**MatterGen:** place the official `mp_20_base` directory, including `config.yaml` and `checkpoints/last.ckpt`, at `final_experiments/experiment_3/vendor/mattergen-official/checkpoints/mp_20_base/`. Keep the retained source and `sampling_conf/default.yaml` for runtime sampling.

**FTCP:** use the original MP-20 CSVs and the retained PyTorch implementation:

```bash
cd third_party/FTCP
"$DELAC_PYTHON" -m ftcp_recon prepare \
  --train-input ../../data/mp_20/train.csv --val-input ../../data/mp_20/val.csv \
  --test-input ../../data/mp_20/test.csv --structure-column cif --niggli \
  --scaler-scope train --max-elms 7 --property-columns formation_energy_per_atom band_gap \
  --output artifacts/mp20_niggli_train_scaler.npz
"$DELAC_PYTHON" -m ftcp_recon train --cache artifacts/mp20_niggli_train_scaler.npz \
  --output runs/mp20_ablation_B_niggli_properties --seed 42 --amp
cd ../..
```

The benchmark uses that run's `best.pt`, preprocessing cache, and fixed element descriptors in `third_party/FTCP/data/`. Recorded training arguments are in `configs/ftcp_training.json`.

### 2.3 Raw generation, relaxation, and evaluation

Audit the provided resources first, then launch the three-seed runs:

```bash
"$DELAC_PYTHON" final_experiments/experiment_speed/run_delac_four_arm.py --arms raw --audit-only
"$DELAC_PYTHON" final_experiments/experiment_speed/run_remaining_models_four_arm.py --arms raw --audit-only
"$DELAC_PYTHON" final_experiments/experiment_speed/run_delac_four_arm.py \
  --arms raw --seeds 13,37,73 --gpu 0 --output-root result/table1/delac
"$DELAC_PYTHON" final_experiments/experiment_speed/run_remaining_models_four_arm.py \
  --arms raw --seeds 13,37,73 --gpu 0 --output-root result/table1/baselines
```

Raw-only execution does not need a filter or replacement candidates. Each generator receives 1,000 requests per seed. The external controller covers the six other generators, including DeLaC (MAF); `--methods` selects a subset.

Both controllers print a `run_...` directory. For each method/seed, the files are `stage1/raw/selected_1000.csv`, `raw/ordinary_workers_2_raw/relaxed_merged.csv`, `raw/sun_relaxed.json`, and `raw/sun_relaxed_labels.csv`. Failed requests remain in the SUN/all denominator. The common protocol uses generation batch size 32, two DPA workers, FIRE/FrechetCellFilter, `fmax=0.05`, 300 steps, and a 3,600-second watchdog. Offline SUN time is excluded from E2E time.

Substitute the printed directories to aggregate all seven generators:

```bash
"$DELAC_PYTHON" tools/summarize_core_tables.py \
  --delac-run result/table1/delac/run_YYYYMMDD_HHMMSS \
  --baseline-run result/table1/baselines/run_YYYYMMDD_HHMMSS --output-dir result/table1/summary
```

The command writes per-seed quality/timing records and the means/sample standard deviations for Tables 1 and 2. CSVs retain full precision; round at presentation. Released Gaussian Raw records can also be read directly from `data/generated/13/`, `37/`, and `73/`: `labels.csv` links to available final CIFs under `cifs/`. Blank component labels mean unavailable original labels, not negative labels. `SOURCE.json` documents within-seed uniqueness and failure status.

### 2.4 Recover a SUN audit that reached its global timeout

The paper's CDVAE seed-13 results use the retained per-chemical-system hull watchdog. If the controller records `sun_timeout`, preserve its completed relaxation and evaluate that CSV separately:

```bash
"$DELAC_PYTHON" final_experiments/diagnostics/cdvae_seed13_sun_timeout/evaluate_sun_per_chemsys_timeout.py \
  --csv /path/to/cdvae/seed_13/raw/ordinary_workers_2_raw/relaxed_merged.csv \
  --output-dir result/sun_recovery/cdvae13_raw --per-chemsys-timeout-seconds 900
```

This writes labels, metrics, and a hull-timeout audit without repeating relaxation. To aggregate a recovery stored outside the original arm directory, create a JSON mapping and pass `--sun-overrides /path/to/sun_overrides.json` to `summarize_core_tables.py`:

```json
{
  "cdvae/13/raw": "/absolute/path/to/result/sun_recovery/cdvae13_raw"
}
```

For a recovered Learned arm, add `cdvae/13/learned_filter` pointing to that arm's recovery directory. The aggregator checks the labels against the original relaxed candidate IDs and CIFs; a missing or mismatched recovery is not silently replaced by another result.

## 3. Table 4: filter training and paired evaluation

### 3.1 Collect development labels and fit the selected filter

Prepare all generators in Section 2 first. The development collection uses seeds 0, 1, 7, 21, 44, and 123:

```bash
"$DELAC_PYTHON" final_experiments/experiment_speed/run_universal_filter_dataset_collection.py \
  --output-root result/filter_collection --gpu 0
export DELAC_FILTER_DATA="$DELAC_ROOT/result/filter_collection/run_YYYYMMDD_HHMMSS"
export DELAC_FILTER_FEATURES="$DELAC_ROOT/result/filter_features/pre_relax_features.csv"
"$DELAC_PYTHON" tools/prepare_filter_features.py \
  --data-root "$DELAC_FILTER_DATA" --output "$DELAC_FILTER_FEATURES"
"$DELAC_PYTHON" final_experiments/experiment_speed/freeze_i3_t2_hgb.py
```

Replace the placeholder with the collected run. Freezing fits the selected I3/T2/HGB configuration on seeds 0, 1, 7, 21, and 123 and calibrates on 44. Keep `i3_t2_hgb.joblib`, `deployment_spec.json`, and `manifest.json` together under `result/final_experiments/experiment_speed/frozen_i3_t2_hgb/`. The supplementary search below is not needed to fit this already selected configuration.

### 3.2 Paired Raw/Learned evaluation

```bash
"$DELAC_PYTHON" final_experiments/experiment_speed/run_delac_four_arm.py --arms raw,learned_filter --audit-only
"$DELAC_PYTHON" final_experiments/experiment_speed/run_remaining_models_four_arm.py --arms raw,learned_filter --audit-only
"$DELAC_PYTHON" final_experiments/experiment_speed/run_delac_four_arm.py \
  --arms raw,learned_filter --seeds 13,37,73 --gpu 0 --output-root result/table4/delac
"$DELAC_PYTHON" final_experiments/experiment_speed/run_remaining_models_four_arm.py \
  --arms raw,learned_filter --seeds 13,37,73 --gpu 0 --output-root result/table4/baselines
"$DELAC_PYTHON" tools/summarize_core_tables.py \
  --delac-run result/table4/delac/run_YYYYMMDD_HHMMSS \
  --baseline-run result/table4/baselines/run_YYYYMMDD_HHMMSS \
  --include-learned --output-dir result/table4/summary
```

Paired execution prepares the shared initial/refill pools for all four policies; `--arms` chooses the independent relaxations. Learned screening replenishes rejected proposals until 1,000 are admitted. Throughput is mean SUN count divided by mean E2E minutes. The paired run's Raw outputs also supply Tables 1 and 2.

Use `--resume-run /path/to/the/run` to resume a paired run. A Raw-only run cannot be promoted to a filtered run: start a fresh paired run to prepare its shared refill pool.

## 4. Other main-text experiments

### Table 2 timing

Reuse a Table 1 or Table 4 run and `tools/summarize_core_tables.py`. Stage 1 includes generator setup, model loading, inference, conversion, selection, and writing. Stage 2 is corrected two-worker relaxation wall time including required serial retries; SUN evaluation is separate.

### Table 3: fixed-interface latent samplers

After Section 2.1, run the five-sampler seed-42 ablation:

```bash
"$DELAC_PYTHON" final_experiments/experiment_2/run_pipeline.py --dry-run
"$DELAC_PYTHON" final_experiments/experiment_2/run_pipeline.py
```

`experiment_2/config.yaml` fixes the decoder, normalization, density settings, and output root. The pipeline fits/samples, decodes, evaluates energies, relaxes, and evaluates SUN. `--samplers`/`--stages` select individual stages; `--force` requests refitting. MAF uses the same checkpoint as Section 2.1.

### Raw counterfactual cost analysis

Use measured relaxation output and SUN labels from the same seed/run:

```bash
export RAW_ARM=/path/to/a/completed/run/seed_13/raw
"$DELAC_PYTHON" tools/analyze_counterfactual.py \
  --raw-relaxed "$RAW_ARM/ordinary_workers_2_raw/relaxed_merged.csv" \
  --sun-labels "$RAW_ARM/sun_relaxed_labels.csv" --id-column candidate_id \
  --artifact result/final_experiments/experiment_speed/frozen_i3_t2_hgb/i3_t2_hgb.joblib \
  --output-dir result/counterfactual_seed13
```

Repeat for 37 and 73. The command uses actual Raw costs and original SUN outcomes, including proposals the filter would reject. Candidate IDs and matching final CIFs are checked before joining labels.

## 5. Supplementary experiments

### Frozen normalization: Supplementary Table S1 and Fig. S2

Keep the main ScaledRMSNorm checkpoint; train the other three configurations:

```bash
"$DELAC_PYTHON" script/runner_for_flow_prep.py --mode train --config_suffix flow_prep_none --seed 42
"$DELAC_PYTHON" script/runner_for_flow_prep.py --mode train --config_suffix flow_prep_zscore --seed 42
"$DELAC_PYTHON" script/runner_for_flow_prep.py --mode train --config_suffix rmsnorm --seed 42
"$DELAC_PYTHON" final_experiments/latent_norm_offline_eval.py \
  --batch-size 64 --structure-samples 1000 --seed 42 --out result/normalization_diagnostics
"$DELAC_PYTHON" tools/paired_reconstruction.py \
  --none result/normalization_diagnostics/none/structure_reconstruction_per_sample.csv \
  --scaled-rmsnorm result/normalization_diagnostics/scaled_rmsnorm/structure_reconstruction_per_sample.csv \
  --output result/normalization_diagnostics/paired_statistics.json
```

`flow_prep_zscore` is the historical name of LayerNorm. This diagnostic uses frozen checkpoints and held-out structures without physical relaxation. Bootstrap settings are exposed by `paired_reconstruction.py --help`.

### Filter ablations: Supplementary Tables S2–S3

Reuse the development data/features from Section 3.1:

```bash
export DELAC_FILTER_DATA=/absolute/path/to/the/development/run
export DELAC_FILTER_FEATURES=/absolute/path/to/pre_relax_features.csv
"$DELAC_PYTHON" final_experiments/experiment_speed/run_hgb_factorial_top3_model_selection.py
```

This performs the seed-isolated feature/target factorial and classifier comparisons. Each fold computes its cost threshold from its own training seeds. Final fitting/calibration remains the separate freeze command.

### Four policies: Supplementary Tables S4–S5

Resume the paired Table 4 runs for the remaining arms:

```bash
"$DELAC_PYTHON" final_experiments/experiment_speed/run_delac_four_arm.py \
  --resume-run /path/to/table4/delac/run --arms rule_filter,rule_union_learned
"$DELAC_PYTHON" final_experiments/experiment_speed/run_remaining_models_four_arm.py \
  --resume-run /path/to/table4/baselines/run --arms rule_filter,rule_union_learned
```

Alternatively, omit `--arms` on a fresh run to relax all four policies independently.

Aggregate the completed four-policy runs with `tools/summarize_core_tables.py --delac-run /path/to/delac/run --baseline-run /path/to/baselines/run --all-arms --output-dir result/four_policy_summary`. If a SUN recovery was needed, pass `--sun-overrides` as in Section 2.4; Rule and Union keys use `rule_filter` and `rule_union_learned`. This also writes `supplementary_four_policy_summary.csv`.

### Complete quality and relaxation diagnostics: Supplementary Table S6

Complete quality metrics are already in each Raw arm's `sun_relaxed.json`. For paired structure diagnostics:

```bash
export SEED_RUN=/path/to/a/completed/run/seed_13
"$DELAC_PYTHON" final_experiments/experiment_3/run_relaxation_metrics_resumable.py \
  --initial "$SEED_RUN/stage1/raw/selected_1000.csv" \
  --relaxed "$SEED_RUN/raw/ordinary_workers_2_raw/relaxed_merged.csv" \
  --id-column candidate_id --tag raw_seed13 --output-dir result/relaxation_diagnostics
```

### Archived representative pairs: Supplementary Fig. S3

The illustrated pairs belong to `experiment_3_seed_extension/scaled_rmsnorm_empirical_gaussian/seed_13`, not the Table 1 four-arm Raw archive. These supplementary outputs are not bundled. Supply the original archived label CSV from that run:

```bash
"$DELAC_PYTHON" tools/export_representatives.py \
  --sun-labels /path/to/experiment_3_seed_extension/scaled_rmsnorm_empirical_gaussian/seed_13/sun_relaxed_labels.csv \
  --output-dir result/representative_structures
```

The tool checks original IDs and compositions before exporting all six CIFs. Alternatively, `--data-dir /path/to/external/pairs` reads an external directory with `pairs.csv` and its referenced CIFs. A new run can produce different structures at those IDs; Table 1 records cannot substitute for these exact illustrated examples. Keep exported pairs outside the upload checkout.

### Fixed DPA-to-hull validation: Supplementary Tables S7–S9

The fixed CIFs and authoritative hull labels are in `data/dpa_validation/`; provenance is in its `SOURCE.json`.

```bash
"$DPA_PYTHON" final_experiments/dpa_mptrj_validation/run_validation.py \
  --data-dir data/dpa_validation --model mlff_choosing/dpa-3.1-mptrj.pth \
  --output-dir result/dpa_validation
```

This runs single-point DPA energies without relaxation and applies MP2020 correction once. No material selection is performed. `--resume` requires matching checkpoint, labels, and CIF contents.
