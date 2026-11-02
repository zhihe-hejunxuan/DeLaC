#!/usr/bin/env python3
"""实验二端到端 pipeline：sampler → AE decode → DPA/SUN → cell relax → SUN/RMSD。"""
from __future__ import annotations

import os, sys

import argparse
import json
import os
import subprocess
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PY_FW = Path(os.environ.get("DELAC_PYTHON", sys.executable))
PY_DPA = Path(os.environ.get("DPA_PYTHON", sys.executable))
PY_DDPM = Path(os.environ.get("DDPM_PYTHON", sys.executable))
TRAINABLE = ("empirical_gaussian", "gmm", "rq_nsf", "latent_ddpm")
ALL = (*TRAINABLE[:2], "maf", *TRAINABLE[2:])


def resolve(path: str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else REPO / value


def command(name: str, args: list[str], dry_run: bool, cwd: Path = REPO, env: dict | None = None) -> None:
    print(f"\n{'=' * 72}\n[{name}]\n{'=' * 72}", flush=True)
    print(" ".join(map(str, args)), flush=True)
    if not dry_run:
        run_env = os.environ.copy()
        run_env["PYTHONUNBUFFERED"] = "1"
        if env:
            run_env.update(env)
        subprocess.run(list(map(str, args)), check=True, cwd=str(cwd), env=run_env)


def ensure_ids(initial: Path, relaxed: Path, staging: Path, tag: str) -> tuple[Path, Path]:
    """Preserve explicit IDs, or recover legacy IDs from unchanged input CIFs."""
    import sys
    sys.path.insert(0,str(REPO))
    from record_ids import validate_ids
    first, final = pd.read_csv(initial), pd.read_csv(relaxed)
    key = next((k for k in ("id","candidate_id") if k in first and k in final),None)
    if key is not None:
        validate_ids(first,key,"initial"); validate_ids(final,key,"relaxed")
        first["id"]=first[key].astype(str); final["id"]=final[key].astype(str)
    else:
        if "cif_str_unrelaxed" not in final:
            raise ValueError("Missing stable candidate IDs and original CIFs; cannot infer pairing from row order")
        normalize=lambda x:str(x).replace("\\n","\n").strip()
        original=first.cif_str.map(normalize)
        if original.duplicated().any():
            raise ValueError("Legacy input contains duplicate CIFs; supply explicit candidate IDs")
        first["id"]=[str(i) for i in range(len(first))]
        lookup=dict(zip(original,first.id))
        final["id"]=final.cif_str_unrelaxed.map(normalize).map(lookup)
        validate_ids(final,"id","relaxed")
    if not set(final.id)<=set(first.id): raise ValueError("Relaxed IDs are absent from the initial structures")
    if "cif_str_unrelaxed" in final:
        source=first.set_index("id").cif_str
        normalize=lambda x:str(x).replace("\\n","\n").strip()
        for row in final.itertuples(index=False):
            if normalize(source.loc[row.id])!=normalize(row.cif_str_unrelaxed):
                raise ValueError(f"Wrong initial structure for ID {row.id}")
    staging.mkdir(parents=True,exist_ok=True)
    paths=(staging/f"{tag}.initial.with_id.csv",staging/f"{tag}.relaxed.with_id.csv")
    first.to_csv(paths[0],index=False);final.to_csv(paths[1],index=False)
    return paths


def run_sampler(sampler: str, cfg: dict, config_path: Path, root: Path, args) -> None:
    seed = int(cfg["experiment"]["seed"])
    num_samples = int(cfg["experiment"]["num_samples"])
    models = root / "models" / sampler
    run = root / "runs" / sampler / f"seed_{seed}"
    eval_dir = run / "evaluation"
    model_ext = ".joblib" if sampler in ("empirical_gaussian", "gmm") else ".pt"
    model = resolve(cfg["fixed"]["maf_checkpoint"]) if sampler=="maf" else models / f"model{model_ext}"
    samples = run / "latents" / f"latent_raw_n{num_samples}.pt"
    cifs = run / "generated_cifs"
    initial_csv = eval_dir / "unrelaxed_dpa.csv"
    relaxed_csv = eval_dir / "cell_relaxed.csv"
    sun_initial = eval_dir / "sun_unrelaxed.json"
    sun_relaxed = eval_dir / "sun_relaxed.json"
    compare = eval_dir / "summary.json"

    for directory in (models, samples.parent, cifs, eval_dir):
        directory.mkdir(parents=True, exist_ok=True)

    stages = args.stages
    sampler_python = PY_DDPM if sampler == "latent_ddpm" else PY_FW
    if "train_sample" in stages and (args.force or not samples.is_file()):
        cmd = [
            sampler_python, "-u", HERE / "sampler.py", "--config", config_path,
            "--sampler", sampler, "--mode", "train_sample",
            "--model", model, "--samples", samples,
        ]
        if args.force:
            cmd.append("--force")
        command(f"{sampler}:train_sample", cmd, args.dry_run)

    if "decode" in stages:
        existing = len(list(cifs.glob("*.cif")))
        if args.force or existing < num_samples:
            command(
                f"{sampler}:decode",
                [
                    PY_FW, "-u", "script/runner_for_flow_prep.py",
                    "--mode", "flow_random_gen_maf",
                    "--config_suffix", cfg["fixed"]["ae_config_suffix"],
                    "--dataset", "mp_20", "--model_scale", "large",
                    "--checkpoint", resolve(cfg["fixed"]["ae_checkpoint"]),
                    "--maf_latent_out", samples, "--gen_save_cif_dir", cifs,
                    "--maf_num_samples", str(num_samples), "--seed", str(seed),
                    "--skip_maf_sample",
                ],
                args.dry_run,
            )
        if not args.dry_run:
            decoded = len(list(cifs.glob("*.cif")))
            minimum = float(cfg.get("decode_gate", {}).get("minimum_success_fraction", 0.5))
            fraction = decoded / num_samples
            decode_manifest = {
                "sampler": sampler,
                "requested": num_samples,
                "decoded_cifs": decoded,
                "decode_success_fraction": fraction,
                "minimum_required": minimum,
                "passed": fraction >= minimum,
            }
            (run / "decode_gate.json").write_text(
                json.dumps(decode_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            if fraction < minimum:
                raise RuntimeError(
                    f"{sampler} decode gate 失败: {decoded}/{num_samples}={fraction:.1%} < {minimum:.1%}"
                )

    if "dpa" in stages and (args.force or not initial_csv.is_file()):
        command(
            f"{sampler}:dpa_unrelaxed",
            [
                PY_DPA, "-u", "script/energy_cal_dpa_cli.py", "--cif_dir", cifs,
                "--csv_out", initial_csv, "--model", resolve(cfg["fixed"]["dpa_model"]),
            ],
            args.dry_run,
        )

    if "sun_unrelaxed" in stages and (args.force or not sun_initial.is_file()):
        command(
            f"{sampler}:sun_unrelaxed",
            [
                PY_FW, "-u", "script/eval_sun_from_csv_cli.py", "--csv", initial_csv,
                "--out_json", sun_initial, "--labels_csv", eval_dir / "sun_unrelaxed_labels.csv",
                "--etot_name", "E_tot_pred_dpa_corr", "--no_double_correction",
            ],
            args.dry_run,
        )

    relax = cfg["relaxation"]
    if "relax" in stages and (args.force or not relaxed_csv.is_file()):
        command(
            f"{sampler}:cell_relax",
            [
                PY_DPA, "-u", "script/dpa_cell_relax_csv.py", "--input", initial_csv,
                "--output", relaxed_csv, "--model", resolve(cfg["fixed"]["dpa_model"]),
                "--mode", relax["mode"], "--fmax", str(relax["fmax"]),
                "--max_steps", str(relax["max_steps"]), "--cif_dir", eval_dir / "relaxed_cifs",
                "--resume",
            ],
            args.dry_run,
        )

    if "sun_relaxed" in stages and (args.force or not sun_relaxed.is_file()):
        command(
            f"{sampler}:sun_relaxed",
            [
                PY_FW, "-u", "script/eval_sun_from_csv_cli.py", "--csv", relaxed_csv,
                "--out_json", sun_relaxed, "--labels_csv", eval_dir / "sun_relaxed_labels.csv",
                "--etot_name", "E_tot_pred_dpa_raw",
            ],
            args.dry_run,
        )

    rmsd_summary = eval_dir / "relaxation_metrics" / f"{sampler}.relaxation_metrics.summary.json"
    if "relax_metrics" in stages and (args.force or not rmsd_summary.is_file()):
        if args.dry_run:
            command(f"{sampler}:relax_metrics", ["ftcp_recon.relaxation_metrics"], True)
        else:
            initial_ids, relaxed_ids = ensure_ids(initial_csv, relaxed_csv, eval_dir / "staging", sampler)
            ftcp = resolve(cfg["paths"]["relaxation_metrics_module_root"])
            python_path = str(ftcp)
            if os.environ.get("PYTHONPATH"):
                python_path += os.pathsep + os.environ["PYTHONPATH"]
            command(
                f"{sampler}:relax_metrics",
                [
                    PY_FW, "-m", "ftcp_recon.relaxation_metrics",
                    "--initial", initial_ids, "--relaxed", relaxed_ids,
                    "--output-dir", rmsd_summary.parent, "--tag", sampler,
                    "--id-column", "id", "--max-steps", str(relax["max_steps"]),
                ],
                False, cwd=ftcp, env={"PYTHONPATH": python_path},
            )

    if "summarize" in stages and not args.dry_run:
        keys = (
            "frac_stable_structures", "frac_novel_unique_stable_structures",
            "frac_unique_structures", "frac_novel_structures",
            "avg_energy_above_hull_per_atom", "avg_structure_validity", "avg_comp_validity",
        )
        before = json.loads(sun_initial.read_text()) if sun_initial.is_file() else {}
        after = json.loads(sun_relaxed.read_text()) if sun_relaxed.is_file() else {}
        def values_only(metrics: dict) -> dict:
            return {
                key: (metrics.get(key) or {}).get("value")
                if isinstance(metrics.get(key), dict) else metrics.get(key)
                for key in keys
            }

        before_labels_path = eval_dir / "sun_unrelaxed_labels.csv"
        after_labels_path = eval_dir / "sun_relaxed_labels.csv"
        before_labels = pd.read_csv(before_labels_path) if before_labels_path.is_file() else pd.DataFrame()
        after_labels = pd.read_csv(after_labels_path) if after_labels_path.is_file() else pd.DataFrame()
        before_count = (
            int(before_labels["SUN"].fillna(False).astype(bool).sum())
            if "SUN" in before_labels else 0
        )
        after_count = (
            int(after_labels["SUN"].fillna(False).astype(bool).sum())
            if "SUN" in after_labels else 0
        )
        payload = {
            "sampler": sampler, "seed": seed, "requested": num_samples,
            "fixed_ae": str(resolve(cfg["fixed"]["ae_checkpoint"])),
            "fixed_latents": str(resolve(cfg["fixed"]["normalized_latents"])),
            "unrelaxed": values_only(before),
            "relaxed": values_only(after),
            "sun_counts": {"unrelaxed": before_count, "relaxed": after_count},
            "sun_all": {
                "unrelaxed": before_count / num_samples,
                "relaxed": after_count / num_samples,
            },
            "sun_evaluable": {
                "unrelaxed": before_count / len(before_labels) if len(before_labels) else None,
                "relaxed": after_count / len(after_labels) if len(after_labels) else None,
            },
            "paths": {"model": str(model), "run": str(run), "relaxation_metrics": str(rmsd_summary)},
        }
        compare.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def aggregate(root: Path, cfg: dict) -> None:
    rows = []
    requested = int(cfg["experiment"]["num_samples"])
    seed = int(cfg["experiment"]["seed"])
    for sampler in ALL:
        path = root / "runs" / sampler / f"seed_{seed}" / "evaluation/summary.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "sampler": sampler, "seed": seed, "requested": requested, "status": "completed",
            "sun_unrelaxed_count": payload["sun_counts"]["unrelaxed"],
            "sun_relaxed_count": payload["sun_counts"]["relaxed"],
            "sun_unrelaxed_all": payload["sun_all"]["unrelaxed"],
            "sun_relaxed_all": payload["sun_all"]["relaxed"],
            "sun_unrelaxed_evaluable": payload["sun_evaluable"]["unrelaxed"],
            "sun_relaxed_evaluable": payload["sun_evaluable"]["relaxed"],
            "stable_unrelaxed_evaluable": payload["unrelaxed"]["frac_stable_structures"],
            "stable_relaxed_evaluable": payload["relaxed"]["frac_stable_structures"],
            "unique_relaxed_evaluable": payload["relaxed"]["frac_unique_structures"],
            "novel_relaxed_evaluable": payload["relaxed"]["frac_novel_structures"],
        })
    table = pd.DataFrame(rows)
    table.to_csv(root / "experiment_2_results.csv", index=False)
    (root / "experiment_2_results.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--samplers", default="all", help="all 或逗号分隔")
    parser.add_argument(
        "--stages", default="all",
        help="all 或 train_sample,decode,dpa,sun_unrelaxed,relax,sun_relaxed,relax_metrics,summarize",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = resolve(args.config)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    selected = list(ALL) if args.samplers == "all" else [x.strip() for x in args.samplers.split(",")]
    unknown = sorted(set(selected) - set(ALL))
    if unknown:
        raise SystemExit(f"未知 sampler: {unknown}")
    all_stages = {
        "train_sample", "decode", "dpa", "sun_unrelaxed", "relax",
        "sun_relaxed", "relax_metrics", "summarize",
    }
    args.stages = all_stages if args.stages == "all" else set(args.stages.split(","))
    if not args.stages <= all_stages:
        raise SystemExit(f"未知 stages: {sorted(args.stages - all_stages)}")

    root = resolve(cfg["paths"]["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    protocol = {
        "config": str(config_path), "ae_checkpoint": str(resolve(cfg["fixed"]["ae_checkpoint"])),
        "normalized_latents": str(resolve(cfg["fixed"]["normalized_latents"])),
        "num_samples": cfg["experiment"]["num_samples"], "seed": cfg["experiment"]["seed"],
    }
    if not args.dry_run:
        (root / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    for sampler in selected:
        run_sampler(sampler, cfg, config_path, root, args)
    if "summarize" in args.stages and not args.dry_run:
        aggregate(root, cfg)


if __name__ == "__main__":
    main()
