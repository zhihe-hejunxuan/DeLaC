#!/usr/bin/env python3
"""Freeze the deployment I3 + T2 + HGB rejector without formal seeds."""
from __future__ import annotations

import os, sys

import argparse, hashlib, json, math, sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import filter_ablation_lib as lib
import filter_dataset as base

DEFAULT_OUT = REPO / "result/final_experiments/experiment_speed/frozen_i3_t2_hgb"
TRAIN_SEEDS = (0, 1, 7, 21, 123)
CALIBRATION_SEED = 44
FEATURES = lib.FULL + ("smact_valid",)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--stable-recall", type=float, default=.99)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = args.output_dir.resolve()
    artifact_path = out / "i3_t2_hgb.joblib"
    manifest_path = out / "manifest.json"
    if artifact_path.is_file() and manifest_path.is_file() and not args.force:
        print(json.dumps(json.loads(manifest_path.read_text()), indent=2, ensure_ascii=False))
        return
    out.mkdir(parents=True, exist_ok=True)

    feature_cache = pd.read_csv(lib.FEATURE_CACHE, low_memory=False)
    parts = []
    for split in ("train", "calibration", "development_test"):
        raw = base.load_split(lib.DATA, split)
        feat = feature_cache.loc[feature_cache.source_split.eq(split),
                                 ["dataset_id", *lib.FULL, "smact_valid"]]
        parts.append(raw.merge(feat, on="dataset_id", validate="one_to_one"))
    data = pd.concat(parts, ignore_index=True)
    formal = sorted(set(pd.to_numeric(data.source_seed).astype(int)) & {13, 37, 42, 73})
    if formal:
        raise RuntimeError(f"formal seed leakage: {formal}")
    train = data.loc[data.label_valid & data.source_seed.isin(TRAIN_SEEDS)].copy()
    cal = data.loc[data.label_valid & data.source_seed.eq(CALIBRATION_SEED)].copy()
    if set(train.source_seed.unique()) != set(TRAIN_SEEDS) or cal.empty:
        raise RuntimeError("frozen train/calibration seed split is incomplete")

    q75 = float(pd.to_numeric(train.relax_steps, errors="coerce").quantile(.75))
    medians = train[list(FEATURES)].apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan).median().fillna(0.)
    x_train = lib.matrix(train, FEATURES, medians)
    stable = lib.truth(train.stable)
    long_relax = lib.long_y(train, q75).astype(bool)
    bad = (~stable) & long_relax
    weights = np.where(bad, 1 + pd.to_numeric(train.relax_steps, errors="coerce").fillna(300).to_numpy() / 300, 1.)
    model = lib.model("HistGradientBoosting", 2026)
    model.fit(x_train, bad.astype(int), sample_weight=weights)
    cal_risk = model.predict_proba(lib.matrix(cal, FEATURES, medians))[:, 1]
    choice = lib.threshold(cal_risk, cal, args.stable_recall)
    if choice is None:
        raise RuntimeError("no feasible calibration threshold")

    payload = {
        "model": model, "features": list(FEATURES), "medians": medians.to_dict(),
        "threshold": float(choice["value"]), "q75_relax_steps": q75,
        "configuration": "I3_full_smact__T2_weighted_joint_bad__HistGradientBoosting",
    }
    joblib.dump(payload, artifact_path)
    deployment_spec = {
        "features": list(FEATURES), "medians": medians.to_dict(),
        "threshold": float(choice["value"]), "artifact": artifact_path.name,
        "artifact_sha256": sha256(artifact_path),
        "scoring_python": os.environ.get("DELAC_PYTHON", sys.executable),
    }
    atomic(deployment_spec, out / "deployment_spec.json")
    manifest = {
        "status": "frozen", "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "artifact": str(artifact_path), "artifact_sha256": sha256(artifact_path),
        "formal_seeds_excluded": [13, 37, 42, 73], "train_seeds": list(TRAIN_SEEDS),
        "calibration_seed": CALIBRATION_SEED, "train_rows": len(train),
        "calibration_rows": len(cal), "features": list(FEATURES),
        "target": "T2_weighted_joint_bad", "classifier": repr(model),
        "q75_relax_steps": q75, "stable_recall_floor": args.stable_recall,
        "threshold": choice, "data_root": str(lib.DATA),
        "feature_cache": str(lib.FEATURE_CACHE),
        "deployment_spec": str(out / "deployment_spec.json"),
    }
    atomic(manifest, manifest_path)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
