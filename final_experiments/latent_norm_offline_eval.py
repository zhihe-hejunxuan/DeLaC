#!/usr/bin/env python3
"""Four-arm deterministic-AE reconstruction and latent-modelability audit.

No unconditional crystal generation, DPA evaluation, or relaxation is run.
Gaussian/Z-score statistics are fitted on train+validation only; test is held out.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2
from torch.utils.data import ConcatDataset, Subset
from torch_geometric.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO))

from dataset import MatDataset
from script.ae_test_reconstruction_eval import (
    DefaultDisorderedStructureMatcher,
    _aggregate_model_metrics,
    _build_runner,
    _decode_batch,
    _finalize_model_metrics,
    _gt_structure,
    _new_model_acc,
    _sample_indices,
    _structure_metrics,
)
from script.runner_for_flow_prep import set_global_seed


METHODS = {
    "none": ("flow_prep_none", "result/FlowPrep/mp_20/large/flow_prep_none/model_899.pth"),
    "layernorm": ("flow_prep_zscore", "result/FlowPrep/mp_20/large/flow_prep_zscore/model_899.pth"),
    "rmsnorm": ("rmsnorm", "result/FlowPrep/mp_20/large/rmsnorm/model_899.pth"),
    "scaled_rmsnorm": ("h_rmsnorm", "result/FlowPrep/mp_20/large/h_rmsnorm/model_899.pth"),
}


def dataset(split: str, conf: dict) -> MatDataset:
    path = REPO / "data/mp_20" / f"{split}.pt"
    return MatDataset(str(path), **conf["data"])


@torch.no_grad()
def encode(model, ds, batch_size: int, device: str) -> torch.Tensor:
    out = []
    for batch in DataLoader(ds, batch_size=batch_size, shuffle=False):
        out.append(model.build_flow_data(batch.to(device)).float().cpu())
    return torch.cat(out)


def latent_metrics(trainval: torch.Tensor, test: torch.Tensor, jitter: float = 1e-6):
    x = trainval.double().numpy()
    y = test.double().numpy()
    mean = x.mean(0)
    std = x.std(0, ddof=0)
    std = np.where(std < 1e-12, 1.0, std)
    xz, yz = (x - mean) / std, (y - mean) / std
    cov = np.cov(xz, rowvar=False)
    cov.flat[:: cov.shape[0] + 1] += jitter
    eig = np.linalg.eigvalsh(cov)
    eig_clip = np.clip(eig, 1e-15, None)
    p = eig_clip / eig_clip.sum()
    effective_rank = float(np.exp(-(p * np.log(p)).sum()))
    condition = float(eig_clip.max() / eig_clip.min())
    corr = np.corrcoef(xz, rowvar=False)
    tri = np.abs(corr[np.triu_indices_from(corr, k=1)])

    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise RuntimeError("regularized covariance is not positive definite")
    solved = np.linalg.solve(cov, yz.T).T
    mahal2 = np.einsum("ij,ij->i", yz, solved)
    d = yz.shape[1]
    nll = 0.5 * (d * math.log(2 * math.pi) + logdet + mahal2)
    cover = {f"mahalanobis_coverage_{q}": float(np.mean(mahal2 <= chi2.ppf(q / 100, d)))
             for q in (50, 90, 95, 99)}
    metrics = {
        "dimension": int(d),
        "fit_samples_train_plus_val": int(len(x)),
        "heldout_test_samples": int(len(y)),
        "raw_l2_norm_mean_trainval": float(np.linalg.norm(x, axis=1).mean()),
        "raw_l2_norm_std_trainval": float(np.linalg.norm(x, axis=1).std()),
        "test_standardized_mean_abs": float(np.abs(yz.mean(0)).mean()),
        "test_standardized_std_rmse_from_one": float(np.sqrt(np.mean((yz.std(0) - 1) ** 2))),
        "covariance_condition_number": condition,
        "effective_rank": effective_rank,
        "mean_abs_offdiag_correlation": float(tri.mean()),
        "heldout_gaussian_nll_mean": float(nll.mean()),
        "heldout_gaussian_nll_per_dim": float(nll.mean() / d),
        "heldout_mahalanobis2_mean": float(mahal2.mean()),
        **cover,
    }
    artifacts = {
        "mean": torch.from_numpy(mean).float(),
        "std": torch.from_numpy(std).float(),
        "covariance_standardized": torch.from_numpy(cov).float(),
        "test_mahalanobis2": torch.from_numpy(mahal2).float(),
    }
    return metrics, artifacts


@torch.no_grad()
def ae_metrics(runner, test_ds, batch_size: int, device: str):
    acc = _new_model_acc()
    totals = {"length_abs": 0.0, "angle_abs": 0.0, "volume_rel": 0.0, "n": 0}
    for batch in DataLoader(test_ds, batch_size=batch_size, shuffle=False):
        batch = batch.to(device)
        loss = runner.model.forward_ae(batch, eval=True, global_step=0)
        _aggregate_model_metrics(loss, acc)
        h = runner.model.encoder_backbone(batch)
        if getattr(runner.model, "h_pre_norm_flag", False):
            h = runner.model.h_pre_norm(h)
        la = torch.cat([batch.scaled_lengths, batch.angles], -1)
        if runner.model.lattice_normalizer is not None:
            la = runner.model.lattice_normalizer.transform(la)
        z_latt = runner.model.fc_lattice_enc(la)
        _, _, pred_l, pred_a = runner.model.ae_decode_from_h(h, z_latt)
        true_l, true_a = batch.lengths, batch.angles
        totals["length_abs"] += (pred_l - true_l).abs().sum().item()
        totals["angle_abs"] += (pred_a - true_a).abs().sum().item()
        # volume is robustly available from lengths/angles without CIF reparsing
        def volume(l, a):
            ar = torch.deg2rad(a)
            ca, cb, cg = torch.cos(ar[:, 0]), torch.cos(ar[:, 1]), torch.cos(ar[:, 2])
            rad = torch.clamp(1 + 2*ca*cb*cg - ca.square() - cb.square() - cg.square(), min=0)
            return l.prod(-1) * torch.sqrt(rad)
        pv, tv = volume(pred_l, pred_a), volume(true_l, true_a)
        totals["volume_rel"] += ((pv-tv).abs()/tv.clamp_min(1e-8)).sum().item()
        totals["n"] += int(len(pred_l))
    out = _finalize_model_metrics(acc)
    out.update({
        "length_mae_angstrom": totals["length_abs"]/(3*totals["n"]),
        "angle_mae_degree": totals["angle_abs"]/(3*totals["n"]),
        "volume_relative_error_mean": totals["volume_rel"]/totals["n"],
    })
    return out


@torch.no_grad()
def structure_reconstruction(runner, test_ds, indices, batch_size: int, device: str, seed: int):
    set_global_seed(seed)
    matcher = DefaultDisorderedStructureMatcher()
    rows, offset = [], 0
    loader = DataLoader(Subset(test_ds, indices), batch_size=batch_size, shuffle=False)
    for batch in loader:
        ids = indices[offset:offset + int(batch.num_graphs)]
        offset += int(batch.num_graphs)
        preds = _decode_batch(runner, batch.to(device), decode_mode="default")
        for idx, pred in zip(ids, preds):
            gt = _gt_structure(test_ds.data_dict_list[idx])
            row = {"index": idx, **_structure_metrics(gt, pred, matcher), "rmsd_angstrom": None}
            if gt is not None and pred is not None:
                try:
                    rms = matcher.get_rms_dist(gt, pred)
                    row["rmsd_angstrom"] = None if rms is None else float(rms[0])
                except Exception:
                    pass
            rows.append(row)
    frame = pd.DataFrame(rows)
    summary = {f"structure_{c}_rate": float(frame[c].dropna().astype(float).mean())
               for c in ["parse_ok","composition_match","comp_match","lattice_match","atomic_position_match","full_match"]}
    vals = pd.to_numeric(frame["rmsd_angstrom"], errors="coerce").dropna()
    summary["structure_rmsd_mean_angstrom_matched"] = float(vals.mean()) if len(vals) else None
    summary["structure_rmsd_median_angstrom_matched"] = float(vals.median()) if len(vals) else None
    summary["structure_reconstruction_samples"] = len(frame)
    return summary, frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--structure-samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--structure-only", action="store_true",
                    help="Reuse completed AE/latent summaries and rerun only coordinate reconstruction")
    ap.add_argument("--out", default="result/final_experiments/latent_norm_offline_eval")
    args = ap.parse_args()
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    methods = [x.strip() for x in args.methods.split(',') if x.strip()]
    summaries = []
    for name in methods:
        t0 = time.time()
        suffix, ckpt = METHODS[name]
        group = out / name
        group.mkdir(exist_ok=True)
        print(f"[START] {name}", flush=True)
        runner, conf, ckpt_path = _build_runner(suffix, "mp_20", "large", "", str(REPO/ckpt), args.device)
        test = dataset("test", conf)
        if args.structure_only:
            summary_path = group / "summary.json"
            if not summary_path.is_file():
                raise FileNotFoundError(f"--structure-only requires {summary_path}")
            summary = json.loads(summary_path.read_text())
        else:
            train, val = dataset("train", conf), dataset("val", conf)
            trainval = ConcatDataset([train, val])
            z_trainval = encode(runner.model, trainval, args.batch_size, args.device)
            z_test = encode(runner.model, test, args.batch_size, args.device)
            lm, latent_art = latent_metrics(z_trainval, z_test)
            torch.save({**latent_art, "checkpoint": str(ckpt_path), "trainval_shape": list(z_trainval.shape),
                        "test_shape": list(z_test.shape)}, group/"latent_gaussian_audit.pt")
            am = ae_metrics(runner, test, args.batch_size, args.device)
            summary = {"method": name, "config_suffix": suffix, "checkpoint": str(ckpt_path),
                       "split_protocol": "Gaussian and Z-score fit on train+validation; test held out",
                       "ae_test": am, "latent_modelability": lm}
        indices = _sample_indices(args.structure_samples, len(test), args.seed)
        sm, rows = structure_reconstruction(runner, test, indices, args.batch_size, args.device, args.seed)
        rows.to_csv(group/"structure_reconstruction_per_sample.csv", index=False)
        summary["structure_reconstruction"] = sm
        summary["elapsed_seconds"] = time.time()-t0
        (group/"summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        summaries.append(summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        del runner, test
        torch.cuda.empty_cache()
    flat=[]
    for s in summaries:
        flat.append({"method":s["method"], **s["ae_test"], **s["latent_modelability"],
                     **s["structure_reconstruction"], "elapsed_seconds":s["elapsed_seconds"]})
    pd.DataFrame(flat).to_csv(out/"summary.csv", index=False)
    (out/"summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False))
    print(f"[DONE] {out/'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
