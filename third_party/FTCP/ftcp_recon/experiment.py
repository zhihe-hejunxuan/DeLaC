from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from pymatgen.analysis.structure_matcher import StructureMatcher
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import MatrixDataset, load_structures
from .model import FTCPVAE
from .representation import FTCPCodec, FTCPConfig


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    if name == "auto": name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu for a basic test")
    return torch.device(name)


def split_indices(n: int, seed: int, train_fraction=.8, val_fraction=.1):
    order = np.random.default_rng(seed).permutation(n)
    n_train, n_val = int(n * train_fraction), int(n * val_fraction)
    return order[:n_train], order[n_train:n_train+n_val], order[n_train+n_val:]


def prepare(args):
    if args.train_input and (not args.val_input or not args.test_input):
        raise ValueError("--train-input requires --val-input and --test-input")
    codec = FTCPCodec(FTCPConfig(args.max_sites, args.max_elms, pad_rows=args.pad_rows,
                                 niggli=args.niggli), args.assets)
    if args.train_input:
        bundles = [load_structures(p, args.structure_column)
                   for p in (args.train_input, args.val_input, args.test_input)]
        ids = sum((b[0] for b in bundles), [])
        structures = sum((b[1] for b in bundles), [])
        import pandas as pd
        frame = pd.concat([b[2] for b in bundles], ignore_index=True)
        source_split = np.concatenate([np.full(len(b[0]), i, np.int8) for i, b in enumerate(bundles)])
    else:
        ids, structures, frame = load_structures(args.input, args.structure_column)
        source_split = None
    matrices, kept_ids, kept_rows, kept_split, rejected = [], [], [], [], []
    for i, (sid, structure) in enumerate(tqdm(zip(ids, structures), total=len(ids), desc="FTCP")):
        try:
            matrices.append(codec.encode(structure)); kept_ids.append(sid); kept_rows.append(i)
            if source_split is not None: kept_split.append(source_split[i])
        except Exception as exc:
            rejected.append({"id": sid, "reason": str(exc)})
    if not matrices: raise RuntimeError("no supported structures were encoded")
    raw = np.stack(matrices).astype(np.float32)
    if source_split is None:
        train, val, test = split_indices(len(raw), args.seed, args.train_fraction, args.val_fraction)
    else:
        kept_split = np.asarray(kept_split)
        train, val, test = (np.flatnonzero(kept_split == i) for i in range(3))
    # Original FTCP min-max scales every matrix row over samples and columns.
    scaler_indices = np.arange(len(raw)) if args.scaler_scope == "all" else train
    row_min = raw[scaler_indices].min(axis=(0, 2), keepdims=True)
    row_max = raw[scaler_indices].max(axis=(0, 2), keepdims=True)
    scale = np.maximum(row_max - row_min, 1e-8)
    norm = np.clip((raw - row_min) / scale, 0, 1).astype(np.float32)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    properties = np.empty((len(raw), 0), np.float32)
    prop_min = np.empty((1, 0), np.float32); prop_scale = np.empty((1, 0), np.float32)
    if args.property_columns:
        missing = set(args.property_columns) - set(frame.columns)
        if missing: raise KeyError(f"missing property columns: {sorted(missing)}")
        properties = frame.loc[kept_rows, args.property_columns].to_numpy(np.float32)
        prop_min = np.nanmin(properties[scaler_indices], axis=0, keepdims=True)
        prop_scale = np.maximum(np.nanmax(properties[scaler_indices], axis=0, keepdims=True) - prop_min, 1e-8)
        properties = (properties - prop_min) / prop_scale
    np.savez_compressed(out, matrices=norm, raw=raw, ids=np.asarray(kept_ids), source_rows=np.asarray(kept_rows),
                        train=train, val=val, test=test, row_min=row_min, row_scale=scale,
                        properties=properties, property_columns=np.asarray(args.property_columns),
                        property_min=prop_min, property_scale=prop_scale,
                        metadata=json.dumps(codec.metadata()))
    out.with_suffix(".rejected.json").write_text(json.dumps(rejected, indent=2))
    print(json.dumps({"encoded": len(raw), "rejected": len(rejected), "shape": raw.shape,
                      "train": len(train), "val": len(val), "test": len(test)}, indent=2))


def load_cache(path):
    return np.load(path, allow_pickle=False)


def train(args):
    seed_all(args.seed); device = choose_device(args.device); cache = load_cache(args.cache)
    shape = tuple(cache["matrices"].shape[1:])
    property_dim = (cache["properties"].shape[1] if "properties" in cache and
                    not args.disable_properties else 0)
    props = cache["properties"] if property_dim else None
    model = FTCPVAE(shape, args.latent_dim, property_dim).to(device)
    train_ds = MatrixDataset(cache["matrices"], cache["train"], props)
    val_ds = MatrixDataset(cache["matrices"], cache["val"], props)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_ds, args.batch_size, num_workers=args.workers,
                            pin_memory=device.type == "cuda")
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.3, patience=4, min_lr=1e-6)
    scaler = GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    run = Path(args.output); run.mkdir(parents=True, exist_ok=True)
    history, best = [], float("inf")
    for epoch in range(1, args.epochs + 1):
        # Keras callback received zero-based epochs 50 and 100.
        if epoch == 51: optimizer.param_groups[0]["lr"] = 1e-4
        elif epoch == 101: optimizer.param_groups[0]["lr"] = 5e-5
        model.train(); sums = {k: 0. for k in ("loss", "recon", "kl", "property")}; seen = 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            x = batch["x"].to(device, non_blocking=True); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=args.amp and device.type == "cuda"):
                output = model(x, sample=True)
                properties = batch.get("properties")
                mask = batch.get("property_mask")
                loss, parts = model.loss(output, x,
                    properties.to(device) if properties is not None else None,
                    mask.to(device) if mask is not None else None,
                    kl_weight=args.kl_weight, property_weight=args.property_weight,
                    legacy_keras_scaling=not args.modern_loss_scaling)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            for k in sums: sums[k] += float(parts[k]) * len(x)
            seen += len(x)
        model.eval(); val_sum = 0.; val_seen = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                properties = batch.get("properties"); mask = batch.get("property_mask")
                loss, parts = model.loss(model(x, sample=False), x,
                    properties.to(device) if properties is not None else None,
                    mask.to(device) if mask is not None else None,
                    kl_weight=args.kl_weight, property_weight=args.property_weight,
                    legacy_keras_scaling=not args.modern_loss_scaling)
                val_sum += float(parts["recon"]) * len(x); val_seen += len(x)
        metrics = {k: v / max(seen, 1) for k, v in sums.items()}
        metrics.update(epoch=epoch, val_recon=val_sum / max(val_seen, 1), lr=optimizer.param_groups[0]["lr"])
        history.append(metrics); scheduler.step(metrics["val_recon"])
        print(json.dumps(metrics))
        state = {"model": model.state_dict(), "shape": shape, "latent_dim": args.latent_dim,
                 "property_dim": property_dim, "epoch": epoch, "args": vars(args), "metrics": metrics}
        torch.save(state, run / "last.pt")
        if metrics["val_recon"] < best:
            best = metrics["val_recon"]; torch.save(state, run / "best.pt")
        (run / "history.json").write_text(json.dumps(history, indent=2))


def evaluate(args):
    device = choose_device(args.device); cache = load_cache(args.cache)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = FTCPVAE(tuple(ckpt["shape"]), ckpt["latent_dim"], ckpt.get("property_dim", 0)).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    meta = json.loads(str(cache["metadata"])); cfg = FTCPConfig(**meta["config"])
    codec = FTCPCodec(cfg, args.assets)
    if args.train_input:
        if not args.val_input or not args.test_input:
            raise ValueError("--train-input requires --val-input and --test-input")
        bundles = [load_structures(p, args.structure_column)
                   for p in (args.train_input, args.val_input, args.test_input)]
        source_structures = sum((b[1] for b in bundles), [])
    else:
        _, source_structures, _ = load_structures(args.input, args.structure_column)
    source_rows = cache["source_rows"]; indices = cache[args.split]
    matcher = StructureMatcher(ltol=args.ltol, stol=args.stol, angle_tol=args.angle_tol,
                               primitive_cell=True, scale=True, attempt_supercell=False)
    loader = DataLoader(MatrixDataset(cache["matrices"], indices), args.batch_size)
    predictions = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="reconstruct"):
            x = batch["x"].to(device)
            rec = model(x, sample=args.sample)["reconstruction"].cpu().numpy()
            predictions.extend(rec)
    row_min, row_scale = cache["row_min"], cache["row_scale"]
    thresholds = args.occupancy_threshold_grid or [args.occupancy_threshold]
    evaluations = []
    for threshold in thresholds:
        decoded_ok = matched = 0; records = []; products = []
        for pred, cache_i in zip(predictions, indices):
            raw = pred * row_scale[0] + row_min[0]
            structure = codec.decode(raw, threshold)
            truth = source_structures[int(source_rows[cache_i])]
            ok = structure is not None; match_error = None
            try:
                same_composition = bool(ok and truth.composition.reduced_composition ==
                                        structure.composition.reduced_composition)
                match = bool(same_composition and matcher.fit(truth, structure))
            except Exception as exc:
                match = False; ok = False; match_error = f"{type(exc).__name__}: {exc}"
            decoded_ok += ok; matched += match
            records.append({"id": str(cache["ids"][cache_i]), "decoded": ok, "matched": match,
                            "true_sites": len(truth), "pred_sites": len(structure) if ok else None,
                            "true_formula": truth.composition.reduced_formula,
                            "pred_formula": structure.composition.reduced_formula if ok else None,
                            "error": match_error})
            products.append({"id": str(cache["ids"][cache_i]), "cache_index": int(cache_i),
                             "decoded": bool(ok), "cif_str": structure.to(fmt="cif") if ok else None})
        result = {"split": args.split, "count": len(indices), "decoded": decoded_ok,
                  "decode_rate": decoded_ok / max(len(indices), 1), "matched": matched,
                  "match_rate": matched / max(len(indices), 1),
                  "matcher": {"ltol": args.ltol, "stol": args.stol, "angle_tol": args.angle_tol},
                  "occupancy_threshold": threshold, "sample_latent": args.sample}
        evaluations.append((result, records, products)); print(json.dumps(result))
    result, records, products = max(evaluations, key=lambda item: item[0]["match_rate"])
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": result,
                               "threshold_summaries": [x[0] for x in evaluations],
                               "structures": records}, indent=2))
    if not args.no_save_products:
        import pandas as pd
        pd.DataFrame(products).to_csv(out.with_suffix(".reconstructions.csv"), index=False)
        np.savez_compressed(out.with_suffix(".decoder_outputs.npz"),
                            normalized_ftcp=np.stack(predictions).astype(np.float32),
                            cache_indices=np.asarray(indices), ids=cache["ids"][indices],
                            occupancy_threshold=result["occupancy_threshold"])
    print(json.dumps(result, indent=2))


def environment(args):
    import importlib.metadata as md
    info = {"time": time.strftime("%Y-%m-%d %H:%M:%S %z"), "python": sys.version,
            "platform": platform.platform(), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(), "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "compute_capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
            "packages": {x: md.version(x) for x in ("numpy", "pandas", "pymatgen", "scikit-learn", "tqdm")}}
    text = json.dumps(info, indent=2); print(text)
    if args.output: Path(args.output).write_text(text)


def parser():
    p = argparse.ArgumentParser(description="PyTorch FTCP reconstruction on MP-20")
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare"); q.set_defaults(func=prepare)
    source = q.add_mutually_exclusive_group(required=True)
    source.add_argument("--input")
    source.add_argument("--train-input")
    q.add_argument("--val-input"); q.add_argument("--test-input"); q.add_argument("--output", required=True)
    q.add_argument("--assets", default="data"); q.add_argument("--structure-column", default="structure")
    q.add_argument("--max-sites", type=int, default=20); q.add_argument("--max-elms", type=int, default=7)
    q.add_argument("--pad-rows", type=int, default=2); q.add_argument("--seed", type=int, default=42)
    q.add_argument("--niggli", action="store_true")
    q.add_argument("--scaler-scope", choices=("train", "all"), default="train")
    q.add_argument("--property-columns", nargs="*", default=[])
    q.add_argument("--train-fraction", type=float, default=.8); q.add_argument("--val-fraction", type=float, default=.1)
    q = sub.add_parser("train"); q.set_defaults(func=train)
    q.add_argument("--cache", required=True); q.add_argument("--output", required=True)
    q.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    q.add_argument("--epochs", type=int, default=200); q.add_argument("--batch-size", type=int, default=256)
    q.add_argument("--workers", type=int, default=4); q.add_argument("--latent-dim", type=int, default=256)
    q.add_argument("--lr", type=float, default=5e-4); q.add_argument("--kl-weight", type=float, default=2.)
    q.add_argument("--property-weight", type=float, default=10.)
    q.add_argument("--disable-properties", action="store_true")
    q.add_argument("--modern-loss-scaling", action="store_true",
                   help="batch-independent KL weight ablation; default reproduces Keras scaling")
    q.add_argument("--seed", type=int, default=42); q.add_argument("--amp", action="store_true")
    q = sub.add_parser("evaluate"); q.set_defaults(func=evaluate)
    q.add_argument("--cache", required=True); q.add_argument("--checkpoint", required=True)
    source = q.add_mutually_exclusive_group(required=True)
    source.add_argument("--input"); source.add_argument("--train-input")
    q.add_argument("--val-input"); q.add_argument("--test-input"); q.add_argument("--output", required=True)
    q.add_argument("--assets", default="data"); q.add_argument("--structure-column", default="structure")
    q.add_argument("--split", choices=("train", "val", "test"), default="test")
    q.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    q.add_argument("--batch-size", type=int, default=256); q.add_argument("--sample", action="store_true")
    q.add_argument("--occupancy-threshold", type=float, default=.5)
    q.add_argument("--occupancy-threshold-grid", type=float, nargs="*")
    q.add_argument("--ltol", type=float, default=.2); q.add_argument("--stol", type=float, default=.3)
    q.add_argument("--angle-tol", type=float, default=5.)
    q.add_argument("--no-save-products", action="store_true",
                   help="write metrics only (use when products were saved by another matcher)")
    q = sub.add_parser("env"); q.set_defaults(func=environment); q.add_argument("--output")
    return p


if __name__ == "__main__":
    args = parser().parse_args(); args.func(args)
