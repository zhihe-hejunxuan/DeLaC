#!/usr/bin/env python3
"""实验二统一 latent sampler：训练与采样均在 h_rmsnorm 的标准化 latent 空间完成。"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parents[2]


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve(path: str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else REPO / value


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_latents(cfg: dict) -> tuple[torch.Tensor, dict]:
    path = resolve(cfg["fixed"]["normalized_latents"])
    data = torch.load(path, map_location="cpu", weights_only=False)
    x = data["flow_feature"].float().contiguous()
    if x.ndim != 2 or not torch.isfinite(x).all():
        raise ValueError(f"非法 latent: shape={tuple(x.shape)} finite={torch.isfinite(x).all().item()}")
    return x, data


def split_latents(x: torch.Tensor, seed: int, val_fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(x), generator=generator)
    n_val = max(1, round(len(x) * val_fraction))
    return x[order[n_val:]], x[order[:n_val]]


def denormalize(x: torch.Tensor, norm: dict) -> torch.Tensor:
    return x.cpu() * norm["std"].float().view(1, -1) + norm["mean"].float().view(1, -1)


def save_samples(path: Path, x_norm: torch.Tensor, norm: dict, sampler: str, seed: int, elapsed: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x_raw = denormalize(x_norm, norm)
    torch.save(x_raw, path)
    manifest = {
        "sampler": sampler,
        "seed": seed,
        "num_samples": len(x_raw),
        "latent_dim": x_raw.shape[1],
        "normalized_latent_source": str(resolve(norm["_source"])),
        "sample_seconds": elapsed,
        "output_is_denormalized_ae_feature": True,
    }
    path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def gaussian_train(x_train: torch.Tensor, model_path: Path, metadata: dict) -> None:
    import joblib
    model_path.parent.mkdir(parents=True, exist_ok=True)
    x = x_train.numpy().astype(np.float64)
    mean = x.mean(axis=0)
    covariance = np.cov(x, rowvar=False)
    jitter = 1e-6
    covariance.flat[:: covariance.shape[0] + 1] += jitter
    joblib.dump({"mean": mean, "covariance": covariance}, model_path)
    metadata.update({"covariance_type": "full", "jitter": jitter, "train_nll": None})


def gaussian_sample(model_path: Path, n: int, seed: int) -> torch.Tensor:
    import joblib
    model = joblib.load(model_path)
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.multivariate_normal(model["mean"], model["covariance"], size=n)).float()


def gmm_train(x_train: torch.Tensor, x_val: torch.Tensor, model_path: Path, cfg: dict, metadata: dict) -> None:
    import joblib
    from sklearn.mixture import GaussianMixture

    opts = cfg["gmm"]
    model = GaussianMixture(
        n_components=int(opts["components"]),
        covariance_type=opts["covariance_type"],
        reg_covar=float(opts["reg_covar"]),
        max_iter=int(opts["max_iter"]),
        n_init=int(opts["n_init"]),
        random_state=int(cfg["experiment"]["seed"]),
        verbose=1,
    )
    model.fit(x_train.numpy())
    joblib.dump(model, model_path)
    metadata.update({
        "components": model.n_components,
        "covariance_type": model.covariance_type,
        "converged": bool(model.converged_),
        "iterations": int(model.n_iter_),
        "train_nll": float(-model.score(x_train.numpy())),
        "validation_nll": float(-model.score(x_val.numpy())),
    })


def gmm_sample(model_path: Path, n: int, seed: int) -> torch.Tensor:
    import joblib
    model = joblib.load(model_path)
    model.set_params(random_state=seed)
    values, _ = model.sample(n)
    return torch.from_numpy(values).float()


def setup_nflows(cfg: dict) -> None:
    root = resolve(cfg["paths"]["official_nflows"])
    if not (root / "nflows").is_dir():
        raise FileNotFoundError(f"缺少官方 nflows: {root}")
    sys.path.insert(0, str(root))


def build_rq_nsf(dim: int, opts: dict):
    from nflows.distributions.normal import StandardNormal
    from nflows.flows.base import Flow
    from nflows.nn.nets import ResidualNet
    from nflows.transforms.base import CompositeTransform
    from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform
    from nflows.transforms.permutations import RandomPermutation

    transforms = []
    for layer in range(int(opts["num_layers"])):
        mask = torch.tensor([(i + layer) % 2 for i in range(dim)], dtype=torch.float32)

        def net_factory(in_features, out_features, _opts=opts):
            return ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=int(_opts["hidden_features"]),
                num_blocks=int(_opts["num_blocks"]),
                activation=torch.nn.functional.relu,
            )

        transforms.extend([
            RandomPermutation(features=dim),
            PiecewiseRationalQuadraticCouplingTransform(
                mask=mask,
                transform_net_create_fn=net_factory,
                num_bins=int(opts["num_bins"]),
                tails="linear",
                tail_bound=float(opts["tail_bound"]),
            ),
        ])
    return Flow(CompositeTransform(transforms), StandardNormal([dim]))


def rq_nsf_train(x_train: torch.Tensor, x_val: torch.Tensor, model_path: Path, cfg: dict, metadata: dict) -> None:
    setup_nflows(cfg)
    opts = cfg["rq_nsf"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flow = build_rq_nsf(x_train.shape[1], opts).to(device)
    loader = DataLoader(
        TensorDataset(x_train), batch_size=int(opts["batch_size"]), shuffle=True,
        generator=torch.Generator().manual_seed(int(cfg["experiment"]["seed"])),
    )
    optimizer = torch.optim.Adam(flow.parameters(), lr=float(opts["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(opts["epochs"]) * max(1, len(loader))
    )
    best = math.inf
    stale = 0
    train_on_all = bool(cfg["split"].get("train_on_all", False))
    history = []
    for epoch in range(1, int(opts["epochs"]) + 1):
        flow.train()
        total = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            loss = -flow.log_prob(batch).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), float(opts["grad_clip"]))
            optimizer.step()
            scheduler.step()
            total += loss.item() * len(batch)
        flow.eval()
        with torch.no_grad():
            val_nll = float(-flow.log_prob(x_val.to(device)).mean().item())
        train_nll = total / len(x_train)
        history.append({"epoch": epoch, "train_nll": train_nll, "validation_nll": val_nll})
        print(f"[RQ-NSF] epoch={epoch} train_nll={train_nll:.6f} val_nll={val_nll:.6f}", flush=True)
        if val_nll < best:
            best, stale = val_nll, 0
            torch.save({"state_dict": flow.state_dict(), "dim": x_train.shape[1], "options": opts}, model_path)
        else:
            stale += 1
            if not train_on_all and stale >= int(opts["patience"]):
                break
        # MAF 的论文 checkpoint 按训练损失选 best 且训练全量数据。train_on_all
        # 模式固定跑满相同 epoch，最终另存 last，best 仅供诊断。
        if train_on_all and epoch == int(opts["epochs"]):
            torch.save({"state_dict": flow.state_dict(), "dim": x_train.shape[1], "options": opts}, model_path)
    model_path.with_suffix(".history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    metadata.update({"validation_nll": best, "epochs_completed": len(history)})


def rq_nsf_sample(model_path: Path, n: int, seed: int, cfg: dict) -> torch.Tensor:
    setup_nflows(cfg)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flow = build_rq_nsf(int(ckpt["dim"]), ckpt["options"]).to(device)
    flow.load_state_dict(ckpt["state_dict"])
    flow.eval()
    seed_all(seed)
    with torch.no_grad():
        return flow.sample(n).cpu()


def ddpm_train(x_train: torch.Tensor, model_path: Path, cfg: dict, metadata: dict) -> None:
    official = resolve(cfg["paths"]["official_tab_ddpm"])
    if not (official / "tab_ddpm").is_dir():
        raise FileNotFoundError(f"缺少官方 TabDDPM: {official}")
    os.environ["TAB_DDPM_ROOT"] = str(official)
    sys.path.insert(0, str(REPO))
    from script.latent_ddpm_utils import save_ddpm_checkpoint, train_ddpm_on_latents

    import joblib
    from sklearn.preprocessing import QuantileTransformer

    opts = dict(cfg["latent_ddpm"])
    preprocessing = opts.get("preprocessing", "quantile_normal")
    if preprocessing != "quantile_normal":
        raise ValueError(f"不支持的 DDPM preprocessing: {preprocessing}")
    transformer = QuantileTransformer(
        n_quantiles=min(int(opts.get("quantile_n_quantiles", 1000)), len(x_train)),
        output_distribution="normal",
        subsample=int(1e9),
        random_state=int(cfg["experiment"]["seed"]),
        copy=True,
    )
    x_ddpm = torch.from_numpy(
        transformer.fit_transform(x_train.numpy()).astype(np.float32, copy=False)
    )
    transform_path = model_path.with_suffix(".preprocess.joblib")
    joblib.dump(transformer, transform_path)

    opts["lr"] = float(opts.pop("learning_rate"))
    diffusion, model_params, history, ema = train_ddpm_on_latents(
        x_ddpm, opts, steps=int(opts["steps"]), device="cuda",
        ema_every=int(opts["ema_every"]),
    )
    save_ddpm_checkpoint(
        str(model_path), diffusion, model_params, opts, ema_state=ema,
        extra={
            "official_source": str(official),
            "num_train": len(x_train),
            "preprocessing": preprocessing,
            "preprocess_path": str(transform_path),
        },
    )
    history.to_csv(model_path.with_suffix(".history.csv"), index=False)
    metadata.update({
        "steps": int(opts["steps"]),
        "train_loss_last": float(history.iloc[-1]["loss"]),
        "preprocessing": preprocessing,
        "preprocess_path": str(transform_path),
        "ddpm_train_mean": float(x_ddpm.mean()),
        "ddpm_train_std": float(x_ddpm.std()),
    })


def ddpm_sample(model_path: Path, n: int, seed: int, cfg: dict) -> torch.Tensor:
    import joblib
    official = resolve(cfg["paths"]["official_tab_ddpm"])
    os.environ["TAB_DDPM_ROOT"] = str(official)
    sys.path.insert(0, str(REPO))
    from script.latent_ddpm_utils import (
        load_ddpm_from_checkpoint,
        sample_ddpm_latents_bounded_x0,
    )

    use_ema = bool(cfg["latent_ddpm"].get("sample_use_ema", True))
    diffusion, opts, _ = load_ddpm_from_checkpoint(
        str(model_path), None, device="cuda", use_ema=use_ema
    )
    values_ddpm, _ = sample_ddpm_latents_bounded_x0(
        diffusion,
        n,
        seed=seed,
        batch_size=int(cfg["latent_ddpm"]["sample_batch_size"]),
        pred_x0_clip=float(cfg["latent_ddpm"].get("sample_pred_x0_clip", 5.0)),
    )
    transform_path = model_path.with_suffix(".preprocess.joblib")
    if not transform_path.is_file():
        raise FileNotFoundError(f"缺少 DDPM inverse transform: {transform_path}")
    transformer = joblib.load(transform_path)
    values = torch.from_numpy(
        transformer.inverse_transform(values_ddpm.numpy()).astype(np.float32, copy=False)
    )
    gate = cfg["latent_ddpm"].get("quality_gate", {})
    global_std = float(values.std())
    row_absmax = values.abs().amax(dim=1)
    row_p99 = float(torch.quantile(row_absmax, 0.99))
    absolute_max = float(values.abs().max())
    diagnostics = {
        "ddpm_space_mean": float(values_ddpm.mean()),
        "ddpm_space_std": float(values_ddpm.std()),
        "latent_norm_mean": float(values.mean()),
        "latent_norm_std": global_std,
        "row_absmax_p99": row_p99,
        "absolute_max": absolute_max,
        "sample_weights": "ema" if use_ema else "raw",
        "pred_x0_clip": float(cfg["latent_ddpm"].get("sample_pred_x0_clip", 5.0)),
    }
    model_path.with_suffix(".sample_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    failures = []
    if not float(gate.get("global_std_min", 0.0)) <= global_std <= float(gate.get("global_std_max", math.inf)):
        failures.append(f"global_std={global_std:.4g}")
    if row_p99 > float(gate.get("row_absmax_p99_max", math.inf)):
        failures.append(f"row_absmax_p99={row_p99:.4g}")
    if absolute_max > float(gate.get("absolute_max", math.inf)):
        failures.append(f"absolute_max={absolute_max:.4g}")
    if failures:
        raise RuntimeError("DDPM latent quality gate 失败: " + ", ".join(failures))
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--sampler", required=True, choices=("empirical_gaussian", "gmm", "maf", "rq_nsf", "latent_ddpm"))
    parser.add_argument("--mode", required=True, choices=("train", "sample", "train_sample"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--samples", help="Required for sample/train_sample; unused for train")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--ddpm_pred_x0_clip",
        type=float,
        default=0.0,
        help="仅用于 DDPM 采样消融；>0 时覆盖 config 的 sample_pred_x0_clip",
    )
    args = parser.parse_args()
    if args.mode != "train" and not args.samples: parser.error("--samples is required when sampling")

    cfg = load_cfg(args.config)
    if args.ddpm_pred_x0_clip > 0:
        cfg["latent_ddpm"]["sample_pred_x0_clip"] = args.ddpm_pred_x0_clip
    seed = int(cfg["experiment"]["seed"])
    if args.sampler == "maf":
        sys.path.insert(0,str(REPO))
        from MAF_script.train_MAF_no_cond import build_maf_no_cond
        seed_all(seed)
        x,norm=load_latents(cfg)
        norm["_source"]=cfg["fixed"]["normalized_latents"]
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt=resolve(args.model)
        if args.mode in ("train","train_sample") and (args.force or not ckpt.is_file()):
            import subprocess
            if ckpt.name != "model_best.pt": parser.error("MAF --model must end in model_best.pt")
            subprocess.run([sys.executable,str(REPO/"MAF_script/train_MAF_no_cond.py"),
                "--pt_path",str(resolve(cfg["fixed"]["normalized_latents"])),
                "--save_dir",str(ckpt.parent),"--seed",str(seed)],check=True,cwd=REPO)
        if args.mode=="train": return
        flow=build_maf_no_cond(x.shape[1],512,10,2,False,True,True,"relu",0.,False,True).to(device)
        flow.load_state_dict(torch.load(ckpt,map_location=device,weights_only=True)); flow.eval()
        if args.mode in ("sample","train_sample"):
            destination=resolve(args.samples)
            if destination.exists() and not args.force: return
            started=time.perf_counter()
            with torch.no_grad(): values=flow.sample(int(cfg["experiment"]["num_samples"])).squeeze(1).cpu()
            save_samples(destination,values,norm,"maf",seed,time.perf_counter()-started)
        return
    seed_all(seed)
    x, norm = load_latents(cfg)
    norm["_source"] = cfg["fixed"]["normalized_latents"]
    split_train, x_val = split_latents(x, seed, float(cfg["split"]["validation_fraction"]))
    train_on_all = bool(cfg["split"].get("train_on_all", False))
    x_train = x if train_on_all else split_train
    model_path = resolve(args.model)
    sample_path = resolve(args.samples) if args.samples else None
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode in ("train", "train_sample") and (args.force or not model_path.exists()):
        metadata = {
            "sampler": args.sampler, "seed": seed, "train_size": len(x_train),
            "validation_size": len(x_val), "latent_dim": x.shape[1],
            "train_on_all": train_on_all,
            "validation_is_diagnostic_only": train_on_all,
            "normalized_latent_source": str(resolve(cfg["fixed"]["normalized_latents"])),
        }
        started = time.perf_counter()
        if args.sampler == "empirical_gaussian":
            gaussian_train(x_train, model_path, metadata)
        elif args.sampler == "gmm":
            gmm_train(x_train, x_val, model_path, cfg, metadata)
        elif args.sampler == "rq_nsf":
            rq_nsf_train(x_train, x_val, model_path, cfg, metadata)
        else:
            ddpm_train(x_train, model_path, cfg, metadata)
        metadata["train_seconds"] = time.perf_counter() - started
        metadata["parameter_count"] = None
        model_path.with_suffix(".manifest.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    if args.mode in ("sample", "train_sample") and (args.force or not sample_path.exists()):
        started = time.perf_counter()
        n = int(cfg["experiment"]["num_samples"])
        if args.sampler == "empirical_gaussian":
            values = gaussian_sample(model_path, n, seed)
        elif args.sampler == "gmm":
            values = gmm_sample(model_path, n, seed)
        elif args.sampler == "rq_nsf":
            values = rq_nsf_sample(model_path, n, seed, cfg)
        else:
            values = ddpm_sample(model_path, n, seed, cfg)
        if values.shape != (n, x.shape[1]) or not torch.isfinite(values).all():
            raise RuntimeError(f"采样输出非法: {tuple(values.shape)}")
        save_samples(sample_path, values, norm, args.sampler, seed, time.perf_counter() - started)


if __name__ == "__main__":
    main()
