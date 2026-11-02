"""TabDDPM 封装：在 144 维归一化 latent 上训练 / 采样（无条件，纯数值特征）。"""
from __future__ import annotations

import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
import pandas as pd
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TDDPM_ROOT = Path(
    os.environ.get("TAB_DDPM_ROOT", str(_REPO_ROOT / "final_experiments/experiment_2/vendor/tab-ddpm-official"))
).expanduser().resolve()


def setup_tab_ddpm_imports() -> Path:
    if not _TDDPM_ROOT.is_dir():
        raise FileNotFoundError(f"缺少 TabDDPM 目录: {_TDDPM_ROOT}")
    for p in (_TDDPM_ROOT, _TDDPM_ROOT / "scripts"):
        ps = str(p)
        if ps not in sys.path:
            sys.path.insert(0, ps)
    return _TDDPM_ROOT


def _default_model_params(x_dim: int, ddpm_cfg: dict) -> dict:
    return {
        "d_in": x_dim,
        "num_classes": 0,
        "is_y_cond": False,
        "dim_t": int(ddpm_cfg.get("dim_t", 128)),
        "rtdl_params": {
            "d_layers": list(ddpm_cfg.get("d_layers", [512, 1024, 1024])),
            "dropout": float(ddpm_cfg.get("dropout", 0.0)),
        },
    }


def build_ddpm_diffusion(x_dim: int, ddpm_cfg: dict, device: torch.device):
    setup_tab_ddpm_imports()
    from tab_ddpm import GaussianMultinomialDiffusion
    from tab_ddpm.modules import MLPDiffusion, ResNetDiffusion

    model_params = _default_model_params(x_dim, ddpm_cfg)
    model_type = ddpm_cfg.get("model_type", "mlp")
    if model_type == "mlp":
        model = MLPDiffusion(**model_params)
    elif model_type == "resnet":
        model = ResNetDiffusion(**model_params)
    else:
        raise ValueError(f"Unknown TabDDPM model_type: {model_type}")
    model.to(device)
    diffusion = GaussianMultinomialDiffusion(
        num_classes=np.array([0]),
        num_numerical_features=x_dim,
        denoise_fn=model,
        num_timesteps=int(ddpm_cfg.get("num_timesteps", 1000)),
        gaussian_loss_type=ddpm_cfg.get("gaussian_loss_type", "mse"),
        scheduler=ddpm_cfg.get("scheduler", "cosine"),
        device=device,
    )
    diffusion.to(device)
    return diffusion, model_params


def update_ema(target_params, source_params, rate: float = 0.999) -> None:
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src.detach(), alpha=1 - rate)


def infinite_loader(x: torch.Tensor, batch_size: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    n = x.shape[0]
    while True:
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            batch_x = x[idx]
            batch_y = torch.zeros(batch_x.shape[0], dtype=torch.long)
            yield batch_x, batch_y


def train_ddpm_on_latents(
    x_norm: torch.Tensor,
    ddpm_cfg: dict,
    device: str = "cuda",
    steps: int = 25000,
    log_every: int = 100,
    print_every: int = 500,
    ema_every: int = 1,
) -> Tuple[Any, dict, pd.DataFrame, Optional[torch.Tensor]]:
    """在归一化 latent 上训练 TabDDPM，返回 (diffusion, model_params, loss_df, ema_state)。"""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    x = x_norm.float().contiguous()
    x_dim = x.shape[1]
    diffusion, model_params = build_ddpm_diffusion(x_dim, ddpm_cfg, dev)

    ema_model = deepcopy(diffusion._denoise_fn)
    for p in ema_model.parameters():
        p.detach_()

    batch_size = int(ddpm_cfg.get("batch_size", 512))
    lr = float(ddpm_cfg.get("lr", 1e-3))
    weight_decay = float(ddpm_cfg.get("weight_decay", 0.0))
    use_ema = bool(ddpm_cfg.get("use_ema", True))

    opt = torch.optim.AdamW(diffusion.parameters(), lr=lr, weight_decay=weight_decay)
    train_iter = infinite_loader(x, batch_size)

    loss_rows = []
    curr_multi = curr_gauss = 0.0
    curr_count = 0

    diffusion.train()
    for step in range(steps):
        batch_x, batch_y = next(train_iter)
        batch_x = batch_x.to(dev)
        batch_y = batch_y.to(dev)
        out_dict = {"y": batch_y}

        frac = (step + 1) / max(1, steps)
        for pg in opt.param_groups:
            pg["lr"] = lr * (1.0 - frac)

        opt.zero_grad(set_to_none=True)
        loss_multi, loss_gauss = diffusion.mixed_loss(batch_x, out_dict)
        loss = loss_multi + loss_gauss
        loss.backward()
        opt.step()

        # 与 TabDDPM 官方 Trainer 一致：默认每个优化 step 更新 EMA。
        if use_ema and (step + 1) % max(1, ema_every) == 0:
            update_ema(ema_model.parameters(), diffusion._denoise_fn.parameters())

        bs = batch_x.shape[0]
        curr_multi += loss_multi.item() * bs
        curr_gauss += loss_gauss.item() * bs
        curr_count += bs

        if (step + 1) % log_every == 0:
            mloss = curr_multi / max(1, curr_count)
            gloss = curr_gauss / max(1, curr_count)
            loss_rows.append({"step": step + 1, "mloss": mloss, "gloss": gloss, "loss": mloss + gloss})
            if (step + 1) % print_every == 0:
                print(
                    f"[DDPM] step {step + 1}/{steps} MLoss={mloss:.4f} GLoss={gloss:.4f} Sum={mloss + gloss:.4f}",
                    flush=True,
                )
            curr_multi = curr_gauss = 0.0
            curr_count = 0

    loss_df = pd.DataFrame(loss_rows)
    ema_state = ema_model.state_dict() if use_ema else None
    return diffusion, model_params, loss_df, ema_state


def save_ddpm_checkpoint(
    path: str,
    diffusion,
    model_params: dict,
    ddpm_cfg: dict,
    ema_state: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> None:
    payload = {
        "model_state_dict": diffusion._denoise_fn.state_dict(),
        "ema_state_dict": ema_state,
        "model_params": model_params,
        "ddpm_cfg": ddpm_cfg,
        "num_numerical_features": int(diffusion.num_numerical_features),
        "K": np.array([0]),
    }
    if extra:
        payload.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_ddpm_from_checkpoint(
    ckpt_path: str,
    ddpm_cfg: dict | None,
    device: str = "cuda",
    use_ema: bool = True,
):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ddpm_cfg or ckpt.get("ddpm_cfg", {})
    x_dim = int(ckpt.get("num_numerical_features", 144))
    model_params = ckpt.get("model_params") or _default_model_params(x_dim, cfg)

    diffusion, _ = build_ddpm_diffusion(x_dim, cfg, dev)
    state = ckpt.get("ema_state_dict") if use_ema else None
    state = state or ckpt.get("model_state_dict")
    if state is None:
        raise KeyError(f"checkpoint 缺少 model_state_dict: {ckpt_path}")
    diffusion._denoise_fn.load_state_dict(state)
    diffusion.eval()
    return diffusion, cfg, model_params


@torch.no_grad()
def sample_ddpm_latents(
    diffusion,
    num_samples: int,
    device: str = "cuda",
    batch_size: int = 256,
    seed: int = 42,
    ddim: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """TabDDPM 采样，返回 (x_norm, x_norm) — 与 MAF 接口一致，均在归一化空间。"""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    diffusion.to(dev)
    diffusion.eval()

    y_dist = torch.tensor([1.0], device=dev)
    samples = []
    remaining = num_samples
    while remaining > 0:
        b = min(batch_size, remaining)
        x_gen, _ = diffusion.sample_all(b, b, y_dist, ddim=ddim)
        if torch.is_tensor(x_gen):
            x_gen = x_gen.float()
        else:
            x_gen = torch.from_numpy(x_gen).float()
        mask = torch.isfinite(x_gen).all(dim=1)
        x_gen = x_gen[mask]
        if x_gen.shape[0] == 0:
            raise RuntimeError("TabDDPM 采样全部为 NaN")
        samples.append(x_gen.cpu())
        remaining -= x_gen.shape[0]

    x_norm = torch.cat(samples, dim=0)[:num_samples]
    return x_norm, x_norm


@torch.no_grad()
def sample_ddpm_latents_bounded_x0(
    diffusion,
    num_samples: int,
    device: str = "cuda",
    batch_size: int = 256,
    seed: int = 42,
    pred_x0_clip: float = 5.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """纯数值 TabDDPM 采样，并在每一步约束预测 x0 到训练支持域。

    官方 ``sample`` 对 ``clip_denoised`` 参数未实际应用；高维连续 latent
    上误差会沿 1000 步反向链累积。这里保留官方 eps 参数化、posterior 和
    noise schedule，仅在重算 posterior 前裁剪 pred_x0。
    """
    import random

    if pred_x0_clip <= 0:
        raise ValueError("pred_x0_clip 必须 > 0")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    diffusion.to(dev)
    diffusion.eval()
    if int(diffusion.num_classes[0]) != 0:
        raise ValueError("bounded_x0 sampler 仅支持本实验的纯数值无条件 DDPM")

    samples = []
    remaining = num_samples
    while remaining > 0:
        b = min(batch_size, remaining)
        x = torch.randn((b, diffusion.num_numerical_features), device=dev)
        y = torch.zeros(b, dtype=torch.long, device=dev)
        out_dict = {"y": y}
        for step in reversed(range(diffusion.num_timesteps)):
            t = torch.full((b,), step, device=dev, dtype=torch.long)
            eps = diffusion._denoise_fn(x.float(), t, **out_dict)
            pred_x0 = diffusion._predict_xstart_from_eps(x_t=x, t=t, eps=eps)
            pred_x0 = pred_x0.clamp(-pred_x0_clip, pred_x0_clip)
            mean, _, log_variance = diffusion.gaussian_q_posterior_mean_variance(
                x_start=pred_x0, x_t=x, t=t
            )
            noise = torch.randn_like(x)
            nonzero = (t != 0).float().view(-1, 1)
            x = mean + nonzero * torch.exp(0.5 * log_variance) * noise
        if not torch.isfinite(x).all():
            raise RuntimeError("bounded-x0 DDPM 采样出现 NaN/Inf")
        samples.append(x.cpu())
        remaining -= b

    x_norm = torch.cat(samples, dim=0)[:num_samples]
    return x_norm, x_norm


def denorm_latents(x_norm: torch.Tensor, norm_data: dict) -> torch.Tensor:
    mean = norm_data["mean"].float()
    std = norm_data["std"].float()
    if mean.dim() == 1:
        mean = mean.unsqueeze(0)
    if std.dim() == 1:
        std = std.unsqueeze(0)
    return x_norm * std + mean
