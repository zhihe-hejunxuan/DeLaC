"""Reconstruction helper functions used by the frozen-normalization diagnostic."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import os, sys, json
import numpy as np
import pandas as pd
import torch
from pymatgen.core import Structure
from torch_geometric.data import DataLoader
from tqdm import tqdm
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
from dataset import MatDataset
from script.runner_for_flow_prep import RunnerFlowPrep, set_global_seed
from script.structure_match_novel_consistent import DefaultDisorderedStructureMatcher, atomic_position_match, comp_match, composition_match, full_match, lattice_match
from script.tools import build_structures_for_m3gnet
from utils import StandardScalerTorch


def _load_conf(config_suffix):
    import importlib
    if config_suffix not in ("flow_prep_none", "flow_prep_zscore", "rmsnorm", "h_rmsnorm"):
        raise ValueError(config_suffix)
    return importlib.import_module("config.FlowPrep.mp_20_conifg_dict_large_"+config_suffix).conf


def _build_runner(config_suffix, dataset, model_scale, result_version, checkpoint, device):
    conf = _load_conf(config_suffix)
    train_ds = MatDataset(str(_REPO_ROOT / "data" / dataset / "train.pt"), **conf["data"])
    runner = RunnerFlowPrep(conf, str(_REPO_ROOT / "data" / dataset / "new_score_norm.txt"), device=device, dataset_name=dataset)
    runner.model.lattice_normalizer = runner._get_normalizer(train_ds)
    checkpoint = Path(checkpoint)
    if not checkpoint.is_absolute(): checkpoint = _REPO_ROOT / checkpoint
    runner.load_model(str(checkpoint))
    runner.model.eval()
    return runner, conf, checkpoint


def _sample_indices(num_samples: int, total: int, seed: int) -> list[int]:
    n = min(num_samples, total)
    rng = np.random.default_rng(seed)
    if n >= total:
        return list(range(total))
    return sorted(rng.choice(total, size=n, replace=False).tolist())


def _gt_structure(data_dict: dict) -> Structure | None:
    try:
        return Structure.from_str(data_dict["cif"], fmt="cif")
    except Exception:
        return None


def _decode_batch(runner, data_batch, decode_mode="default", coordgen_kwargs=None):
    structs, _ = _reconstruction_with_coordgen(runner, data_batch, coordgen_kwargs or {})
    return structs


def _reconstruction_with_coordgen(
    runner: RunnerFlowPrep,
    data_batch,
    coordgen_kwargs: dict,
    frac_coords_init: torch.Tensor | None = None,
) -> tuple[list[Structure | None], torch.Tensor]:
    """与 reconstuction_crys 相同，但可传入 coordgen.generate 的额外参数。"""
    model = runner.model
    try:
        h = model.encoder_backbone(data_batch)
        if getattr(model, "h_pre_norm_flag", False):
            h = model.h_pre_norm(h)

        lengths_angles = torch.cat([data_batch.scaled_lengths, data_batch.angles], dim=-1)
        if model.lattice_normalizer is not None:
            lengths_angles = model.lattice_normalizer.transform(lengths_angles)
        latent_latt = model.fc_lattice_enc(lengths_angles)

        num_atoms, atom_types, lengths, angles = model.ae_decode_from_h(h, latent_latt)
        if hasattr(model, "_encode_h_to_latents"):
            _, latent_pos, _ = model._encode_h_to_latents(h)
        else:
            latent_pos = model.h_to_latent_pos(model.h_norm_for_pos(h))

        batch = torch.repeat_interleave(
            torch.arange(len(num_atoms), device=lengths.device), num_atoms
        )
        if frac_coords_init is None and coordgen_kwargs.get("init_mode") == "seed_per_graph":
            frac_coords_init = _build_seed_per_graph_frac_init(
                data_batch, num_atoms, batch, lengths.device, coordgen_kwargs.get("init_seed", 42)
            )
        elif frac_coords_init is None and coordgen_kwargs.get("init_mode") == "zeros":
            frac_coords_init = torch.zeros(batch.shape[0], 3, device=lengths.device)
        elif frac_coords_init is None and coordgen_kwargs.get("init_mode") == "gt_frac":
            frac_coords_init = _build_gt_frac_init(data_batch, num_atoms, batch, lengths.device)

        gen_kw = {
            k: v
            for k, v in coordgen_kwargs.items()
            if k not in ("init_mode", "init_seed", "share_init_across_passes")
        }
        if frac_coords_init is not None:
            gen_kw["frac_coords_init"] = frac_coords_init

        frac_coords = model.coordgen.generate(
            latents=latent_pos,
            num_atoms=num_atoms,
            atom_types=atom_types - 1,
            lengths=lengths,
            angles=angles,
            noise_start=model.noise_start,
            noise_end=model.noise_end,
            num_gen_steps=model.num_time_steps,
            **gen_kw,
        )

        structs = build_structures_for_m3gnet(
            num_atoms,
            atom_types,
            lengths,
            angles,
            frac_coords,
            angles_in_radians=False,
            zero_based_types=False,
        )
        return structs, frac_coords_init
    except Exception as exc:
        print(f"[WARN] reconstruction batch failed: {type(exc).__name__}: {exc}", flush=True)
        n = int(getattr(data_batch, "num_graphs", 1))
        return [None] * n, frac_coords_init


def _build_gt_frac_init(data_batch, num_atoms: torch.Tensor, batch: torch.Tensor, device) -> torch.Tensor:
    """GT 分数坐标作起点（仅当 decode 后原子数与 GT 一致时使用，否则回退 zeros）。"""
    gt_frac = data_batch.frac_coords.to(device=device, dtype=torch.float32)
    out = torch.zeros(batch.shape[0], 3, device=device)
    ptr = 0
    gt_ptr = 0
    for g in range(len(num_atoms)):
        n_dec = int(num_atoms[g].item())
        n_gt = int(data_batch.num_atoms[g].item()) if hasattr(data_batch, "num_atoms") else n_dec
        if n_dec == n_gt and gt_ptr + n_gt <= gt_frac.shape[0]:
            out[ptr : ptr + n_dec] = gt_frac[gt_ptr : gt_ptr + n_gt]
        ptr += n_dec
        gt_ptr += n_gt
    return out


def _build_seed_per_graph_frac_init(
    data_batch, num_atoms: torch.Tensor, batch: torch.Tensor, device, base_seed: int
) -> torch.Tensor:
    """每条晶体用固定 seed（base_seed + graph_id）生成 Uniform[-0.5,0.5] 起点。"""
    out = torch.empty(batch.shape[0], 3, device=device)
    ptr = 0
    for g in range(len(num_atoms)):
        n = int(num_atoms[g].item())
        gen = torch.Generator(device=device)
        gen.manual_seed(base_seed + g)
        out[ptr : ptr + n] = torch.rand(n, 3, device=device, generator=gen) - 0.5
        ptr += n
    return out


def _aggregate_model_metrics(loss_dict: dict, acc: dict[str, float]) -> None:
    acc["_n"] += int(loss_dict["total_elem_type_num_num"])
    acc["elem_type_num_correct"] += int(loss_dict["elem_type_num_correct"])
    acc["elem_type_match"] += int(loss_dict["composition_correct"][0])
    acc["elem_num_match"] += int(loss_dict["composition_correct"][1])
    acc["comp_match"] += int(loss_dict["composition_correct"][2])
    if "composition_correct_nogt" in loss_dict:
        acc["comp_match_nogt"] += int(loss_dict["composition_correct_nogt"][2])
        acc["_has_nogt"] = True


def _finalize_model_metrics(acc: dict[str, Any]) -> dict[str, float]:
    denom = max(1, acc["_n"])
    out = {
        "elem_type_num_acc": acc["elem_type_num_correct"] / denom,
        "elem_type_match": acc["elem_type_match"] / denom,
        "elem_num_match": acc["elem_num_match"] / denom,
        "comp_match": acc["comp_match"] / denom,
        "comp_match_nogt": (acc["comp_match_nogt"] / denom) if acc.get("_has_nogt") else float("nan"),
        "n_samples": int(acc["_n"]),
    }
    return out


def _new_model_acc() -> dict[str, Any]:
    return {
        "_n": 0,
        "_has_nogt": False,
        "elem_type_num_correct": 0,
        "elem_type_match": 0,
        "elem_num_match": 0,
        "comp_match": 0,
        "comp_match_nogt": 0,
    }


def _structure_metrics(
    gt: Structure | None,
    pred: Structure | None,
    matcher: DefaultDisorderedStructureMatcher,
) -> dict[str, bool | None]:
    if gt is None or pred is None:
        return {
            "parse_ok": False,
            "composition_match": None,
            "comp_match": None,
            "lattice_match": None,
            "atomic_position_match": None,
            "full_match": None,
        }
    return {
        "parse_ok": True,
        "composition_match": composition_match(gt, pred, matcher),
        "comp_match": comp_match(gt, pred),
        "lattice_match": lattice_match(gt, pred, matcher),
        "atomic_position_match": atomic_position_match(gt, pred, matcher),
        "full_match": full_match(gt, pred, matcher),
    }

