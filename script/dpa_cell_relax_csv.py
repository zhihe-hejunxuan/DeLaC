#!/usr/bin/env python3
"""通用 DPA 变胞/定胞弛豫（对齐 project_memory 统一弛豫做法）。

环境：neqV
  FrechetCellFilter + FIRE（mode=cell）或仅原子（mode=fixed）
  fmax=0.05，max_steps=300，--resume
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
from deepmd.calculator import DP
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

_DEFAULT_DPA = "mlff_choosing/dpa-3.1-mptrj.pth"
_REPO = Path(__file__).resolve().parent.parent


def _atomic_to_csv(frame: pd.DataFrame, path: Path) -> None:
    """Keep resume/watchdog readers from observing a partially written CSV."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_progress(path: Path | None, **payload) -> None:
    """Cheap per-crystal heartbeat independent of the larger CSV checkpoint."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _read_structure(value) -> Structure | None:
    if pd.isna(value):
        return None
    try:
        return Structure.from_str(str(value).replace("\\n", "\n"), fmt="cif")
    except Exception:
        return None


def _bool_series(s: pd.Series) -> pd.Series:
    return s.fillna(False).astype(str).str.lower().isin(("true", "1"))


def _verify_gpu_process() -> dict | None:
    """Return this process' NVIDIA allocation, not merely CUDA availability."""
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_gpu_memory,gpu_uuid",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=30, check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) < 4:
            continue
        try:
            pid, used = int(fields[0]), int(fields[2])
        except ValueError:
            continue
        if pid == os.getpid() and used > 0:
            return {
                "pid": pid, "process_name": fields[1], "used_memory_mib": used,
                "gpu_uuid": fields[3],
            }
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--model", type=str, default=_DEFAULT_DPA)
    ap.add_argument("--mode", choices=("fixed", "cell"), default="cell")
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--save_interval", type=int, default=10)
    ap.add_argument(
        "--progress_file", type=str, default="",
        help="Optional atomic JSON heartbeat written after every completed crystal",
    )
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--cif_dir", type=str, default="")
    ap.add_argument("--cif_name", type=str, default="cif_str")
    ap.add_argument(
        "--cuda_memory_fraction", type=float, default=None,
        help="Optional per-process PyTorch CUDA allocator limit in (0, 1]",
    )
    ap.add_argument(
        "--require_gpu", action="store_true",
        help="首次真实 DPA 计算后，要求本进程出现在 NVIDIA compute-apps 中",
    )
    args = ap.parse_args()

    if args.cuda_memory_fraction is not None:
        if not 0.0 < args.cuda_memory_fraction <= 1.0:
            raise ValueError("--cuda_memory_fraction must be in (0, 1]")
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA memory fraction requested but CUDA is unavailable")
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, device=0)
        print(
            f"[CUDA LIMIT] per_process_memory_fraction={args.cuda_memory_fraction:.6f}",
            flush=True,
        )

    src = Path(args.input)
    if not src.is_file():
        src = _REPO / args.input
    out = Path(args.output)
    if not out.is_absolute():
        out = _REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    progress_file = Path(args.progress_file).expanduser().resolve() if args.progress_file else None

    source = pd.read_csv(src)
    if args.cif_name not in source.columns:
        raise RuntimeError(f"缺少列 {args.cif_name}")

    cif_dir = Path(args.cif_dir) if args.cif_dir else out.parent / f"cif_{args.mode}_relaxed"
    if not cif_dir.is_absolute():
        cif_dir = _REPO / cif_dir
    cif_dir.mkdir(parents=True, exist_ok=True)

    if out.exists() and args.resume:
        frame = pd.read_csv(out)
        if len(frame) != len(source):
            raise ValueError(f"resume 行数不一致: {len(frame)} vs {len(source)}")
    else:
        frame = source.copy()
        # 保留未弛豫 CIF
        if "cif_str_unrelaxed" not in frame.columns:
            frame["cif_str_unrelaxed"] = frame[args.cif_name]
        defaults = {
            "relax_mode": args.mode,
            "relax_done": False,
            "relax_converged": False,
            "relax_steps": np.nan,
            "relax_wall_seconds": np.nan,
            "relax_error": "",
            "initial_energy_dpa_raw": np.nan,
            "initial_max_force_eV_per_A": np.nan,
            "final_max_force_eV_per_A": np.nan,
            "initial_volume_A3": np.nan,
            "final_volume_A3": np.nan,
            "volume_change_fraction": np.nan,
            "E_tot_pred_dpa_raw": np.nan,
            "dpa_done": False,
            "relaxed_cif_path": "",
            "relax_failure_type": "",
            "relax_retry_required": False,
        }
        for col, val in defaults.items():
            if col not in frame.columns:
                frame[col] = val

    done = _bool_series(frame["relax_done"])
    _atomic_progress(
        progress_file, status="starting", pid=os.getpid(),
        completed=int(done.sum()), total=len(frame), active_position=None,
        updated_monotonic=time.monotonic(),
    )
    print(
        f"[INFO] mode={args.mode} fmax={args.fmax} max_steps={args.max_steps} "
        f"done={int(done.sum())}/{len(frame)} -> {out}"
    )

    calc = DP(model=args.model)
    adaptor = AseAtomsAdaptor()

    # 先用一个真实输入触发 DeepMD 后端和 CUDA context，再按当前 PID 验证。
    # 此判断不会被逐结构的异常捕获吞掉，因此 --require_gpu 可防止静默回退 CPU。
    if args.require_gpu:
        probe_structure = None
        src_col = "cif_str_unrelaxed" if "cif_str_unrelaxed" in frame.columns else args.cif_name
        for value in frame[src_col]:
            probe_structure = _read_structure(value)
            if probe_structure is not None:
                break
        if probe_structure is None:
            raise RuntimeError("GPU probe 找不到可解析的输入 CIF")
        probe_atoms = adaptor.get_atoms(probe_structure)
        probe_atoms.calc = calc
        probe_atoms.get_potential_energy()
        gpu_process = _verify_gpu_process()
        verified = gpu_process is not None
        print(
            f"[GPU VERIFY] GPU_RELAXATION_ACTIVE={verified}"
            + (f" details={gpu_process}" if gpu_process else ""),
            flush=True,
        )
        if not verified:
            raise RuntimeError("--require_gpu 已启用，但无法确认当前弛豫进程占用 NVIDIA GPU")

    for position in range(len(frame)):
        if done.iloc[position]:
            continue
        crystal_started = time.perf_counter()
        _atomic_progress(
            progress_file, status="running", pid=os.getpid(),
            completed=int(_bool_series(frame["relax_done"]).sum()), total=len(frame),
            active_position=position, updated_monotonic=time.monotonic(),
        )
        # resume 时 cif_str 可能已是弛豫后结构，优先用未弛豫备份
        src_col = "cif_str_unrelaxed" if "cif_str_unrelaxed" in frame.columns else args.cif_name
        structure = _read_structure(frame.at[position, src_col])
        sample_id = frame.at[position, "id"] if "id" in frame.columns else position
        try:
            if structure is None:
                raise ValueError("unable to parse input CIF")
            atoms = adaptor.get_atoms(structure)
            atoms.calc = calc
            initial_volume = float(atoms.get_volume())
            initial_energy = float(atoms.get_potential_energy())
            initial_max_force = float(np.linalg.norm(atoms.get_forces(), axis=1).max())

            target = atoms if args.mode == "fixed" else FrechetCellFilter(atoms)
            optimizer = FIRE(target, logfile=None)
            converged = bool(optimizer.run(fmax=args.fmax, steps=args.max_steps))

            final_energy = float(atoms.get_potential_energy())
            final_max_force = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
            final_volume = float(atoms.get_volume())
            final_structure = adaptor.get_structure(atoms)
            cif_str = final_structure.to(fmt="cif")

            rel = f"{int(sample_id):05d}.cif"
            (cif_dir / rel).write_text(cif_str, encoding="utf-8")

            frame.at[position, args.cif_name] = cif_str
            frame.at[position, "relax_converged"] = converged
            frame.at[position, "relax_steps"] = optimizer.get_number_of_steps()
            frame.at[position, "initial_energy_dpa_raw"] = initial_energy
            frame.at[position, "initial_max_force_eV_per_A"] = initial_max_force
            frame.at[position, "final_max_force_eV_per_A"] = final_max_force
            frame.at[position, "initial_volume_A3"] = initial_volume
            frame.at[position, "final_volume_A3"] = final_volume
            frame.at[position, "volume_change_fraction"] = final_volume / initial_volume - 1.0
            frame.at[position, "E_tot_pred_dpa_raw"] = final_energy
            frame.at[position, "dpa_done"] = True
            frame.at[position, "relaxed_cif_path"] = str(Path(cif_dir.name) / rel)
            frame.at[position, "relax_error"] = ""
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            lowered = message.lower()
            is_cuda_oom = any(marker in lowered for marker in (
                "cuda out of memory", "outofmemoryerror",
                "out-of-memory (oom)", "cublas_status_alloc_failed",
            ))
            frame.at[position, "relax_error"] = message
            frame.at[position, "relax_failure_type"] = (
                "cuda_oom" if is_cuda_oom else "relax_exception"
            )
            frame.at[position, "relax_retry_required"] = is_cuda_oom
            frame.at[position, "dpa_done"] = False
            # cif_str_unrelaxed remains available for audit.  Do not leave the
            # initial CIF in the relaxed column: downstream RMSD/retention code
            # would otherwise count a failed relaxation as a perfect match.
            frame.at[position, args.cif_name] = np.nan

        frame.at[position, "relax_mode"] = args.mode
        frame.at[position, "relax_wall_seconds"] = time.perf_counter() - crystal_started
        frame.at[position, "relax_done"] = True
        _atomic_progress(
            progress_file, status="running", pid=os.getpid(),
            completed=int(_bool_series(frame["relax_done"]).sum()), total=len(frame),
            active_position=None, last_completed_position=position,
            updated_monotonic=time.monotonic(),
        )
        if (position + 1) % args.save_interval == 0:
            _atomic_to_csv(frame, out)
            print(f"[{args.mode}] {position + 1}/{len(frame)} saved", flush=True)

    _atomic_to_csv(frame, out)
    _atomic_progress(
        progress_file, status="completed", pid=os.getpid(), completed=len(frame),
        total=len(frame), active_position=None, updated_monotonic=time.monotonic(),
    )
    summary = {
        "mode": args.mode,
        "input": str(src.resolve()),
        "output": str(out.resolve()),
        "cif_dir": str(cif_dir.resolve()),
        "total": len(frame),
        "converged": int(_bool_series(frame["relax_converged"]).sum()),
        "energies": int(frame["E_tot_pred_dpa_raw"].notna().sum()),
        "errors": int(frame["relax_error"].fillna("").ne("").sum()),
        "fmax": args.fmax,
        "max_steps": args.max_steps,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    out.with_suffix(".relax_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
