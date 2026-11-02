#!/usr/bin/env python3
"""Shared DPA relaxation worker controller used by the paper experiments."""
from __future__ import annotations

import os, sys

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Structure


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CRYSFV_PY = Path(os.environ.get("DELAC_PYTHON", sys.executable))
NEQV_PY = Path(os.environ.get("DPA_PYTHON", sys.executable))
DPA_MODEL = REPO / "mlff_choosing/dpa-3.1-mptrj.pth"
RELAX_SCRIPT = REPO / "script/dpa_cell_relax_csv.py"
WATCHDOG = HERE / "relax_with_watchdog.py"
DEFAULT_INPUT = (
    REPO / "result/final_experiments/experiment_speed/stage1/run_20260816_091609"
    / "stage2/scaled_rmsnorm_empirical_gaussian/seed_42/generated_success.csv"
)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_cif(value: object) -> Structure | None:
    if pd.isna(value):
        return None
    try:
        return Structure.from_str(str(value).replace("\\n", "\n"), fmt="cif")
    except Exception:
        return None


def estimated_costs(frame: pd.DataFrame, strategy: str, cutoff: float) -> pd.Series:
    """Pre-timing deterministic cost proxies used only for static scheduling."""
    if strategy == "round_robin":
        return pd.Series(np.ones(len(frame)), index=frame.index, dtype=float)
    values = []
    for value in frame["cif_str"]:
        structure = parse_cif(value)
        if structure is None:
            values.append(1.0)
            continue
        atoms = float(len(structure))
        if strategy == "lpt_atoms":
            values.append(atoms)
        elif strategy == "lpt_neighbors":
            # Directed periodic neighbor edges approximate DPA graph work/memory.
            centers, _, _, _ = structure.get_neighbor_list(cutoff)
            values.append(atoms + float(len(centers)))
        else:
            raise ValueError(f"unknown sharding strategy: {strategy}")
    return pd.Series(values, index=frame.index, dtype=float)


def make_shards(frame: pd.DataFrame, workers: int, strategy: str,
                cutoff: float) -> tuple[list[pd.DataFrame], dict]:
    costs = estimated_costs(frame, strategy, cutoff)
    if strategy == "round_robin":
        shards = [frame.iloc[worker::workers].copy() for worker in range(workers)]
    else:
        bins: list[list[int]] = [[] for _ in range(workers)]
        loads = [0.0] * workers
        ordered = sorted(frame.index, key=lambda idx: (-float(costs.loc[idx]), int(idx)))
        for idx in ordered:
            worker = min(range(workers), key=lambda item: (loads[item], item))
            bins[worker].append(idx)
            loads[worker] += float(costs.loc[idx])
        # Preserve source order inside each shard so request audit remains simple.
        shards = [frame.loc[sorted(indices)].copy() for indices in bins]
    loads = [float(costs.loc[shard.index].sum()) for shard in shards]
    return shards, {
        "strategy": strategy, "neighbor_cutoff_A": cutoff,
        "estimated_total_cost": float(costs.sum()),
        "estimated_worker_costs": loads,
        "worker_rows": [len(shard) for shard in shards],
        "estimated_load_imbalance": (
            max(loads) / (sum(loads) / len(loads)) - 1.0 if loads and sum(loads) else 0.0
        ),
    }
def prepare_fixed_input(source: Path, destination: Path, count: int) -> dict:
    """Create once, then refuse to silently change the selected workload."""
    if destination.is_file():
        frame = pd.read_csv(destination)
        if len(frame) != count:
            raise RuntimeError(f"existing fixed input has {len(frame)} rows, expected {count}")
        return {"path": str(destination), "rows": len(frame), "sha256": sha256(destination)}

    source_frame = pd.read_csv(source)
    if "cif_str" not in source_frame:
        raise RuntimeError(f"{source} has no cif_str column")
    if "request_id" not in source_frame:
        source_frame.insert(0, "request_id", range(len(source_frame)))
    selected = []
    for _, row in source_frame.iterrows():
        structure = parse_cif(row["cif_str"])
        if structure is None:
            continue
        copy = row.copy()
        copy["benchmark_num_atoms"] = len(structure)
        selected.append(copy)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"only {len(selected)} parseable CIFs in {source}; need {count}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selected).to_csv(destination, index=False)
    return {"path": str(destination), "rows": count, "sha256": sha256(destination)}


def gpu_snapshot(gpu: str) -> dict:
    command = [
        "nvidia-smi", f"--id={gpu}",
        "--query-gpu=name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=20, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError("nvidia-smi GPU preflight failed: " + result.stdout.strip())
    fields = [part.strip() for part in result.stdout.strip().split(",")]
    if len(fields) != 4:
        raise RuntimeError("unexpected nvidia-smi output: " + result.stdout.strip())
    return {
        "name": fields[0], "total_mib": int(fields[1]),
        "used_mib": int(fields[2]), "utilization_percent": int(fields[3]),
    }


def process_table() -> dict[int, int]:
    result = subprocess.run(["ps", "-e", "-o", "pid=,ppid="], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    table = {}
    for line in result.stdout.splitlines():
        try:
            pid, ppid = (int(value) for value in line.split())
            table[pid] = ppid
        except ValueError:
            continue
    return table


def descendants(roots: list[int]) -> set[int]:
    table = process_table()
    found = set(roots)
    changed = True
    while changed:
        changed = False
        for pid, ppid in table.items():
            if ppid in found and pid not in found:
                found.add(pid)
                changed = True
    return found


def terminate_owned(processes: list[subprocess.Popen], grace: float = 20.0) -> None:
    owned = descendants([process.pid for process in processes if process.poll() is None])
    for pid in sorted(owned, reverse=True):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        live = [pid for pid in owned if Path(f"/proc/{pid}").exists()]
        if not live:
            return
        time.sleep(0.25)
    for pid in sorted(owned, reverse=True):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def bools(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def native_scalar(value: object) -> object:
    return value.item() if isinstance(value, np.generic) else value


def safe_print(*values: object, **kwargs: object) -> None:
    """Progress output must never abort an expensive scientific run."""
    try:
        print(*values, **kwargs)
    except (BrokenPipeError, OSError):
        pass


def merge_outputs(config_dir: Path, workers: int, fixed: pd.DataFrame) -> tuple[Path, dict]:
    frames = []
    manifests = []
    for worker in range(workers):
        output = config_dir / f"worker_{worker:02d}/relaxed.csv"
        manifest = config_dir / f"worker_{worker:02d}/manifest.json"
        if not output.is_file() or not manifest.is_file():
            raise RuntimeError(f"worker {worker} output incomplete")
        frames.append(pd.read_csv(output))
        manifests.append(json.loads(manifest.read_text(encoding="utf-8")))
    merged = pd.concat(frames, ignore_index=True).sort_values("request_id").reset_index(drop=True)
    expected_ids = fixed["request_id"].astype(str).tolist()
    actual_ids = merged["request_id"].astype(str).tolist()
    if actual_ids != sorted(expected_ids, key=lambda value: float(value)):
        # Sorting strings and numbers differs; compare sets plus cardinality instead.
        if len(actual_ids) != len(expected_ids) or set(actual_ids) != set(expected_ids):
            raise RuntimeError("merged request_id set differs from fixed input")
    merged_path = config_dir / "relaxed_merged.csv"
    merged.to_csv(merged_path, index=False)
    done = bools(merged["relax_done"])
    converged = bools(merged["relax_converged"])
    dpa_done = bools(merged["dpa_done"])
    steps = pd.to_numeric(merged.get("relax_steps"), errors="coerce")
    atoms = pd.to_numeric(merged.get("benchmark_num_atoms"), errors="coerce")
    summary = {
        "total": len(merged), "relax_done": int(done.sum()),
        "converged": int(converged.sum()), "dpa_done": int(dpa_done.sum()),
        "errors": int(merged["relax_error"].fillna("").astype(str).str.strip().ne("").sum()),
        "timeouts": sum(int(item.get("timeouts", 0)) for item in manifests),
        "total_relax_steps": float(steps.sum()),
        "mean_relax_steps": float(steps.mean()) if steps.notna().any() else None,
        "total_atom_steps": float((steps * atoms).sum()),
        "worker_manifests": manifests,
    }
    return merged_path, summary


def consistency(candidate: Path, baseline: Path) -> dict:
    left = pd.read_csv(baseline).set_index("request_id")
    right = pd.read_csv(candidate).set_index("request_id")
    common = left.index.intersection(right.index)
    e0 = pd.to_numeric(left.loc[common, "E_tot_pred_dpa_raw"], errors="coerce")
    e1 = pd.to_numeric(right.loc[common, "E_tot_pred_dpa_raw"], errors="coerce")
    valid = e0.notna() & e1.notna()
    delta = (e0[valid] - e1[valid]).abs()
    c0 = bools(left.loc[common, "relax_converged"])
    c1 = bools(right.loc[common, "relax_converged"])
    s0 = pd.to_numeric(left.loc[common, "relax_steps"], errors="coerce")
    s1 = pd.to_numeric(right.loc[common, "relax_steps"], errors="coerce")
    return {
        "common": len(common), "energy_pairs": int(valid.sum()),
        "max_abs_final_energy_difference_eV": float(delta.max()) if len(delta) else None,
        "mean_abs_final_energy_difference_eV": float(delta.mean()) if len(delta) else None,
        "convergence_mismatches": int((c0 != c1).sum()),
        "step_mismatches": int(((s0 != s1) & s0.notna() & s1.notna()).sum()),
    }


def retry_rows(config_dir: Path, workers: int) -> list[tuple[int, object]]:
    rows = []
    for worker in range(workers):
        output = config_dir / f"worker_{worker:02d}/relaxed.csv"
        if not output.is_file():
            continue
        frame = pd.read_csv(output)
        if "relax_retry_required" not in frame:
            continue
        for position in np.flatnonzero(bools(frame["relax_retry_required"]).to_numpy()):
            value = frame.at[position, "request_id"] if "request_id" in frame else position
            request_id = native_scalar(value)
            rows.append((worker, request_id))
    return rows


def replace_retry_result(worker_output: Path, request_id: object,
                         retry_output: Path, source_worker: int,
                         concurrent_workers: int) -> dict:
    frame = pd.read_csv(worker_output)
    retried = pd.read_csv(retry_output)
    for column in ("relax_error", "relax_watchdog_note", "relax_failure_type"):
        if column in frame:
            frame[column] = frame[column].astype(object)
    source_mask = frame["request_id"].astype(str) == str(request_id)
    retry_mask = retried["request_id"].astype(str) == str(request_id)
    if int(source_mask.sum()) != 1 or int(retry_mask.sum()) != 1:
        raise RuntimeError(f"retry request_id={request_id} is not unique")
    retry_row = retried.loc[retry_mask].iloc[0]
    success = bools(pd.Series([retry_row.get("dpa_done", False)])).iloc[0]
    for column in retried.columns:
        if column not in frame:
            frame[column] = np.nan
        frame.loc[source_mask, column] = retry_row[column]
    frame.loc[source_mask, "relax_retry_required"] = False
    if not success:
        frame.loc[source_mask, "relax_failure_type"] = "intrinsic_cuda_oom_after_serial_retry"
        frame.loc[source_mask, "relax_error"] = (
            "IntrinsicCUDAOOM: failed during concurrent execution and single-worker retry"
        )
        frame.loc[source_mask, "relax_watchdog_note"] = (
            f"single-worker retry completed at {now()} "
            f"after workers={concurrent_workers}"
        )
    temporary = worker_output.with_suffix(worker_output.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(worker_output)
    return {
        "request_id": native_scalar(request_id), "source_worker": source_worker,
        "concurrent_workers": concurrent_workers,
        "serial_retry_success": bool(success),
        "final_failure_type": None if success else "intrinsic_cuda_oom_after_serial_retry",
        "retry_output": str(retry_output),
    }


def run_serial_retries(args: argparse.Namespace, config_dir: Path, workers: int,
                       fixed: pd.DataFrame, environment: dict[str, str],
                       serial_memory_fraction: float) -> list[dict]:
    pending = retry_rows(config_dir, workers)
    records = []
    if not pending:
        return records
    safe_print(f"[SERIAL RETRY] workers={workers} OOM rows={len(pending)}", flush=True)
    for worker, request_id in pending:
        worker_output = config_dir / f"worker_{worker:02d}/relaxed.csv"
        if workers == 1:
            # The original attempt was already isolated and used the full
            # single-worker budget; a second identical attempt adds no evidence.
            frame = pd.read_csv(worker_output)
            mask = frame["request_id"].astype(str) == str(request_id)
            frame.loc[mask, "relax_retry_required"] = False
            frame.loc[mask, "relax_failure_type"] = "intrinsic_cuda_oom_single_worker"
            frame.loc[mask, "relax_error"] = (
                "IntrinsicCUDAOOM: failed with the full single-worker memory budget"
            )
            temporary = worker_output.with_suffix(worker_output.suffix + ".tmp")
            frame.to_csv(temporary, index=False)
            temporary.replace(worker_output)
            records.append({
                "request_id": request_id, "source_worker": worker,
                "serial_retry_success": False,
                "serial_retry_skipped": "original attempt already single-worker",
                "final_failure_type": "intrinsic_cuda_oom_single_worker",
            })
            continue
        safe_id = str(request_id).replace("/", "_")
        retry_dir = config_dir / "serial_retry" / f"request_{safe_id}"
        retry_dir.mkdir(parents=True, exist_ok=True)
        retry_input = retry_dir / "input.csv"
        selected = fixed.loc[fixed["request_id"].astype(str) == str(request_id)]
        if len(selected) != 1:
            raise RuntimeError(f"fixed input request_id={request_id} is not unique")
        selected.to_csv(retry_input, index=False)
        retry_output = retry_dir / "relaxed.csv"
        command = [
            str(CRYSFV_PY), "-u", str(WATCHDOG),
            "--input", str(retry_input), "--output", str(retry_output),
            "--cif-dir", str(retry_dir / "relaxed_cifs"),
            "--worker-python", str(NEQV_PY), "--relax-script", str(RELAX_SCRIPT),
            "--model", str(DPA_MODEL), "--mode", "cell", "--fmax", str(args.fmax),
            "--max-steps", str(args.max_steps), "--crystal-timeout-seconds",
            str(args.serial_retry_timeout_seconds), "--gpu", args.gpu,
            "--cuda-memory-fraction", str(serial_memory_fraction),
            "--log", str(retry_dir / "relaxation.log"),
            "--manifest", str(retry_dir / "manifest.json"),
        ]
        retry_started = time.perf_counter()
        with (retry_dir / "controller.log").open("a", encoding="utf-8") as handle:
            result = subprocess.run(
                command, cwd=REPO, env=environment, stdout=handle,
                stderr=subprocess.STDOUT, check=False,
            )
        if result.returncode != 0 or not retry_output.is_file():
            raise RuntimeError(
                f"serial retry controller failed request_id={request_id} "
                f"code={result.returncode}"
            )
        record = replace_retry_result(
            worker_output, request_id, retry_output, worker, workers
        )
        record["serial_memory_fraction"] = serial_memory_fraction
        record["retry_wall_seconds"] = time.perf_counter() - retry_started
        records.append(record)
    return records


def run_configuration(args: argparse.Namespace, fixed_path: Path, workers: int,
                      root: Path) -> dict:
    configuration_name = getattr(args, "configuration_name", f"workers_{workers}")
    config_dir = root / configuration_name
    complete_path = config_dir / "summary.json"
    previous: dict = {}
    if complete_path.is_file():
        previous = json.loads(complete_path.read_text(encoding="utf-8"))
        if previous.get("status") == "completed":
            print(f"[RESUME SKIP] workers={workers}", flush=True)
            return previous
    config_dir.mkdir(parents=True, exist_ok=True)
    fixed = pd.read_csv(fixed_path)
    processes: list[subprocess.Popen] = []
    logs = []
    commands = []
    shards, sharding = make_shards(
        fixed, workers, getattr(args, "sharding_strategy", "round_robin"),
        getattr(args, "neighbor_cutoff", 6.0),
    )
    workers_already_complete = True
    for worker in range(workers):
        worker_dir = config_dir / f"worker_{worker:02d}"
        worker_dir.mkdir(parents=True, exist_ok=True)
        shard = shards[worker]
        shard_path = worker_dir / "input.csv"
        shard.to_csv(shard_path, index=False)
        existing_output = worker_dir / "relaxed.csv"
        existing_manifest = worker_dir / "manifest.json"
        complete = False
        if existing_output.is_file() and existing_manifest.is_file():
            try:
                manifest_data = json.loads(existing_manifest.read_text(encoding="utf-8"))
                output_rows = pd.read_csv(existing_output, usecols=["request_id"])
                complete = (manifest_data.get("status") == "completed" and len(output_rows) == len(shard))
            except (OSError, ValueError, json.JSONDecodeError):
                complete = False
        workers_already_complete &= complete
        command = [
            str(CRYSFV_PY), "-u", str(WATCHDOG),
            "--input", str(shard_path), "--output", str(worker_dir / "relaxed.csv"),
            "--cif-dir", str(worker_dir / "relaxed_cifs"),
            "--worker-python", str(NEQV_PY), "--relax-script", str(RELAX_SCRIPT),
            "--model", str(DPA_MODEL), "--mode", "cell", "--fmax", str(args.fmax),
            "--max-steps", str(args.max_steps), "--crystal-timeout-seconds",
            str(args.crystal_timeout_seconds), "--gpu", args.gpu,
            "--save-interval", str(getattr(args, "save_interval", 1)),
            "--log", str(worker_dir / "relaxation.log"),
            "--manifest", str(worker_dir / "manifest.json"),
        ]
        commands.append(command)
    if args.dry_run:
        return {"status": "dry_run", "workers": workers,
                "sharding": sharding, "configuration_name": configuration_name,
                "commands": [[str(value) for value in command] for command in commands]}

    baseline_gpu = gpu_snapshot(args.gpu)
    if baseline_gpu["used_mib"] >= args.memory_limit_mib:
        raise RuntimeError(
            f"GPU already uses {baseline_gpu['used_mib']} MiB, above limit "
            f"{args.memory_limit_mib} MiB"
        )
    allocator_budget_mib = (
        args.memory_limit_mib - baseline_gpu["used_mib"] - args.memory_safety_mib
    )
    if allocator_budget_mib <= 0:
        raise RuntimeError(
            f"no CUDA allocator budget remains after baseline and safety reserve: "
            f"{allocator_budget_mib} MiB"
        )
    per_worker_fraction = allocator_budget_mib / workers / baseline_gpu["total_mib"]
    serial_memory_fraction = allocator_budget_mib / baseline_gpu["total_mib"]
    if not 0.0 < per_worker_fraction <= 1.0:
        raise RuntimeError(f"invalid per-worker CUDA memory fraction {per_worker_fraction}")
    if workers_already_complete:
        commands = []
    else:
        for command in commands:
            command.extend(["--cuda-memory-fraction", str(per_worker_fraction)])
    base_elapsed = float(previous.get("wall_seconds", 0.0) or 0.0)
    # If the controller was killed while its independent workers continued,
    # summary.json can lag behind the completed worker manifests.  Recover a
    # conservative wall-clock lower bound before resuming so timing is not
    # silently under-reported.  Concurrent worker durations must use max, not
    # sum.
    recovered_worker_seconds = []
    for worker in range(workers):
        manifest = config_dir / f"worker_{worker:02d}/manifest.json"
        if not manifest.is_file():
            continue
        try:
            manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
            duration = float(manifest_data.get("duration_seconds", 0.0) or 0.0)
            if duration > 0:
                recovered_worker_seconds.append(duration)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    if recovered_worker_seconds:
        recovered = max(recovered_worker_seconds)
        if recovered > base_elapsed:
            safe_print(
                f"[RESUME TIMING] summary={base_elapsed:.3f}s "
                f"worker_manifest_lower_bound={recovered:.3f}s",
                flush=True,
            )
            base_elapsed = recovered
    started = time.perf_counter()
    started_at = now()
    samples = []
    try:
        environment = os.environ.copy()
        environment.update({
            "CUDA_VISIBLE_DEVICES": args.gpu, "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": str(args.omp_threads),
            "DP_INTRA_OP_PARALLELISM_THREADS": str(args.dp_intra_threads),
            "DP_INTER_OP_PARALLELISM_THREADS": str(args.dp_inter_threads),
        })
        for worker, command in enumerate(commands):
            handle = (config_dir / f"worker_{worker:02d}/controller.log").open(
                "a", encoding="utf-8"
            )
            logs.append(handle)
            processes.append(subprocess.Popen(
                command, cwd=REPO, env=environment, stdout=handle,
                stderr=subprocess.STDOUT, start_new_session=True,
            ))
        while any(process.poll() is None for process in processes):
            snap = gpu_snapshot(args.gpu)
            snap["elapsed_seconds"] = base_elapsed + time.perf_counter() - started
            samples.append(snap)
            atomic_json({
                "status": "running", "workers": workers,
                "started_at": previous.get("started_at", started_at),
                "last_resumed_at": started_at,
                "wall_seconds": snap["elapsed_seconds"],
                "gpu": {"last_sample": snap, "memory_limit_mib": args.memory_limit_mib},
            }, complete_path)
            if snap["used_mib"] > args.memory_limit_mib:
                terminate_owned(processes)
                raise RuntimeError(
                    f"GPU memory hard limit exceeded: {snap['used_mib']} > "
                    f"{args.memory_limit_mib} MiB"
                )
            time.sleep(args.monitor_interval)
        codes = [process.wait() for process in processes]
        if any(code != 0 for code in codes):
            raise RuntimeError(f"worker controllers failed: exit codes={codes}")
    except BaseException:
        terminate_owned(processes)
        raise
    finally:
        for handle in logs:
            handle.close()
    primary_active_seconds = time.perf_counter() - started
    if workers_already_complete and recovered_worker_seconds:
        primary_wall_seconds = max(recovered_worker_seconds)
    else:
        primary_wall_seconds = base_elapsed + primary_active_seconds
    retry_started = time.perf_counter()
    serial_retry_records = run_serial_retries(
        args, config_dir, workers, fixed, environment, serial_memory_fraction
    )
    serial_retry_wall_seconds = time.perf_counter() - retry_started
    merge_started = time.perf_counter()
    merged, relaxed = merge_outputs(config_dir, workers, fixed)
    merge_wall_seconds = time.perf_counter() - merge_started
    corrected_seconds = primary_wall_seconds + serial_retry_wall_seconds + merge_wall_seconds
    elapsed = base_elapsed + time.perf_counter() - started
    peak = max((item["used_mib"] for item in samples), default=baseline_gpu["used_mib"])
    utils = [item["utilization_percent"] for item in samples]
    result = {
        "status": "completed", "workers": workers,
        "started_at": previous.get("started_at", started_at),
        "last_resumed_at": started_at, "finished_at": now(),
        "wall_seconds": elapsed,
        "T_stage2_corrected_seconds": corrected_seconds,
        "structures_per_second": len(fixed) / elapsed,
        "atom_steps_per_second": relaxed["total_atom_steps"] / elapsed,
        "gpu": {
            "device": baseline_gpu["name"], "baseline_used_mib": baseline_gpu["used_mib"],
            "peak_total_used_mib": peak, "peak_incremental_mib": peak - baseline_gpu["used_mib"],
            "memory_limit_mib": args.memory_limit_mib,
            "memory_safety_mib": args.memory_safety_mib,
            "per_worker_cuda_memory_fraction": per_worker_fraction,
            "serial_retry_cuda_memory_fraction": serial_memory_fraction,
            "mean_sampled_utilization_percent": float(np.mean(utils)) if utils else None,
            "peak_sampled_utilization_percent": max(utils) if utils else None,
            "monitor_samples": len(samples),
        },
        "protocol": {
            "model": str(DPA_MODEL), "mode": "cell", "optimizer": "FIRE",
            "cell_filter": "FrechetCellFilter", "fmax_eV_per_A": args.fmax,
            "max_steps": args.max_steps, "crystal_timeout_seconds": args.crystal_timeout_seconds,
            "omp_threads_per_worker": args.omp_threads,
            "dp_intra_threads_per_worker": args.dp_intra_threads,
            "dp_inter_threads_per_worker": args.dp_inter_threads,
            "csv_save_interval": getattr(args, "save_interval", 1),
        },
        "timing_corrected": {
            "definition": "primary two-worker active wall + required serial OOM retries + merge; excludes downtime and controller restart/re-scan overhead",
            "primary_two_worker_wall_seconds": primary_wall_seconds,
            "serial_retry_wall_seconds": serial_retry_wall_seconds,
            "merge_wall_seconds": merge_wall_seconds,
            "T_stage2_corrected_seconds": corrected_seconds,
            "workers_already_complete_on_resume": workers_already_complete,
        },
        "sharding": sharding,
        "relaxation": relaxed, "serial_retry_records": serial_retry_records,
        "merged_output": str(merged),
    }
    atomic_json(result, complete_path)
    return result


