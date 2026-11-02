#!/usr/bin/env python3
"""Run the sequential DPA relaxer with a per-crystal progress watchdog.

Normal structures share one long-lived DPA process.  If the currently active
structure makes no completed-row progress before the timeout, the whole worker
process group is killed, that row is marked as timed out, and --resume starts a
fresh worker at the next row.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def bools(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def native_scalar(value: object) -> object:
    """Convert NumPy/Pandas scalar IDs before JSON serialization."""
    return value.item() if isinstance(value, np.generic) else value


def ensure_failure_dtypes(frame: pd.DataFrame) -> None:
    """CSV reload turns all-empty text columns into float NaN columns."""
    for column in ("relax_error", "relax_watchdog_note", "relax_failure_type"):
        if column not in frame:
            frame[column] = ""
        frame[column] = frame[column].astype(object)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_json(value: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def read_progress(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def initialize(source: Path, output: Path, cif_name: str, mode: str) -> None:
    if output.is_file():
        frame = pd.read_csv(output)
        expected = len(pd.read_csv(source, usecols=[cif_name]))
        if len(frame) != expected:
            raise RuntimeError(f"resume 行数不一致: {len(frame)} != {expected}")
        return
    frame = pd.read_csv(source)
    if cif_name not in frame:
        raise RuntimeError(f"输入缺少 {cif_name}: {source}")
    if "cif_str_unrelaxed" not in frame:
        frame["cif_str_unrelaxed"] = frame[cif_name]
    defaults: dict[str, object] = {
        "relax_mode": mode,
        "relax_done": False,
        "relax_converged": False,
        "relax_steps": np.nan,
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
        "relax_timeout": False,
        "relax_watchdog_note": "",
        "relax_failure_type": "",
        "relax_retry_required": False,
    }
    for column, value in defaults.items():
        if column not in frame:
            frame[column] = value
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_csv(frame, output)


def progress(output: Path) -> tuple[int, int | None, int]:
    frame = pd.read_csv(output)
    done = bools(frame["relax_done"])
    pending = np.flatnonzero(~done.to_numpy())
    return int(done.sum()), int(pending[0]) if len(pending) else None, len(frame)


def checkpoint_manifest(args, base_elapsed: float, current_started: float,
                        launches: int, timeout_records: list[dict[str, object]],
                        worker_failure_records: list[dict[str, object]],
                        status: str, completed: int, total: int) -> None:
    atomic_json({
        "status": status, "started_or_resumed_at": now(),
        "duration_seconds": base_elapsed + (time.perf_counter() - current_started),
        "input": str(args.input), "output": str(args.output),
        "cif_dir": str(args.cif_dir), "total": total, "relax_done": completed,
        "timeouts": len(timeout_records), "timeout_records": timeout_records,
        "worker_failures": len(worker_failure_records),
        "worker_failure_records": worker_failure_records,
        "worker_launches_this_invocation": launches,
        "crystal_timeout_seconds": args.crystal_timeout_seconds,
        "fmax": args.fmax, "max_steps": args.max_steps, "gpu": args.gpu,
    }, args.manifest)


def terminate_group(process: subprocess.Popen, grace_seconds: float) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=grace_seconds)
        return
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"无法终止弛豫进程组 pid={process.pid}")


def mark_timeout(output: Path, position: int, timeout_seconds: float,
                 cif_name: str) -> object:
    frame = pd.read_csv(output)
    ensure_failure_dtypes(frame)
    if bools(frame["relax_done"]).iloc[position]:
        value = frame.at[position, "request_id"] if "request_id" in frame else position
        return native_scalar(value)
    value = frame.at[position, "request_id"] if "request_id" in frame else position
    request_id = native_scalar(value)
    frame.at[position, "relax_done"] = True
    frame.at[position, "relax_converged"] = False
    frame.at[position, "dpa_done"] = False
    frame.at[position, "relax_timeout"] = True
    frame.at[position, "relax_failure_type"] = "timeout"
    frame.at[position, "relax_retry_required"] = False
    # Preserve cif_str_unrelaxed but exclude this row from relaxed SUN and
    # before/after structure metrics.
    frame.at[position, cif_name] = np.nan
    frame.at[position, "relax_error"] = (
        f"TimeoutError: no completed-row progress for {timeout_seconds:g} seconds"
    )
    frame.at[position, "relax_watchdog_note"] = f"force-killed worker at {now()}"
    atomic_csv(frame, output)
    return request_id


def classify_worker_failure(log: Path, launch_offset: int) -> tuple[str, str]:
    """Classify only text emitted by the most recent worker launch."""
    try:
        with log.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(launch_offset)
            tail = handle.read()[-20000:]
    except OSError as exc:
        return "worker_error", f"unable to read worker log: {exc}"
    lowered = tail.lower()
    oom_markers = (
        "cuda out of memory", "outofmemoryerror",
        "still throws an out-of-memory (oom) error", "cublas_status_alloc_failed",
    )
    if any(marker in lowered for marker in oom_markers):
        return "cuda_oom", tail
    return "worker_error", tail


def mark_worker_failure(output: Path, position: int, failure_type: str,
                        returncode: int, cif_name: str) -> object:
    frame = pd.read_csv(output)
    ensure_failure_dtypes(frame)
    if bools(frame["relax_done"]).iloc[position]:
        value = frame.at[position, "request_id"] if "request_id" in frame else position
        return native_scalar(value)
    value = frame.at[position, "request_id"] if "request_id" in frame else position
    request_id = native_scalar(value)
    frame.at[position, "relax_done"] = True
    frame.at[position, "relax_converged"] = False
    frame.at[position, "dpa_done"] = False
    frame.at[position, "relax_timeout"] = False
    frame.at[position, "relax_failure_type"] = failure_type
    frame.at[position, "relax_retry_required"] = failure_type == "cuda_oom"
    frame.at[position, cif_name] = np.nan
    frame.at[position, "relax_error"] = (
        f"WorkerError[{failure_type}]: relax subprocess exited code={returncode}"
    )
    frame.at[position, "relax_watchdog_note"] = f"worker failure recorded at {now()}"
    atomic_csv(frame, output)
    return request_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cif-dir", type=Path, required=True)
    parser.add_argument("--worker-python", type=Path, required=True)
    parser.add_argument("--relax-script", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mode", choices=("fixed", "cell"), default="cell")
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--crystal-timeout-seconds", type=float, default=3600)
    parser.add_argument("--poll-seconds", type=float, default=2)
    parser.add_argument("--kill-grace-seconds", type=float, default=30)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--cuda-memory-fraction", type=float, default=None)
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--cif-name", default="cif_str")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    for attribute in ("input", "output", "cif_dir", "worker_python", "relax_script",
                      "model", "log", "manifest"):
        setattr(args, attribute, getattr(args, attribute).expanduser().resolve())
    if args.crystal_timeout_seconds <= args.poll_seconds:
        raise SystemExit("单晶体超时必须大于轮询间隔")
    initialize(args.input, args.output, args.cif_name, args.mode)
    args.cif_dir.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    progress_file = args.output.with_suffix(args.output.suffix + ".progress.json")

    previous = {}
    if args.manifest.is_file():
        try:
            previous = json.loads(args.manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
    completed_before, pending_before, _ = progress(args.output)
    if pending_before is None and previous.get("status") == "completed":
        print(json.dumps(previous, indent=2, ensure_ascii=False), flush=True)
        return
    base_elapsed = float(previous.get("duration_seconds", 0.0) or 0.0)
    started = time.perf_counter()
    started_at = now()
    timeout_records: list[dict[str, object]] = []
    timeout_records.extend(previous.get("timeout_records", []))
    worker_failure_records: list[dict[str, object]] = []
    worker_failure_records.extend(previous.get("worker_failure_records", []))
    launches = 0
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": args.gpu, "PYTHONUNBUFFERED": "1"})

    with args.log.open("a", encoding="utf-8") as log:
        while True:
            completed, pending, total = progress(args.output)
            if pending is None:
                break
            launches += 1
            timed_out_this_launch = False
            command = [
                str(args.worker_python), "-u", str(args.relax_script),
                "--input", str(args.input), "--output", str(args.output),
                "--model", str(args.model), "--mode", args.mode,
                "--fmax", str(args.fmax), "--max_steps", str(args.max_steps),
                "--save_interval", str(args.save_interval),
                "--progress_file", str(progress_file),
                "--resume", "--require_gpu",
                "--cif_dir", str(args.cif_dir), "--cif_name", args.cif_name,
            ]
            if args.cuda_memory_fraction is not None:
                command.extend([
                    "--cuda_memory_fraction", str(args.cuda_memory_fraction),
                ])
            log.write(f"\n[WATCHDOG LAUNCH {launches}] {now()} pending_position={pending}\n")
            log.write("command=" + " ".join(command) + "\n")
            log.flush()
            launch_log_offset = log.tell()
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, cwd=args.relax_script.parent.parent,
                env=env, start_new_session=True,
            )
            last_completed = completed
            last_progress = time.monotonic()
            last_mtime = args.output.stat().st_mtime_ns
            last_progress_mtime = progress_file.stat().st_mtime_ns if progress_file.exists() else 0
            current_pending = pending
            last_checkpoint = time.monotonic()
            checkpoint_manifest(
                args, base_elapsed, started, launches, timeout_records,
                worker_failure_records,
                "running", completed, total,
            )
            while process.poll() is None:
                time.sleep(args.poll_seconds)
                current_mtime = args.output.stat().st_mtime_ns
                current_progress_mtime = (
                    progress_file.stat().st_mtime_ns if progress_file.exists() else 0
                )
                if current_progress_mtime != last_progress_mtime:
                    heartbeat = read_progress(progress_file)
                    last_progress_mtime = current_progress_mtime
                    active = heartbeat.get("active_position")
                    if active is not None:
                        current_pending = int(active)
                    heartbeat_completed = int(heartbeat.get("completed", last_completed))
                    if heartbeat_completed > last_completed:
                        last_completed = heartbeat_completed
                    last_progress = time.monotonic()
                if current_mtime != last_mtime:
                    current_completed, current_pending, _ = progress(args.output)
                    last_mtime = current_mtime
                    if current_completed > last_completed:
                        last_completed = current_completed
                        last_progress = time.monotonic()
                        print(f"[PROGRESS] {current_completed}/{total}", flush=True)
                    checkpoint_manifest(
                        args, base_elapsed, started, launches, timeout_records,
                        worker_failure_records,
                        "running", current_completed, total,
                    )
                    last_checkpoint = time.monotonic()
                elif time.monotonic() - last_checkpoint >= 60:
                    checkpoint_manifest(
                        args, base_elapsed, started, launches, timeout_records,
                        worker_failure_records,
                        "running", last_completed, total,
                    )
                    last_checkpoint = time.monotonic()
                if time.monotonic() - last_progress > args.crystal_timeout_seconds:
                    active_position = current_pending
                    terminate_group(process, args.kill_grace_seconds)
                    if active_position is not None:
                        request_id = mark_timeout(
                            args.output, active_position, args.crystal_timeout_seconds,
                            args.cif_name,
                        )
                        record = {
                            "position": active_position, "request_id": request_id,
                            "timeout_seconds": args.crystal_timeout_seconds,
                            "killed_at": now(), "worker_launch": launches,
                        }
                        timeout_records.append(record)
                        log.write("[WATCHDOG TIMEOUT] " + json.dumps(record) + "\n")
                        log.flush()
                        print(f"[TIMEOUT] request_id={request_id}; worker restarted", flush=True)
                    timed_out_this_launch = True
                    break
            if (not timed_out_this_launch and process.poll() is not None
                    and process.returncode != 0):
                log.flush()
                failure_type, failure_log = classify_worker_failure(
                    args.log, launch_log_offset
                )
                active_position = current_pending
                if failure_type != "cuda_oom" or active_position is None:
                    raise RuntimeError(
                        f"弛豫 worker 异常退出 code={process.returncode} "
                        f"type={failure_type}; 查看 {args.log}"
                    )
                request_id = mark_worker_failure(
                    args.output, active_position, failure_type,
                    int(process.returncode), args.cif_name,
                )
                record = {
                    "position": active_position, "request_id": request_id,
                    "failure_type": failure_type, "returncode": int(process.returncode),
                    "failed_at": now(), "worker_launch": launches,
                    "log_tail": failure_log[-4000:],
                }
                worker_failure_records.append(record)
                log.write("[WATCHDOG WORKER FAILURE] " + json.dumps(record) + "\n")
                log.flush()
                print(
                    f"[CUDA OOM] request_id={request_id}; queued for serial retry; "
                    "worker restarted",
                    flush=True,
                )

    elapsed = base_elapsed + (time.perf_counter() - started)
    frame = pd.read_csv(args.output)
    done = bools(frame["relax_done"])
    converged = bools(frame["relax_converged"])
    dpa_done = bools(frame["dpa_done"])
    result = {
        "status": "completed" if bool(done.all()) else "incomplete",
        "started_at": started_at, "finished_at": now(),
        "duration_seconds": elapsed, "input": str(args.input),
        "output": str(args.output), "cif_dir": str(args.cif_dir),
        "total": len(frame), "relax_done": int(done.sum()),
        "converged": int(converged.sum()), "dpa_done": int(dpa_done.sum()),
        "errors": int(frame["relax_error"].fillna("").astype(str).str.strip().ne("").sum()),
        "timeouts": len(timeout_records), "timeout_records": timeout_records,
        "worker_failures": len(worker_failure_records),
        "worker_failure_records": worker_failure_records,
        "worker_launches": launches, "crystal_timeout_seconds": args.crystal_timeout_seconds,
        "fmax": args.fmax, "max_steps": args.max_steps, "gpu": args.gpu,
    }
    atomic_json(result, args.manifest)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
