#!/usr/bin/env python3
"""Per-crystal, timeout-bounded and resumable relaxation metrics.

The numerical definitions and final summary are delegated to
``ftcp_recon.relaxation_metrics``.  This wrapper only changes execution and
checkpointing: one isolated child process per structure pair, a hard timeout,
and one JSON checkpoint per request_id.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import queue
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


REPO = Path(__file__).resolve().parents[2]
FTCP_ROOT = REPO / "third_party/FTCP"
sys.path.insert(0, str(FTCP_ROOT))
from ftcp_recon.relaxation_metrics import (  # noqa: E402
    compute_details,
    summarize,
    summary_markdown,
)


SUMMARY_COLUMNS_DEFAULTS: dict[str, Any] = {
    "paired_relaxed_row": False,
    "initial_structure_ok": False,
    "relaxed_structure_ok": False,
    "structure_pair_ok": False,
    "analysis_error": "",
    "analysis_timeout": False,
    "analysis_elapsed_seconds": math.nan,
    "relax_done": False,
    "relax_converged": False,
    "relax_steps": math.nan,
    "relax_error": "",
    "initial_energy_dpa_raw": math.nan,
    "E_tot_pred_dpa_raw": math.nan,
    "initial_max_force_eV_per_A": math.nan,
    "final_max_force_eV_per_A": math.nan,
    "max_force_drop_eV_per_A": math.nan,
    "initial_volume_A3": math.nan,
    "final_volume_A3": math.nan,
    "volume_change_fraction": math.nan,
    "volume_change_fraction_recomputed": math.nan,
    "is_valid": False,
    "hull_exclude_reason": "",
    "dpa_done": False,
    "mattergen_rmsd_A": math.nan,
    "mattergen_rmsd_alignment_succeeded": False,
    "mattergen_rmsd_used_fallback": False,
    "energy_drop_eV_per_atom": math.nan,
    "retained_strict": False,
    "retained_loose": False,
    "space_group_preserved": False,
    "lattice_a_change_fraction": math.nan,
    "lattice_b_change_fraction": math.nan,
    "lattice_c_change_fraction": math.nan,
    "alpha_change_abs_deg": math.nan,
    "beta_change_abs_deg": math.nan,
    "gamma_change_abs_deg": math.nan,
}


def atomic_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def json_safe_record(record: dict[str, Any]) -> dict[str, Any]:
    # Pandas normalizes numpy scalars, NaN and timestamps into JSON-compatible
    # primitives consistently with the project's CSV/JSON artifacts.
    return json.loads(pd.DataFrame([record]).to_json(orient="records"))[0]


def item_name(value: Any) -> str:
    text = str(value)
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)
    return safe[:180] or "empty_id"


def worker(initial_row: dict[str, Any], relaxed_row: dict[str, Any] | None,
           id_column: str, output_queue: mp.Queue) -> None:
    start = time.monotonic()
    try:
        initial = pd.DataFrame([initial_row])
        relaxed = (pd.DataFrame([relaxed_row]) if relaxed_row is not None
                   else pd.DataFrame(columns=[id_column]))
        record = compute_details(initial, relaxed, id_column).iloc[0].to_dict()
        record["analysis_elapsed_seconds"] = time.monotonic() - start
        record["analysis_timeout"] = False
        output_queue.put(("ok", record))
    except BaseException as exc:
        output_queue.put((
            "error",
            f"{type(exc).__name__}:{exc}",
            time.monotonic() - start,
        ))


def bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value) if not pd.isna(value) else False


def fallback_record(identifier: Any, id_column: str, error: str,
                    elapsed: float, timed_out: bool,
                    initial_row: dict[str, Any],
                    relaxed_row: dict[str, Any] | None) -> dict[str, Any]:
    result = dict(SUMMARY_COLUMNS_DEFAULTS)
    result[id_column] = identifier
    result["analysis_error"] = error
    result["analysis_timeout"] = timed_out
    result["analysis_elapsed_seconds"] = elapsed
    result["paired_relaxed_row"] = relaxed_row is not None
    # A StructureMatcher timeout must not erase cheap relaxation facts such as
    # convergence, steps, force and energies. Structure-derived pair metrics
    # remain false/NaN, so the pathological crystal is skipped only where the
    # expensive analysis could not finish.
    if relaxed_row is not None:
        for name in (
            "relax_done", "relax_converged", "relax_steps", "relax_error",
            "initial_energy_dpa_raw", "E_tot_pred_dpa_raw",
            "initial_max_force_eV_per_A", "final_max_force_eV_per_A",
            "initial_volume_A3", "final_volume_A3", "volume_change_fraction",
            "is_valid", "hull_exclude_reason", "dpa_done",
        ):
            if name in relaxed_row:
                result[name] = relaxed_row[name]
        for name in ("relax_done", "relax_converged", "is_valid", "dpa_done"):
            result[name] = bool_value(result[name])
    return result


def load_checkpoints(items_dir: Path, id_column: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(items_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            records[str(record[id_column])] = record
        except Exception as exc:
            print(f"[CHECKPOINT INVALID] {path}: {exc}", flush=True)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", required=True, type=Path)
    parser.add_argument("--relaxed", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--id-column", default="candidate_id")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument(
        "--timeout-seconds", type=int, default=3600,
        help="Hard wall-time limit for one crystal pair (default: 1 hour)",
    )
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    items_dir = args.output_dir / f"{args.tag}.relaxation_metrics.items"
    items_dir.mkdir(parents=True, exist_ok=True)
    initial = pd.read_csv(args.initial)
    relaxed = pd.read_csv(args.relaxed)
    if args.id_column not in initial or args.id_column not in relaxed:
        raise SystemExit(f"Both CSVs must contain {args.id_column!r}")
    if initial[args.id_column].duplicated().any() or relaxed[args.id_column].duplicated().any():
        raise SystemExit(f"{args.id_column!r} must be unique")

    relaxed_by_id = {
        str(row[args.id_column]): row.to_dict() for _, row in relaxed.iterrows()
    }
    records = load_checkpoints(items_dir, args.id_column)
    total = len(initial)
    print(
        f"[RESUME] completed={len(records)}/{total}, timeout={args.timeout_seconds}s, "
        f"items={items_dir}", flush=True,
    )
    context = mp.get_context("fork")
    for position, (_, row) in enumerate(initial.iterrows(), 1):
        identifier = row[args.id_column]
        key = str(identifier)
        if key in records:
            continue
        output_queue = context.Queue(maxsize=1)
        process = context.Process(
            target=worker,
            args=(row.to_dict(), relaxed_by_id.get(key), args.id_column, output_queue),
        )
        started = time.monotonic()
        process.start()
        process.join(args.timeout_seconds)
        elapsed = time.monotonic() - started
        if process.is_alive():
            process.terminate()
            process.join(10)
            if process.is_alive():
                process.kill()
                process.join()
            record = fallback_record(
                identifier, args.id_column,
                f"analysis_timeout_after_{args.timeout_seconds}s",
                elapsed, True, row.to_dict(), relaxed_by_id.get(key),
            )
            outcome = "TIMEOUT"
        else:
            try:
                message = output_queue.get(timeout=5)
            except queue.Empty:
                message = ("error", f"worker_exit_{process.exitcode}_without_result", elapsed)
            if message[0] == "ok":
                record = message[1]
                outcome = "OK"
            else:
                record = fallback_record(
                    identifier, args.id_column, message[1], float(message[2]), False,
                    row.to_dict(), relaxed_by_id.get(key),
                )
                outcome = "ERROR"
        output_queue.close()
        record = json_safe_record(record)
        records[key] = record
        checkpoint = items_dir / f"{item_name(identifier)}.json"
        atomic_text(checkpoint, json.dumps(record, indent=2, ensure_ascii=False))
        print(
            f"[{position}/{total}] id={identifier} {outcome} "
            f"elapsed={elapsed:.2f}s saved={checkpoint.name}", flush=True,
        )

    # Preserve original initial-row order in the canonical details artifact.
    ordered = [records[str(value)] for value in initial[args.id_column]]
    details = pd.DataFrame(ordered)
    for column, default in SUMMARY_COLUMNS_DEFAULTS.items():
        if column not in details:
            details[column] = default
    details_path = args.output_dir / f"{args.tag}.relaxation_metrics.csv"
    details.to_csv(details_path, index=False)
    summary = summarize(details, args.initial, args.relaxed, args.max_steps)
    timeout_count = int(details["analysis_timeout"].fillna(False).astype(bool).sum())
    summary["timeout_policy"] = {
        "per_crystal_timeout_seconds": args.timeout_seconds,
        "timed_out_crystals": timeout_count,
        "execution": "one isolated process per crystal pair",
        "checkpoint_items": str(items_dir.resolve()),
    }
    summary_path = args.output_dir / f"{args.tag}.relaxation_metrics.summary.json"
    markdown_path = args.output_dir / f"{args.tag}.relaxation_metrics.md"
    atomic_text(summary_path, json.dumps(summary, indent=2, ensure_ascii=False))
    atomic_text(markdown_path, summary_markdown(args.tag, summary))
    print(json.dumps({
        "details": str(details_path.resolve()),
        "summary": str(summary_path.resolve()),
        "markdown": str(markdown_path.resolve()),
        "counts": summary["counts"],
        "timed_out_crystals": timeout_count,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
