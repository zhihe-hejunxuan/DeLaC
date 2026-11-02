#!/usr/bin/env python3
"""SUN evaluator with a killable timeout for each chemical-system hull."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "Energy_evaluation"))

from evaluate_sun_from_csv import load_structures_and_etot  # noqa: E402
from evaluation.reference.presets import ReferenceMP2020Correction
from evaluation.metrics.evaluator import MetricsEvaluator  # noqa: E402
from evaluation.metrics.structure import StructureMetricsCapability  # noqa: E402
from evaluation.utils.structure_matcher import DefaultDisorderedStructureMatcher  # noqa: E402

_ENERGY_CAPABILITY = None


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def hull_child(chemsys: str, queue) -> None:
    started = time.perf_counter()
    try:
        values = _ENERGY_CAPABILITY._get_energy_above_hull_per_atom_chemsys(chemsys)
        queue.put({"status": "completed", "elapsed_seconds": time.perf_counter() - started,
                   "values": [float(value) for value in values]})
    except BaseException as exc:
        queue.put({"status": "failed", "elapsed_seconds": time.perf_counter() - started,
                   "error": repr(exc)})


def run_one_hull(capability, chemsys: str, timeout_seconds: float) -> dict:
    global _ENERGY_CAPABILITY
    _ENERGY_CAPABILITY = capability
    context = mp.get_context("fork")
    queue = context.Queue()
    process = context.Process(target=hull_child, args=(chemsys, queue), daemon=True)
    started = time.perf_counter()
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(10)
        if process.is_alive():
            process.kill()
            process.join(10)
        queue.close()
        return {"status": "timeout", "elapsed_seconds": time.perf_counter() - started,
                "timeout_seconds": timeout_seconds}
    result = queue.get() if not queue.empty() else {
        "status": "failed", "elapsed_seconds": time.perf_counter() - started,
        "error": f"child exited without result (exit_code={process.exitcode})",
    }
    queue.close()
    return result


def material_record(entry, valid_indices: list[int], source: pd.DataFrame) -> dict:
    valid_position = int(entry.entry_id)
    source_row = int(valid_indices[valid_position])
    row = source.iloc[source_row]
    return {"entry_id": valid_position, "source_row_index": source_row,
            "candidate_id": str(row.get("candidate_id", "")),
            "request_id": str(row.get("request_id", "")),
            "formula": entry.composition.reduced_formula,
            "num_atoms": float(entry.composition.num_atoms)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-chemsys-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--child-failure-retries", type=int, default=2)
    parser.add_argument("--requested-count", type=int, default=1000)
    args = parser.parse_args()
    if args.per_chemsys_timeout_seconds <= 0 or args.requested_count <= 0 or args.child_failure_retries < 0:
        parser.error("timeouts and requested count must be positive")

    csv_path, output = args.csv.resolve(), args.output_dir.resolve()
    if output.exists() and any(output.iterdir()): raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    audit_path = output / "chemsys_hull_audit.csv"
    anomalies_path = output / "skipped_hull_anomalies.json"
    labels_path = output / "sun_relaxed_labels.csv"
    metrics_path = output / "sun_relaxed.json"
    status = {"status": "loading", "started_at": now(), "source_csv": str(csv_path),
              "requested_count": args.requested_count,
              "per_chemsys_timeout_seconds": args.per_chemsys_timeout_seconds,
              "child_failure_retries": args.child_failure_retries,
              "global_timeout_seconds": None, "energy_column": "E_tot_pred_dpa_raw",
              "energy_correction": "MP2020 exactly once"}
    atomic_json(status_path, status)

    structures, energies, valid_indices, source = load_structures_and_etot(
        str(csv_path), "E_tot_pred_dpa_raw", "cif_str")
    reference = ReferenceMP2020Correction()
    evaluator = MetricsEvaluator.from_structures_and_energies(
        structures=structures, energies=energies, reference=reference,
        structure_matcher=DefaultDisorderedStructureMatcher())
    energy_capability = evaluator.energy_capability
    items = list(energy_capability.dataset.entries_by_chemsys.items())
    energy_above_hull = np.full(len(structures), np.nan, dtype=float)
    audit_records, anomalies = [], []
    status.update(status="computing_hulls", parsed_rows=len(structures), distinct_chemsys=len(items))
    atomic_json(status_path, status)

    for position, (chemsys, entries) in enumerate(items, start=1):
        result = run_one_hull(energy_capability, chemsys, args.per_chemsys_timeout_seconds)
        attempts = 1
        # A fast non-zero child exit is an execution/isolation failure, not a
        # scientific reason to reject the material. Retry it; a genuine hull
        # timeout is never retried and remains capped at exactly 15 minutes.
        while result["status"] == "failed" and attempts <= args.child_failure_retries:
            attempts += 1
            result = run_one_hull(energy_capability, chemsys, args.per_chemsys_timeout_seconds)
        record = {"position_1_based": position, "chemsys": chemsys,
                  "n_elements": len(chemsys.split("-")), "n_materials": len(entries),
                  "status": result["status"], "elapsed_seconds": result.get("elapsed_seconds"),
                  "attempts": attempts, "error": result.get("error", "")}
        audit_records.append(record)
        if result["status"] == "completed":
            values = result["values"]
            if len(values) != len(entries):
                raise RuntimeError(f"result length mismatch for {chemsys}")
            for entry, value in zip(entries, values):
                energy_above_hull[int(entry.entry_id)] = float(value)
        else:
            anomalies.append({**record, "materials": [material_record(e, valid_indices, source) for e in entries]})
        pd.DataFrame(audit_records).to_csv(audit_path, index=False)
        atomic_json(anomalies_path, {"anomalies": anomalies})
        status.update(completed_chemsys=position, skipped_chemsys=len(anomalies),
                      skipped_materials=sum(item["n_materials"] for item in anomalies),
                      last_chemsys=chemsys, last_chemsys_status=result["status"], updated_at=now())
        atomic_json(status_path, status)
        print(f"[HULL] {position}/{len(items)} {chemsys} status={result['status']} "
              f"seconds={result.get('elapsed_seconds', float('nan')):.3f}", flush=True)

    retained_positions = np.flatnonzero(np.isfinite(energy_above_hull)).tolist()
    if not retained_positions:
        raise RuntimeError("all materials were skipped during hull evaluation")
    retained_summaries = [energy_capability._structure_summaries[i] for i in retained_positions]
    structure_capability = StructureMetricsCapability(
        structure_summaries=retained_summaries, reference_dataset=reference,
        structure_matcher=DefaultDisorderedStructureMatcher())
    status.update(status="computing_structure_labels", retained_materials=len(retained_positions), updated_at=now())
    atomic_json(status_path, status)
    is_unique = np.asarray(structure_capability.is_unique, dtype=bool)
    is_novel = np.asarray(structure_capability.is_novel, dtype=bool)
    retained_ehull = energy_above_hull[retained_positions]
    is_stable = retained_ehull <= float(energy_capability.stability_threshold)
    is_sun = is_stable & is_unique & is_novel

    label_rows = []
    for local_index, valid_position in enumerate(retained_positions):
        source_row = int(valid_indices[valid_position])
        label_rows.append({**source.iloc[source_row].to_dict(), "original_index": source_row,
                           "energy_above_hull": float(retained_ehull[local_index]),
                           "stable": bool(is_stable[local_index]), "unique": bool(is_unique[local_index]),
                           "novel": bool(is_novel[local_index]), "SUN": bool(is_sun[local_index])})
    pd.DataFrame(label_rows).to_csv(labels_path, index=False)
    sun_count = int(is_sun.sum())
    metrics = {"sun_count": sun_count, "stable_count": int(is_stable.sum()),
               "unique_count": int(is_unique.sum()), "novel_count": int(is_novel.sum()),
               "parsed_materials_before_hull_timeout": len(structures),
               "retained_materials_after_hull_timeout": len(retained_positions),
               "skipped_hull_materials": len(structures) - len(retained_positions),
               "sun_percent_of_retained": 100.0 * sun_count / len(retained_positions),
               "sun_percent_of_requested": 100.0 * sun_count / args.requested_count,
               "stability_threshold_eV_per_atom": float(energy_capability.stability_threshold),
               "per_chemsys_timeout_seconds": args.per_chemsys_timeout_seconds,
               "child_failure_retries": args.child_failure_retries,
               "anomalies": anomalies}
    atomic_json(metrics_path, metrics)
    status.update(status="completed", finished_at=now(), metrics=metrics,
                  metrics_json=str(metrics_path), labels_csv=str(labels_path),
                  hull_audit_csv=str(audit_path), anomalies_json=str(anomalies_path))
    atomic_json(status_path, status)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
